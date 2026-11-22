"""Persist blind model rollouts and report metrics with explicit denominators."""
import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

from .deepseek_trajectories import atomic_json, run_lock
from .evaluate_trajectories import model_generator, rollout, summarize
from .expand_sft import read_jsonl
from .prepare_sft import sha256
from .trajectory_sources import OUTPUT


def metric_definition_digest(source):
    names = {'rate', 'wilson', 'process_checks', 'percentile', 'metrics'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Missing metric definitions')
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[])).encode()).hexdigest()


def load_completed(output, cases, data):
    rows = []
    for i, case in enumerate(cases, 1):
        path = output/'episodes'/f'{i:03d}-{case["id"]}.json'
        if not path.exists():
            break
        row = json.loads(path.read_text(encoding='utf-8'))
        if row['case_id'] != case['id']:
            raise ValueError('Saved episode order changed')
        raw = iter(g['raw'] for g in row['generations'])
        replay = rollout(case, 'model', data, lambda _: next(raw))
        replay['checks'].update(process_checks(case, replay))
        for field in ('actions', 'observations', 'invalid', 'checks'):
            if row[field] != replay[field]:
                raise ValueError(f'Saved episode changed: {case["id"]}: {field}')
        if next(raw, None) is not None:
            raise ValueError('Unused saved generation')
        rows.append(row)
    if len(list((output/'episodes').glob('*.json'))) != len(rows):
        raise ValueError('Saved episodes are not a contiguous prefix')
    return rows


def rate(correct, total):
    return {"correct": correct, "total": total, "rate": correct / total if total else None}


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p, d = correct / total, 1 + z*z / total
    center = (p + z*z / (2*total)) / d
    margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / d
    return [center-margin, center+margin]


def process_checks(case, row):
    dependent_ok, scope_ok = None, None
    if case['group'] == 'conditional_bridge':
        selected_year, saw_dependent, dependent_ok = None, False, True
        # Observations correspond to successfully executed, non-final actions.
        for action, observation in zip(row['actions'], row['observations']):
            if action.get('tool') == 'search' and action.get('period') == 'H1':
                saw_dependent = True
                dependent_ok &= selected_year is not None and action.get('year') == selected_year
            if observation.get('operation') == 'argmax':
                selected_year = observation['selected_year']
        dependent_ok = bool(dependent_ok and saw_dependent and selected_year is not None)
    if case['group'] == 'cutoff':
        scope_ok = any(a.get('tool') == 'search' and
            f"{a.get('company_code')}-{a.get('year')}-{a.get('period')}" in case['required_reports']
            for a in row['actions'][:len(row['observations'])])
    return {'conditional_dependency_correct': dependent_ok, 'cutoff_scope_searched': scope_ok,
            'strict_passed': bool(row['checks']['passed'] and dependent_ok is not False and scope_ok is not False)}


def percentile(values, p):
    ordered = sorted(values)
    point = (len(ordered)-1)*p
    low = math.floor(point)
    return ordered[low] + (ordered[math.ceil(point)]-ordered[low])*(point-low)


def metrics(rows, cases):
    by_id = {c['id']: c for c in cases}
    n = len(rows)
    out = {}
    for name, field in [('strict_pass_at_1', 'strict_passed'), ('legacy_pass_at_1', 'passed'),
                        ('finished', 'finished'), ('status_correct', 'status_correct'),
                        ('citations_correct', 'citations_correct'), ('valid_episodes', 'valid_actions'),
                        ('within_budget', 'budget_ok')]:
        count = sum(bool(r['checks'][field]) for r in rows)
        out[name] = rate(count, n)
    out['strict_pass_at_1']['wilson_95_ci'] = wilson(out['strict_pass_at_1']['correct'], n)
    numeric = [r for r in rows if by_id[r['case_id']]['expected']['status'] == 'answered']
    calculations = [r for r in rows if by_id[r['case_id']].get('calculation')]
    out['numeric_exact_with_unit'] = rate(sum(r['checks']['numeric_correct'] is True for r in numeric), len(numeric))
    out['required_calculation_correct'] = rate(sum(r['checks']['calculation_correct'] for r in calculations), len(calculations))
    for field in ('conditional_dependency_correct', 'cutoff_scope_searched'):
        applicable = [r for r in rows if r['checks'][field] is not None]
        out[field] = rate(sum(r['checks'][field] for r in applicable), len(applicable))
    applicable = [r for r in rows if r['checks']['evidence_recall'] is not None]
    out['evidence_recall_macro'] = mean(r['checks']['evidence_recall'] for r in applicable) if applicable else None
    recovered, required_count = 0, 0
    for row in rows:
        required = set(by_id[row['case_id']]['expected']['citations'])
        found = {h['id'] for obs in row['observations'] if obs.get('tool') == 'search' for h in obs['hits']}
        recovered += len(required & found)
        required_count += len(required)
    out['evidence_recall_micro'] = rate(recovered, required_count)
    calls = sum(len(r.get('generations', [])) for r in rows)
    invalid = sum(len(r['invalid']) for r in rows)
    out['valid_model_actions'] = rate(calls-invalid, calls) if calls else None
    out['premature_stop_episodes'] = sum(r['checks']['premature_stop'] for r in rows)
    out['extra_search_episodes'] = sum(r['checks']['extra_searches'] > 0 for r in rows)
    out['extra_searches'] = sum(r['checks']['extra_searches'] for r in rows)
    out['average_search_attempts'] = mean(r['checks']['searches'] for r in rows)
    out['average_executed_calculations'] = mean(sum(o.get('tool') == 'calculate' for o in r['observations']) for r in rows)
    out['average_model_calls'] = calls/n
    durations = [r['elapsed_seconds'] for r in rows]
    out['episode_latency_seconds'] = {'sum': sum(durations), 'mean': mean(durations),
        'median': percentile(durations, .5), 'p95': percentile(durations, .95)}
    groups = sorted({r['group'] for r in rows})
    out['by_group'] = {g: rate(sum(r['checks']['strict_passed'] for r in rows if r['group'] == g),
                                    sum(r['group'] == g for r in rows)) for g in groups}
    out['macro_group_pass_rate'] = mean(v['rate'] for v in out['by_group'].values())
    out['by_expected_status'] = {s: rate(sum(r['checks']['strict_passed'] for r in rows if by_id[r['case_id']]['expected']['status'] == s),
        sum(by_id[r['case_id']]['expected']['status'] == s for r in rows))
        for s in sorted({by_id[r['case_id']]['expected']['status'] for r in rows})}
    return out


def evaluate(snapshot, adapter, output, data=OUTPUT, split='test', resume=False, empty_cache=False):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'config.json').exists() and not resume:
        raise ValueError('Use a new evaluation directory; existing results will not be overwritten')
    if resume and ((output/'report.json').exists() or not (output/'config.json').exists()):
        raise ValueError('Resume requires an incomplete evaluation with a config')
    start = perf_counter()
    with run_lock(output):
        manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
        for name in (f'{split}-questions.jsonl', 'sources.json', 'corpus.sqlite3'):
            if sha256(data/name) != manifest['files_sha256'][name]:
                raise ValueError(f'Evaluation input changed: {name}')
        cases = read_jsonl(data/f'{split}-questions.jsonl')
        config = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'split': split,
            'snapshot': str(snapshot), 'adapter': str(adapter) if adapter else None,
            'adapter_sha256': sha256(adapter/'adapter_model.safetensors') if adapter else None,
            'dataset_manifest_sha256': sha256(data/'manifest.json'), 'cases': len(cases),
            'protocol': manifest['protocol'], 'do_sample': False, 'max_new_tokens': 384,
            'gold_visible': False, 'rule_fallback': False, 'enable_thinking': False,
            'allocator_cleanup_after_call': empty_cache,
            'evaluator_sha256': sha256(Path(__file__)),
            'rollout_source_sha256': sha256(Path(__file__).with_name('evaluate_trajectories.py')),
            'scorer_sha256': sha256(Path(__file__).with_name('trajectory_runtime.py')),
            'versions': {k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'peft')}}
        previous_seconds = 0
        rows = []
        if resume:
            original = json.loads((output/'config.json').read_text(encoding='utf-8'))
            for key in config:
                old_value = original.get(key, False) if key == 'allocator_cleanup_after_call' else original.get(key)
                if key not in ('started_at_utc', 'evaluator_sha256') and config[key] != old_value:
                    raise ValueError(f'Resume configuration changed: {key}')
            old_source = (output/'evaluator-source.py').read_text(encoding='utf-8')
            if sha256(output/'evaluator-source.py') != original['evaluator_sha256']:
                raise ValueError('Original evaluator snapshot changed')
            if metric_definition_digest(old_source) != metric_definition_digest(Path(__file__).read_text(encoding='utf-8')):
                raise ValueError('Metric definitions changed')
            rows = load_completed(output, cases, data)
            previous_status = json.loads((output/'status.json').read_text(encoding='utf-8'))
            previous_seconds = previous_status.get('elapsed_seconds', sum(r['elapsed_seconds'] for r in rows))
            invocation = {'resumed_at_utc': config['started_at_utc'], 'resumed_from_case': len(rows),
                'evaluator_sha256': config['evaluator_sha256'], 'previous_recorded_seconds': previous_seconds}
            stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
            atomic_json(output/f'resume-{stamp}.json', invocation)
            atomic_text_path = output/f'evaluator-resume-{stamp}.py'
            atomic_text_path.write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
            config = original
        else:
            atomic_json(output/'config.json', config)
            (output/'evaluator-source.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
        completed = len(rows)
        atomic_json(output/'status.json', {'state': 'loading', 'completed': completed, 'total': len(cases),
                                           'elapsed_seconds': previous_seconds})
        try:
            reference_rows = []
            for case in cases:
                row = rollout(case, 'reference', data)
                row['checks'].update(process_checks(case, row))
                reference_rows.append(row)
            if not all(r['checks']['strict_passed'] for r in reference_rows):
                raise ValueError('Reference replay preflight failed')
            atomic_json(output/'preflight.json', {'policy': 'reference', 'model_performance': False,
                'cases': len(cases), 'strict_passed': len(cases)})
            import torch
            torch.manual_seed(42)
            generate = model_generator(snapshot, adapter)
            torch.cuda.reset_peak_memory_stats()
            generations = []
            def measured(messages):
                before = perf_counter()
                raw = generate(messages)
                if empty_cache:
                    torch.cuda.empty_cache()
                generations.append({'raw': raw, 'seconds': perf_counter()-before})
                return raw
            for i, case in enumerate(cases, 1):
                if i <= completed:
                    continue
                generations = []
                row = rollout(case, 'model', data, measured)
                row['generations'] = generations
                row['checks'].update(process_checks(case, row))
                rows.append(row)
                atomic_json(output/'episodes'/f'{i:03d}-{case["id"]}.json', row)
                state = {'state': 'evaluating', 'completed': i, 'total': len(cases),
                    'strict_passed': sum(r['checks']['strict_passed'] for r in rows),
                    'last_case': case['id'], 'elapsed_seconds': previous_seconds+perf_counter()-start}
                atomic_json(output/'status.json', state)
                print(json.dumps(state), flush=True)
            report = {'config': config, 'summary': summarize(rows), 'metrics': metrics(rows, cases),
                'total_wall_seconds': previous_seconds+perf_counter()-start,
                'time_scope': 'Recorded active time through committed episodes plus current invocation; excludes offline gap and uncommitted interrupted work.',
                'resumed_from_case': completed,
                'metric_definition_sha256': metric_definition_digest(Path(__file__).read_text(encoding='utf-8')),
                'peak_memory_scope': 'latest_invocation_only' if resume else 'full_invocation',
                'peak_process_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'peak_process_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                'metric_notes': ['Strict pass adds causal argmax-before-dependent-search and cutoff-scope checks to the unchanged legacy scorer.',
                    'Numeric exact match includes unit, denominator is all answered tasks, including failures.',
                    'Calculation metric denominator includes only tasks requiring a calculator operation.',
                    'Citations metric requires correct answer status and the exact required evidence set.',
                    'Evidence macro recall excludes empty gold sets; micro recall counts evidence occurrences per task.',
                    'Calls include invalid model outputs; one deterministic rollout per question; no retries or oracle repair.',
                    'Wilson interval treats questions as independent; shared reports/templates limit this assumption.'],
                'rows': rows}
            atomic_json(output/'report.json', report)
            atomic_json(output/'status.json', {'state': 'complete', 'completed': len(rows), 'total': len(cases)})
            print(json.dumps(report['metrics'], ensure_ascii=False, indent=2), flush=True)
            return report
        except BaseException as error:
            atomic_json(output/'status.json', {'state': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=OUTPUT)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--empty-cache', action='store_true', help='Release unused allocator cache after each generation; does not change decoding')
    args = parser.parse_args()
    evaluate(args.snapshot.resolve(), args.adapter.resolve() if args.adapter else None,
             args.output.resolve(), args.data.resolve(), args.split, args.resume, args.empty_cache)
