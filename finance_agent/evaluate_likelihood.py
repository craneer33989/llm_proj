"""Held-out completion/EOS token NLL, perplexity and teacher-forced accuracy."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
from time import perf_counter

from .deepseek_trajectories import atomic_json, run_lock
from .expand_sft import read_jsonl
from .prepare_sft import sha256
from .trajectory_sources import OUTPUT


def encode_reference(tokenizer, row):
    if row['split'] not in ('test', 'validation') or len(row['completion']) != 1 or row['completion'][0]['role'] != 'assistant':
        raise ValueError('Expected one held-out reference action')
    rendered = tokenizer.apply_chat_template(row['prompt'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tokenizer(rendered, add_special_tokens=False).input_ids
    target = tokenizer(row['completion'][0]['content'], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    if not prefix or not target or tokenizer.eos_token_id is None:
        raise ValueError('Missing prefix, target or EOS')
    return prefix + target, target


def aggregate(rows):
    tokens = sum(r['target_tokens'] for r in rows)
    nll = sum(r['nll_sum'] for r in rows)
    correct = sum(r['correct_tokens'] for r in rows)
    loss = nll/tokens
    return {'actions': len(rows), 'target_tokens_including_eos': tokens,
            'cross_entropy_nats_per_token': loss, 'perplexity': math.exp(loss),
            'teacher_forced_token_accuracy': correct/tokens, 'correct_tokens': correct,
            'mean_action_loss': sum(r['nll_sum']/r['target_tokens'] for r in rows)/len(rows)}


def load_prefix(path, rows, encoded):
    records = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    if len(records) > len(rows):
        raise ValueError('Too many saved likelihood records')
    for saved,row,(_,target) in zip(records,rows,encoded):
        if any(saved[k] != row[k] for k in ('case_id','turn','group','action_type')) or saved['target_tokens'] != len(target):
            raise ValueError('Saved likelihood order/tokenization changed')
        if not math.isfinite(saved['nll_sum']) or saved['nll_sum'] < 0 or not 0 <= saved['correct_tokens'] <= len(target):
            raise ValueError('Invalid saved likelihood values')
    return records


def evaluate(snapshot, adapter, output, data=OUTPUT, split='test', resume=False):
    if (output/'config.json').exists() and not resume:
        raise ValueError('Use a new output directory')
    if resume and ((output/'report.json').exists() or not (output/'config.json').exists()):
        raise ValueError('Resume requires an incomplete likelihood run')
    start = perf_counter()
    with run_lock(output):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this BF16 comparison')
        manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
        filename = f'{split}-actions.jsonl'
        if sha256(data/filename) != manifest['files_sha256'][filename]:
            raise ValueError('Held-out reference actions changed')
        config = {'snapshot': str(snapshot), 'adapter': str(adapter), 'split': split,
                  'adapter_sha256': sha256(adapter/'adapter_model.safetensors'),
                  'dataset_manifest_sha256': sha256(data/'manifest.json'),
                  'reference_actions_sha256': sha256(data/filename), 'source_sha256': sha256(Path(__file__)),
                  'loss_scope': 'Only reference completion tokens and EOS; all prompt tokens excluded.',
                  'normalization': 'Sum of token NLL divided by all target tokens; no per-action weighting.',
                  'evaluation_mode': 'Teacher-forced reference trajectory likelihood; separate from blind model rollouts.',
                  'dtype': 'bfloat16', 'attention': 'sdpa', 'use_cache': False}
        previous_seconds = 0
        if resume:
            if json.loads((output/'config.json').read_text(encoding='utf-8')) != config:
                raise ValueError('Likelihood resume inputs/settings/source changed')
            status_path = output/'status.json'
            previous_seconds = json.loads(status_path.read_text(encoding='utf-8')).get('elapsed_seconds',0) if status_path.exists() else 0
        else:
            atomic_json(output/'config.json', config)
            (output/'evaluator-source.py').write_bytes(Path(__file__).read_bytes())
        try:
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            rows = read_jsonl(data/filename)
            encoded = [encode_reference(tokenizer, r) for r in rows]
            base = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
            model = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
            if max(len(s) for s, _ in encoded) > model.config.max_position_embeddings:
                raise ValueError('Reference exceeds native context; no truncation allowed')
            results, resumed_counts = {}, {}
            for mode in ('sft', 'base'):
                records = load_prefix(output/f'{mode}-actions.json',rows,encoded) if resume else []
                completed = len(records)
                resumed_counts[mode] = completed
                context = model.disable_adapter() if mode == 'base' else nullcontext()
                with context, torch.inference_mode():
                    for i, (row, (sequence, target)) in enumerate(zip(rows, encoded), 1):
                        if i <= completed:
                            continue
                        ids = torch.tensor([sequence], device='cuda')
                        logits = model(input_ids=ids, logits_to_keep=len(target)+1, use_cache=False).logits[0, :-1].float()
                        labels = torch.tensor(target, device='cuda')
                        nll = torch.nn.functional.cross_entropy(logits, labels, reduction='sum')
                        records.append({'case_id': row['case_id'], 'turn': row['turn'], 'group': row['group'],
                            'action_type': row['action_type'], 'target_tokens': len(target),
                            'nll_sum': nll.item(), 'correct_tokens': (logits.argmax(-1)==labels).sum().item()})
                        del ids, logits, labels, nll
                        if i % 20 == 0 or i == len(rows):
                            state = {'state': 'evaluating', 'mode': mode, 'completed_actions': i,
                                     'total_actions': len(rows), 'elapsed_seconds': previous_seconds+perf_counter()-start}
                            atomic_json(output/'status.json', state)
                            atomic_json(output/f'{mode}-actions.json', records)
                            print(json.dumps(state), flush=True)
                groups = defaultdict(list)
                for r in records:
                    groups[r['group']].append(r)
                results[mode] = {**aggregate(records), 'by_group': {g: aggregate(v) for g, v in groups.items()}}
                atomic_json(output/f'{mode}-metrics.json', results[mode])
            report = {'config': config, 'metrics': results, 'max_sequence_tokens': max(len(s) for s, _ in encoded),
                      'total_wall_seconds': previous_seconds+perf_counter()-start, 'resumed_actions':resumed_counts,
                      'notes': ['No parameters are updated; adapter is enabled for SFT and disabled for the identical base model.',
                        'PPL is exp(token-weighted held-out completion cross entropy), including EOS, not exp(mean training loss).',
                        'Token accuracy uses the gold preceding tokens, and is not free-generation answer accuracy.',
                        'Equivalent search wording or JSON order may receive different likelihood; judge task success separately.']}
            atomic_json(output/'report.json', report)
            atomic_json(output/'status.json', {'state': 'complete'})
            print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
            return report
        except BaseException as error:
            atomic_json(output/'status.json', {'state': 'failed', 'error': str(error),'elapsed_seconds':previous_seconds+perf_counter()-start})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['test', 'validation'], default='test')
    parser.add_argument('--data', type=Path, default=OUTPUT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    evaluate(args.snapshot.resolve(), args.adapter.resolve(), args.output.resolve(), data=args.data.resolve(),split=args.split,resume=args.resume)
