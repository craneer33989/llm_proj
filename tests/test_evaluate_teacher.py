from copy import deepcopy

import json
import pytest
from finance_agent.deepseek_trajectories import atomic_json
from finance_agent.evaluate_teacher import load_completed, metric_definition_digest, metrics, process_checks, wilson
from finance_agent.evaluate_trajectories import rollout
from finance_agent.expand_sft import read_jsonl
from finance_agent.trajectory_sources import OUTPUT
from finance_agent.report_teacher_eval import supplemental_metrics


def cases():
    return read_jsonl(OUTPUT / 'test-questions.jsonl')


def test_reference_rollouts_pass_strict_process_checks():
    for case in cases():
        row = rollout(case, 'reference')
        assert process_checks(case, row)['strict_passed']


def test_correct_final_answer_cannot_hide_early_dependent_search():
    case = next(c for c in cases() if c['group'] == 'conditional_bridge')
    row = rollout(case, 'reference')
    index = next(i for i, a in enumerate(row['actions']) if a.get('period') == 'H1')
    action, observation = row['actions'].pop(index), row['observations'].pop(index)
    row['actions'].insert(0, action)
    row['observations'].insert(0, observation)
    assert row['checks']['passed']  # Legacy result alone would hide the ordering error.
    assert not process_checks(case, row)['strict_passed']


def test_unsearched_cutoff_abstention_does_not_pass_strict_check():
    case = next(c for c in cases() if c['group'] == 'cutoff')
    row = rollout(case, 'reference')
    row['actions'] = [row['actions'][-1]]
    row['observations'] = []
    assert not process_checks(case, row)['cutoff_scope_searched']


def test_failed_numeric_tasks_stay_in_denominator_and_noop_calculations_do_not():
    numeric = next(c for c in cases() if c['group'] == 'ratio')
    abstention = next(c for c in cases() if c['group'] == 'cutoff')
    rows = [rollout(c, 'reference') for c in [numeric, abstention]]
    for c, row in zip([numeric, abstention], rows):
        row['checks'].update(process_checks(c, row))
        row['generations'] = [{} for _ in row['actions']]
    rows[0]['checks']['numeric_correct'] = False
    rows[0]['checks']['strict_passed'] = False
    result = metrics(rows, [numeric, abstention])
    assert result['numeric_exact_with_unit'] == {'correct': 0, 'total': 1, 'rate': 0}
    assert result['required_calculation_correct']['total'] == 1
    assert result['strict_pass_at_1']['rate'] == .5
    assert result['evidence_recall_micro']['rate'] == 1


def test_wilson_handles_all_pass_and_all_fail():
    assert 0 <= wilson(0, 120)[0] < wilson(0, 120)[1] < 1
    assert 0 < wilson(120, 120)[0] < wilson(120, 120)[1] <= 1


def test_resume_replays_saved_outputs_and_rejects_tampered_checks(tmp_path):
    case = cases()[0]
    row = rollout(case, 'reference')
    row['generations'] = [{'raw': json.dumps(a)} for a in row['actions']]
    row['checks']['valid_actions'] = True
    row['checks'].update(process_checks(case, row))
    path = tmp_path/'episodes'/f'001-{case["id"]}.json'
    atomic_json(path, row)
    assert len(load_completed(tmp_path, [case], OUTPUT)) == 1
    row['checks']['strict_passed'] = False
    atomic_json(path, row)
    with pytest.raises(ValueError, match='Saved episode changed'):
        load_completed(tmp_path, [case], OUTPUT)


def test_resume_metric_fingerprint_ignores_plumbing_but_not_grading():
    from pathlib import Path
    source = Path('finance_agent/evaluate_teacher.py').read_text(encoding='utf-8')
    assert metric_definition_digest(source) == metric_definition_digest(source+'\n# operational comment\n')
    assert metric_definition_digest(source) != metric_definition_digest(source.replace('z = 1.959963984540054', 'z = 1'))


def test_status_errors_and_unfinished_answers_affect_f1_without_hiding_valid_citations():
    test_cases = [{'expected': {'status': s, 'citations': []}}
                  for s in ['answered', 'answered', 'incomparable', 'insufficient']]
    rows = [{'checks': {'finished': True}, 'actions': [{'status': 'answered', 'citations': []}], 'observations': []}
            for _ in range(3)]
    rows.append({'checks': {'finished': False}, 'actions': [], 'observations': []})
    for row in rows:
        row['checks'].update(status_correct=False, numeric_correct=False)
    result = supplemental_metrics(rows, test_cases)
    assert result['status_classification']['answered']['precision'] == 2/3
    assert result['status_classification']['answered']['recall'] == 1
    assert result['status_classification']['answered']['f1'] == .8
    assert result['status_classification']['incomparable']['recall'] == 0
    assert result['status_macro_f1'] == pytest.approx(.8/3)
    assert result['citation_set_exact_independent_of_status']['correct'] == 3
