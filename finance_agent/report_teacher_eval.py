"""Build paired, auditable base/SFT test metrics from completed evaluations."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .deepseek_trajectories import atomic_json, atomic_text
from .evaluate_teacher import metrics, rate
from .expand_sft import read_jsonl
from .prepare_sft import sha256
from .trajectory_sources import OUTPUT

GROUPS = {'direct': '单跳取值', 'cross_document': '跨报告差值', 'cross_company': '跨公司比较',
    'cutoff': '披露日期限制', 'ratio': '比率计算', 'three_document': '三报告汇总',
    'conditional_bridge': '比较后条件检索', 'yoy': '同比计算', 'incomparable': '不可比口径识别'}


def supplemental_metrics(rows, cases):
    labels = ('answered', 'insufficient', 'incomparable')
    pairs, exact_citations = [], 0
    for row, case in zip(rows, cases):
        final = row['actions'][-1] if row['checks']['finished'] else {}
        pairs.append((case['expected']['status'], final.get('status', 'unfinished_or_invalid')))
        observed = {hit['id'] for obs in row['observations'] if obs.get('tool') == 'search' for hit in obs['hits']}
        expected = set(case['expected']['citations'])
        exact_citations += bool(row['checks']['finished'] and set(final.get('citations', [])) == expected and expected <= observed)
    classification = {}
    for label in labels:
        tp = sum(e == label and p == label for e, p in pairs)
        support = sum(e == label for e, _ in pairs)
        predicted = sum(p == label for _, p in pairs)
        precision = tp/predicted if predicted else 0.0
        recall = tp/support if support else 0.0
        f1 = 2*tp/(support+predicted) if support+predicted else 0.0
        classification[label] = {'support': support, 'predicted': predicted, 'true_positive': tp,
                                 'precision': precision, 'recall': recall, 'f1': f1}
    answer_matches = sum(r['checks']['status_correct'] and r['checks']['numeric_correct'] is not False for r in rows)
    support_total = sum(v['support'] for v in classification.values())
    raw_outputs = [g['raw'] for r in rows for g in r.get('generations', [])]
    json_valid, object_valid = 0, 0
    for raw in raw_outputs:
        try:
            parsed = json.loads(raw.strip())
            json_valid += 1
            object_valid += isinstance(parsed, dict)
        except (ValueError, TypeError):
            pass
    return {'answer_exact_match': rate(answer_matches, len(rows)),
            'json_syntax_valid': rate(json_valid, len(raw_outputs)),
            'json_object_valid': rate(object_valid, len(raw_outputs)),
            'citation_set_exact_independent_of_status': rate(exact_citations, len(rows)),
            'status_classification': classification,
            'status_macro_precision': sum(v['precision'] for v in classification.values())/len(labels),
            'status_macro_recall': sum(v['recall'] for v in classification.values())/len(labels),
            'status_macro_f1': sum(v['f1'] for v in classification.values())/len(labels),
            'status_weighted_f1': sum(v['support']*v['f1'] for v in classification.values())/support_total}


def build(sft_dir, base_dir, output, data=OUTPUT, likelihood_dir=None, training_dir=None):
    reports = {name: json.loads((path/'report.json').read_text(encoding='utf-8'))
               for name, path in [('sft', sft_dir), ('base', base_dir)]}
    split = reports['sft']['config']['split']
    if split not in ('test', 'validation'):
        raise ValueError('Reports must use a held-out split')
    split_label = '测试集' if split == 'test' else '验证集'
    cases = read_jsonl(data/f'{split}-questions.jsonl')
    ids = [c['id'] for c in cases]
    for report in reports.values():
        assert [r['case_id'] for r in report['rows']] == ids
        assert report['config']['split'] == split
        assert report['config']['dataset_manifest_sha256'] == sha256(data/'manifest.json')
        assert report['metrics'] == metrics(report['rows'], cases)
    for field in ('snapshot', 'dataset_manifest_sha256', 'do_sample', 'max_new_tokens',
                  'enable_thinking', 'gold_visible', 'rule_fallback',
                  'rollout_source_sha256', 'scorer_sha256', 'versions'):
        assert reports['sft']['config'][field] == reports['base']['config'][field], field
    assert reports['sft']['metric_definition_sha256'] == reports['base']['metric_definition_sha256']
    assert reports['base']['config']['adapter'] is None
    assert reports['sft']['config']['adapter'] is not None
    paired = Counter()
    comparisons = []
    for s, b in zip(reports['sft']['rows'], reports['base']['rows']):
        sp, bp = s['checks']['strict_passed'], b['checks']['strict_passed']
        outcome = 'both_pass' if sp and bp else 'improved' if sp else 'regressed' if bp else 'both_fail'
        paired[outcome] += 1
        comparisons.append({'case_id': s['case_id'], 'group': s['group'], 'sft_pass': sp,
                            'base_pass': bp, 'outcome': outcome})
    details = {}
    for name, report in reports.items():
        confusion = Counter()
        reasons = Counter()
        for case, row in zip(cases, report['rows']):
            predicted = row['actions'][-1].get('status', 'unfinished') if row['checks']['finished'] else ('invalid' if row['invalid'] else 'unfinished')
            confusion[(case['expected']['status'], predicted)] += 1
            for error in row['invalid']:
                reasons[error['error']] += 1
        details[name] = {'status_confusion': [{'expected': e, 'predicted': p, 'count': v}
            for (e, p), v in sorted(confusion.items())], 'invalid_reasons': dict(reasons),
            'failures': [{'case_id': r['case_id'], 'group': r['group'], 'question': cases[i]['question'],
                'checks': r['checks'], 'invalid': r['invalid'], 'actions': r['actions']}
                for i, r in enumerate(report['rows']) if not r['checks']['strict_passed']]}
    result = {'test_cases': len(cases), 'evaluation_cases': len(cases), 'split': split, 'model_selection_on_test': False,
        'report_builder_sha256': sha256(Path(__file__)),
        'reports_sha256': {name: sha256(path/'report.json') for name, path in [('sft', sft_dir), ('base', base_dir)]},
        'metrics': {name: {**report['metrics'], **supplemental_metrics(report['rows'], cases)} for name, report in reports.items()},
        'paired': {k: paired[k] for k in ('both_pass', 'improved', 'regressed', 'both_fail')},
        'strict_pass_difference_percentage_points': 100*(reports['sft']['metrics']['strict_pass_at_1']['rate']-reports['base']['metrics']['strict_pass_at_1']['rate']),
        'details': details, 'cases': comparisons}
    if likelihood_dir:
        likelihood = json.loads((likelihood_dir/'report.json').read_text(encoding='utf-8'))
        assert likelihood['config']['split'] == split
        for field in ('snapshot', 'adapter_sha256', 'dataset_manifest_sha256'):
            assert likelihood['config'][field] == reports['sft']['config'][field]
        result['likelihood'] = likelihood['metrics']
        result['likelihood_report_sha256'] = sha256(likelihood_dir/'report.json')
    if training_dir:
        training = json.loads((training_dir/'report.json').read_text(encoding='utf-8'))
        assert training['adapter_sha256'] == reports['sft']['config']['adapter_sha256']
        losses = [r['loss'] for r in training['steps']]
        window = min(20, len(losses))
        result['training'] = {'optimizer_updates': training['completed_updates'],
            'action_examples_seen': training['action_examples_seen'],
            'loss_window_updates': window,
            'first_20_update_loss_mean': sum(losses[:window])/window,
            'last_20_update_loss_mean': sum(losses[-window:])/window,
            'final_update_loss': losses[-1], 'all_update_loss_mean': sum(losses)/len(losses),
            'loss_scope': training['config']['loss_scope'],
            'normalization_note': 'Training logs average per-action losses equally; held-out CE weights each target token equally.'}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'metrics.json', result)
    def display(value):
        if value['rate'] is None:
            return '不适用'
        return f"{value['correct']}/{value['total']}（{100*value['rate']:.2f}%）"
    sft, base = (result['metrics'][n] for n in ('sft', 'base'))
    steps = result.get('training', {}).get('optimizer_updates', '训练结束')
    lines = [f'# 完整{split_label}：教师 SFT 与原始基座', '',
        f'同一固定本地基座，SFT 使用第 {steps} 步最终 LoRA。{split_label}共 {len(cases)} 题，按公司与报告隔离；'
        '每题一次完整自主 rollout，贪心解码、关闭思考、每次最多生成 384 token，不提供 gold，不使用规则修复或参考动作兜底。', '',
        '| 指标 | SFT | 原始基座 |', '| --- | ---: | ---: |']
    for key, title in [('strict_pass_at_1', '完整轨迹严格通过率'), ('legacy_pass_at_1', '原评分器通过率'),
        ('answer_exact_match', '答案语义完全匹配 EM（状态、数值、单位）'),
        ('numeric_exact_with_unit', '数值与单位完全正确'), ('citations_correct', '引用与最终状态均正确'),
        ('citation_set_exact_independent_of_status', '引用集合完全正确（独立于状态）'),
        ('required_calculation_correct', '必要计算执行正确'), ('conditional_dependency_correct', '条件检索因果顺序正确'),
        ('cutoff_scope_searched', '截止日期题实际检索目标范围'), ('status_correct', '最终状态正确'),
        ('finished', '任务完成'), ('valid_episodes', '全程动作合法'), ('valid_model_actions', '单次模型动作合法'),
        ('json_object_valid', 'JSON 对象格式有效（不保证动作协议正确）'),
        ('within_budget', '未超出动作与检索预算'), ('evidence_recall_micro', '必要证据召回（逐题累计）')]:
        lines.append(f'| {title} | {display(sft[key])} | {display(base[key])} |')
    for key, title in [('average_search_attempts', '平均检索尝试/题'), ('average_executed_calculations', '平均已执行计算/题'),
                       ('average_model_calls', '平均模型调用/题'), ('premature_stop_episodes', '缺证据提前停止题数'),
                       ('extra_search_episodes', '证据齐全后继续检索题数')]:
        lines.append(f'| {title} | {sft[key]:.3f} | {base[key]:.3f} |')
    lines += ['', '## 各题型严格通过率', '', '| 题型 | SFT | 原始基座 |', '| --- | ---: | ---: |']
    for group in sorted(sft['by_group']):
        lines.append(f"| {GROUPS.get(group, group)} | {display(sft['by_group'][group])} | {display(base['by_group'][group])} |")
    lines += ['', f"{len(sft['by_group'])} 类题型宏平均通过率：SFT {100*sft['macro_group_pass_rate']:.2f}%，基座 {100*base['macro_group_pass_rate']:.2f}%。",
        '', '## 常用分类指标', '', '| 最终状态三分类指标 | SFT | 原始基座 |', '| --- | ---: | ---: |',
        *[f"| {label} | {sft[key]:.4f} | {base[key]:.4f} |" for key, label in
          [('status_macro_precision','Macro Precision'), ('status_macro_recall','Macro Recall'),
           ('status_macro_f1','Macro F1'), ('status_weighted_f1','Weighted F1')]], '',
        '状态分类 Accuracy 见上表“最终状态正确”；逐类 precision、recall、F1 见 metrics.json。分类对象是 answered／insufficient／incomparable；未完成／非法输出计作对应真实类别的漏判。'
        '这些 F1 反映状态判断，不能代替要求数值、引用和工具执行都正确的完整任务成功率。',
        '', f"配对结果：共同通过 {paired['both_pass']} 题，SFT 改善 {paired['improved']} 题，退步 {paired['regressed']} 题，共同失败 {paired['both_fail']} 题。"
        f"严格通过率变化 {result['strict_pass_difference_percentage_points']:+.2f} 个百分点。", '',
        ]
    if 'likelihood' in result:
        lines += [f'## {split_label} Loss、PPL 与 Token Accuracy', '', '| 指标 | SFT | 原始基座 |', '| --- | ---: | ---: |']
        for key, label in [('cross_entropy_nats_per_token', 'Cross-entropy Loss（nats/token，越低越好）'),
                           ('perplexity', 'Perplexity / PPL（越低越好）'),
                           ('teacher_forced_token_accuracy', '参考动作 Token Accuracy（越高越好）')]:
            lines.append(f"| {label} | {result['likelihood']['sft'][key]:.6f} | {result['likelihood']['base'][key]:.6f} |")
        lines += ['', f"共 {result['likelihood']['sft']['actions']} 条测试参考动作、"
            f"{result['likelihood']['sft']['target_tokens_including_eos']} 个监督 token（含 EOS）。"
            '提示和历史观察均不计 loss；交叉熵按全部目标 token 加权，PPL=exp(交叉熵)。'
            '此处使用标准轨迹前缀和标准前序 token，属于 teacher forcing，独立于前面的盲生成评测；'
            'Token Accuracy 不是自由生成答案正确率。等价检索措辞／JSON 顺序也会影响 PPL，不能只凭 PPL 判断任务能力。', '']
    if 'training' in result:
        tr = result['training']
        lines += ['## 本轮训练 Loss', '',
            f"完成 {tr['optimizer_updates']} 次更新、{tr['action_examples_seen']} 条动作；"
            f"前 {tr['loss_window_updates']} 步平均 loss={tr['first_20_update_loss_mean']:.6f}，末 {tr['loss_window_updates']} 步={tr['last_20_update_loss_mean']:.6f}，"
            f"最后一步={tr['final_update_loss']:.6f}。训练日志按动作等权，测试交叉熵按 token 加权，二者不作直接数值比较。", '']
    lines += ['## 失败与结果解释', '',
        f"SFT 失败 {len(details['sft']['failures'])} 题；下面保留问题与实际最终状态，完整动作见 metrics.json。", '']
    for failure in details['sft']['failures']:
        final = failure['actions'][-1] if failure['actions'] else {}
        lines += [f"- `{failure['case_id']}`：{failure['question']} 实际最终状态为 `{final.get('status', 'unfinished')}`。"]
    lines += ['', f"基座 {reports['base']['summary']['invalid_episodes']}/{len(cases)} 题出现非法动作；"
        f"其中 Unknown action 共 {details['base']['invalid_reasons'].get('Unknown action',0)} 次，另有 JSON 格式和无效数值等错误。"
        '该前后差异主要体现固定提示下的工具协议适配与任务执行能力，通用推理能力需要另外设计评测。', '']
    lines += ['## 速度、显存与口径', '', '| 项目 | SFT | 原始基座 |', '| --- | ---: | ---: |']
    for key, title in [('mean', '平均每题秒数'), ('median', '每题延迟中位数'), ('p95', '每题延迟 P95')]:
        lines.append(f"| {title} | {sft['episode_latency_seconds'][key]:.2f} | {base['episode_latency_seconds'][key]:.2f} |")
    for key, title, scale in [('total_wall_seconds', '已记录运行分钟（含初始化和预检）', 60),
                             ('peak_process_allocated_mib', '最后运行进程峰值分配 GiB', 1024),
                             ('peak_process_reserved_mib', '最后运行进程峰值预留 GiB', 1024)]:
        lines.append(f"| {title} | {reports['sft'][key]/scale:.2f} | {reports['base'][key]/scale:.2f} |")
    lo, hi = sft['strict_pass_at_1']['wilson_95_ci']
    lines += ['', f'SFT 严格通过率的 Wilson 95% 区间：{100*lo:.2f}%—{100*hi:.2f}%。题目共享报告与模板，不能将其当作完全独立样本的稳健泛化区间。', '',
        '严格通过要求数值、单位、状态、完整引用、必要计算、预算与停止行为全部正确，并额外要求条件选年题先取得 argmax 观察再按其结果检索，截止日期题确实检索目标报告。'
        f"数值指标分母为全部 {sft['numeric_exact_with_unit']['total']} 道可回答题，失败题保留在分母；计算指标只统计要求实际计算的题。引用正确也要求最终状态正确。", '',
        '该评测集由程序构题，训练与评测共享任务模板。成绩用于衡量当前工具协议下的财报任务能力，不能外推为开放域推理或跨领域能力。'
        '验证集可用于选参；根据最终测试集错误改训练后，应新增独立测试集。共享 GPU 负载影响速度。'
        f"SFT 从第 {reports['sft'].get('resumed_from_case', 0)} 题恢复；已记录时间不含停机间隔及中断时未提交的工作。"
        '显存峰值统计范围以原始报告的 peak_memory_scope 为准。'
        f"生成后缓存回收：基座={reports['base']['config'].get('allocator_cleanup_after_call',False)}，SFT={reports['sft']['config'].get('allocator_cleanup_after_call',False)}。"
        '缓存策略与 GPU 负载不同的运行不能用于严格比较模型速度。'
        '基座大量题目在首轮失败、实际执行动作更少，这也会缩短单题耗时。'
        '预留值是 PyTorch 缓存分配器统计，必须与整卡物理显存占用分开解读。', '',
        f'完整逐题 SFT 记录：[{sft_dir.name}]({sft_dir.as_posix()}/report.json)。',
        f'完整逐题基座记录：[{base_dir.name}]({base_dir.as_posix()}/report.json)。',
        '汇总、状态混淆、失败记录和逐题配对： [metrics.json](metrics.json)。', '']
    atomic_text(output/'README.md', '\n'.join(lines))
    print(json.dumps({'paired': result['paired'], 'sft': sft['strict_pass_at_1'],
                      'base': base['strict_pass_at_1'], 'output': str(output)}, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--likelihood', type=Path)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--data', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.sft.resolve(), args.base.resolve(), args.output.resolve(), data=args.data.resolve(),
          likelihood_dir=args.likelihood.resolve() if args.likelihood else None,
          training_dir=args.training.resolve() if args.training else None)
