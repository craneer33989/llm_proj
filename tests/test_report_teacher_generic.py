import json

from finance_agent.deepseek_trajectories import atomic_json
from finance_agent.evaluate_teacher import metrics, process_checks
from finance_agent.evaluate_trajectories import rollout, summarize
from finance_agent.expand_sft import read_jsonl
from finance_agent.prepare_sft import sha256
from finance_agent.report_teacher_eval import build
from finance_agent.trajectory_sources import OUTPUT


def test_report_uses_actual_validation_size_and_short_training_window(tmp_path):
    cases=read_jsonl(OUTPUT/'validation-questions.jsonl')[:2]
    data=tmp_path/'data'; data.mkdir()
    (data/'validation-questions.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf-8')
    atomic_json(data/'manifest.json',{})
    rows=[]
    for case in cases:
        row=rollout(case,'reference')
        row['checks'].update(process_checks(case,row))
        row['generations']=[{'raw':json.dumps(a)} for a in row['actions']]
        rows.append(row)
    cfg={'split':'validation','snapshot':'local-snapshot','dataset_manifest_sha256':sha256(data/'manifest.json'),
         'do_sample':False,'max_new_tokens':384,'enable_thinking':False,'gold_visible':False,
         'rule_fallback':False,'rollout_source_sha256':'same','scorer_sha256':'same','versions':{}}
    common={'summary':summarize(rows),'metrics':metrics(rows,cases),'rows':rows,'metric_definition_sha256':'same',
            'total_wall_seconds':1,'peak_process_allocated_mib':1,'peak_process_reserved_mib':1}
    base,sft,training=tmp_path/'base',tmp_path/'sft',tmp_path/'training'
    atomic_json(base/'report.json',{**common,'config':{**cfg,'adapter':None,'adapter_sha256':None}})
    atomic_json(sft/'report.json',{**common,'config':{**cfg,'adapter':'local-adapter','adapter_sha256':'weight-hash'}})
    atomic_json(training/'report.json',{'adapter_sha256':'weight-hash','completed_updates':3,'action_examples_seen':3,
        'config':{'loss_scope':'completion'},'steps':[{'loss':v} for v in [3.,2.,1.]]})
    result=build(sft,base,tmp_path/'report',data=data,training_dir=training)
    text=(tmp_path/'report/README.md').read_text(encoding='utf-8')
    assert result['split']=='validation' and result['evaluation_cases']==2
    assert result['training']['first_20_update_loss_mean']==2
    assert '验证集共 2 题' in text and '第 3 步' in text
    assert '前 3 步平均 loss=2.000000' in text
    assert '789' not in text and '120 题' not in text
