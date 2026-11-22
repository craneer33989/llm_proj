from copy import deepcopy
import json

import pytest

from finance_agent.deepseek_trajectories import atomic_json
from finance_agent.prepare_sft import sha256
from finance_agent.prepare_training_data import validate_release


@pytest.fixture
def inputs(tmp_path):
    data, release=tmp_path/'data',tmp_path/'release'
    data.mkdir(); release.mkdir()
    names=['sources.json','corpus.sqlite3','validation-questions.jsonl','test-questions.jsonl',
           'validation-actions.jsonl','test-actions.jsonl']
    for name in names:
        (data/name).write_text('{}\n',encoding='utf-8')
    for split in ('validation','test'):
        (data/f'{split}-questions.jsonl').write_text(json.dumps({'id':split+'-one'})+'\n',encoding='utf-8')
    atomic_json(data/'manifest.json',{'files_sha256':{n:sha256(data/n) for n in names},
        'splits':{s:{'companies':[c],'reports':[c+'-2024-FY']} for s,c in
                  [('train','000333'),('validation','600001'),('test','600002')]}})
    row={'case_id':'train-one','turn':1,'group':'direct','split':'train','gold_visible':False,
         'prompt':[{'role':'system','content':'Protocol'}, {'role':'user','content':json.dumps({'companies':[{'code':'000333'}]})}],
         'completion':[{'role':'assistant','content':'{"tool":"search"}'}]}
    def write(rows):
        (release/'train-actions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        atomic_json(release/'manifest.json',{'all_exported_trajectories_reexecuted':True,'gold_visible':False,
            'source_coverage':{'companies':['000333'],'required_reports':['000333-2024-FY']},
            'files_sha256':{'train-actions.jsonl':sha256(release/'train-actions.jsonl')},'sft_actions':len(rows)})
    write([row])
    return release,data,row,write


def test_verified_release_is_accepted(inputs):
    release,data,row,_=inputs
    assert validate_release(release,data)[0]==[row]


def test_updated_manifest_cannot_hide_duplicate_or_gold_rows(inputs):
    release,data,row,write=inputs
    write([row,row])
    with pytest.raises(ValueError,match='Duplicate'):
        validate_release(release,data)
    write([{**row,'gold_visible':True}])
    with pytest.raises(ValueError,match='gold labels'):
        validate_release(release,data)


def test_heldout_company_and_observed_report_are_rejected(inputs):
    release,data,row,write=inputs
    bad=deepcopy(row)
    bad['prompt'][1]['content']=json.dumps({'companies':[{'code':'600002'}]})
    write([bad])
    with pytest.raises(ValueError,match='Held-out company'):
        validate_release(release,data)
    bad=deepcopy(row)
    bad['prompt'] += [{'role':'assistant','content':'{"tool":"search"}'},
        {'role':'user','content':'工具观察：'+json.dumps({'hits':[{'report_id':'600002-2024-FY'}]})}]
    write([bad])
    with pytest.raises(ValueError,match='Held-out report'):
        validate_release(release,data)


def test_release_hash_tampering_is_detected(inputs):
    release,data,_,_=inputs
    (release/'train-actions.jsonl').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='file changed'):
        validate_release(release,data)
