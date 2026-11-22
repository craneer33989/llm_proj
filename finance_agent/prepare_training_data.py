"""Freeze and inspect an audited financial teacher release before training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .deepseek_trajectories import atomic_json, read_jsonl
from .prepare_sft import sha256
from .sft_teacher import encode_example


def validate_release(release, data):
    manifest = json.loads((release/'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('all_exported_trajectories_reexecuted') or manifest.get('gold_visible') is not False:
        raise ValueError('Use an independently audited blind teacher release')
    for name, digest in manifest['files_sha256'].items():
        if Path(name).is_absolute() or Path(name).name != name:
            raise ValueError('Release manifest must contain plain filenames')
        path = (release/name).resolve()
        if not path.is_relative_to(release.resolve()) or sha256(path) != digest:
            raise ValueError(f'Release file changed: {name}')
    dataset = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for name in ('sources.json','corpus.sqlite3','validation-questions.jsonl','validation-actions.jsonl',
                 'test-questions.jsonl','test-actions.jsonl'):
        if sha256(data/name) != dataset['files_sha256'][name]:
            raise ValueError(f'Evaluation data changed: {name}')
    training_companies = set(dataset['splits']['train']['companies'])
    reports = set(dataset['splits']['train']['reports'])
    for split in ('validation','test'):
        if not training_companies.isdisjoint(dataset['splits'][split]['companies']) or not reports.isdisjoint(dataset['splits'][split]['reports']):
            raise ValueError('Dataset train/held-out reports are not isolated')
    coverage = manifest['source_coverage']
    if not set(coverage['companies']) <= training_companies or not set(coverage['required_reports']) <= reports:
        raise ValueError('Teacher coverage is not confined to training reports')
    heldout = {c['id'] for s in ('validation','test') for c in read_jsonl(data/f'{s}-questions.jsonl')}
    rows = read_jsonl(release/'train-actions.jsonl')
    if not rows:
        raise ValueError('Training input is empty')
    seen, content_seen = set(), set()
    for row in rows:
        key = (row['case_id'], row['turn'])
        if key in seen or row['case_id'] in heldout:
            raise ValueError('Duplicate action or held-out task in training input')
        seen.add(key)
        if row.get('split') != 'train' or row.get('gold_visible') is not False or any(k in row for k in ('expected','oracle_actions')):
            raise ValueError('Training input contains gold labels or a held-out split')
        messages, completion = row['prompt'], row['completion']
        if len(messages) < 2 or len(messages)%2 or [m['role'] for m in messages] != ['system','user']+['assistant','user']*((len(messages)-2)//2):
            raise ValueError('Expected a complete alternating tool history')
        if len(completion) != 1 or completion[0]['role'] != 'assistant' or not isinstance(json.loads(completion[0]['content']),dict):
            raise ValueError('Expected one assistant JSON action')
        user = json.loads(messages[1]['content'])
        if not {c['code'] for c in user['companies']} <= training_companies:
            raise ValueError('Held-out company in training prompt')
        for message in messages[2::2]:
            if message['role'] != 'assistant' or not isinstance(json.loads(message['content']),dict):
                raise ValueError('Invalid historical assistant action')
        for message in messages[3::2]:
            if not message['content'].startswith('工具观察：'):
                raise ValueError('Missing tool observation')
            observation = json.loads(message['content'][len('工具观察：'):])
            if any(hit['report_id'] not in reports for hit in observation.get('hits',[])):
                raise ValueError('Held-out report in training observation')
        fingerprint = json.dumps([messages,completion],ensure_ascii=False,sort_keys=True)
        if fingerprint in content_seen:
            raise ValueError('Duplicate prompt/completion pair')
        content_seen.add(fingerprint)
    if len(rows) != manifest['sft_actions']:
        raise ValueError('Release action count changed')
    return rows, manifest


def prepare(release, data, snapshot, output, max_length=8192):
    rows, manifest = validate_release(release, data)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    encoded = [encode_example(tokenizer,r,max_length) for r in rows]
    frozen = output/'release'
    frozen.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json',*manifest['files_sha256']]:
        target = frozen/name
        if target.exists():
            if sha256(target) != sha256(release/name):
                raise ValueError('Prepared snapshot differs; use a new experiment directory')
        else:
            shutil.copyfile(release/name,target)
    lengths = sorted(len(r['sequence']) for r in encoded)
    report = {'source_release':str(release),'source_manifest_sha256':sha256(release/'manifest.json'),
        'dataset_manifest_sha256':sha256(data/'manifest.json'),'prepared_release':str(frozen),
        'train_actions_sha256':sha256(frozen/'train-actions.jsonl'),'rows':len(rows),
        'tasks':len({r['case_id'] for r in rows}),'by_group':dict(Counter(r['group'] for r in rows)),
        'max_length_limit':max_length,'sequence_max':max(lengths),
        'sequence_median':lengths[len(lengths)//2],'sequence_p95':lengths[int((len(lengths)-1)*.95)],
        'sequence_tokens_total':sum(lengths),'target_tokens_total':sum(len(r['target']) for r in encoded),
        'gold_visible':False,'heldout_reports_excluded':True,'truncated_rows':0,
        'frozen_files_sha256':{p.name:sha256(p) for p in frozen.iterdir() if p.is_file()}}
    atomic_json(output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('release','data','snapshot','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max-length',type=int,default=8192)
    args=parser.parse_args()
    prepare(args.release.resolve(),args.data.resolve(),args.snapshot.resolve(),args.output.resolve(),args.max_length)
