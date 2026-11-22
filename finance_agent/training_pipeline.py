"""Configuration-driven financial SFT preparation, segmented training and reports."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

from .deepseek_trajectories import atomic_json, atomic_text, run_lock
from .filings import ROOT
from .prepare_sft import sha256
from .prepare_training_data import validate_release

DEFAULT = {
    'schema_version': 1, 'run_dir': None,
    'data': {'release': 'data/sft/deepseek-20261113', 'evaluation': 'data/trajectories/v1'},
    'model': {'snapshot': '.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'},
    'method': {'name':'lora_sft','rank':16,'alpha':32,'dropout':0.,'target_modules':['q_proj','k_proj','v_proj','o_proj']},
    'training': {'max_updates':None,'accumulation':4,'max_length':8192,'learning_rate':1e-4,'seed':42,
                 'selection':'random','speed_warmup_updates':2,'checkpoint_every':50,'segment_updates':100},
    'evaluation': {'split':'validation','empty_cache':True},
    'runtime': {'model_python':'.model-venv/Scripts/python.exe','report_python':'.venv/Scripts/python.exe'},
    'artifacts': None,
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def resolve(value):
    path = Path(value)
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path):
    raw, config = read(path), deepcopy(DEFAULT)
    unknown = set(raw)-set(DEFAULT)
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    for key,value in raw.items():
        if isinstance(DEFAULT[key],dict):
            if not isinstance(value,dict) or set(value)-set(DEFAULT[key]):
                raise ValueError(f'Invalid configuration group: {key}')
            config[key].update(value)
        else:
            config[key] = value
    if config['schema_version'] != 1 or not config['run_dir']:
        raise ValueError('schema_version=1 and run_dir are required')
    config['run_dir'] = str(resolve(config['run_dir']))
    if not Path(config['run_dir']).is_relative_to(ROOT.resolve()) or Path(config['run_dir']) == ROOT.resolve():
        raise ValueError('run_dir must be a child of the workspace')
    for group, keys in [('data',('release','evaluation')),('model',('snapshot',)),('runtime',('model_python','report_python'))]:
        for key in keys:
            config[group][key] = str(resolve(config[group][key]))
    if config['artifacts'] is not None:
        if set(config['artifacts']) != {'training','base','sft','likelihood'}:
            raise ValueError('Historical artifacts require training/base/sft/likelihood directories')
        config['artifacts'] = {k:str(resolve(v)) for k,v in config['artifacts'].items()}
    method, train = config['method'], config['training']
    if method['name'] != 'lora_sft':
        raise ValueError('Implemented method: lora_sft (BF16); RL/full/quantized training are not implemented here')
    for value in [method['rank'], method['alpha'],train['accumulation'],train['max_length'],train['checkpoint_every']]:
        if type(value) is not int or value <= 0:
            raise ValueError('Ranks, lengths and update counts must be positive integers')
    for key in ('max_updates','segment_updates'):
        if train[key] is not None and (type(train[key]) is not int or train[key] <= 0):
            raise ValueError(f'{key} must be null or a positive integer')
    if not 0 < train['learning_rate'] <= 1 or not 0 <= method['dropout'] < 1:
        raise ValueError('Invalid learning rate or dropout')
    if type(train['seed']) is not int or type(train['speed_warmup_updates']) is not int or train['speed_warmup_updates'] < 0:
        raise ValueError('Invalid seed/speed warmup')
    allowed = {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
    modules = method['target_modules']
    if not isinstance(modules,list) or not modules or len(set(modules)) != len(modules) or not set(modules)<=allowed:
        raise ValueError('Invalid Qwen3 LoRA target_modules')
    if train['selection'] not in ('random','mixed') or config['evaluation']['split'] not in ('validation','test'):
        raise ValueError('Invalid selection/evaluation split')
    if type(config['evaluation']['empty_cache']) is not bool:
        raise ValueError('empty_cache must be a boolean')
    return config


def identity(config):
    value = deepcopy(config)
    for key in ('segment_updates','checkpoint_every'):
        value['training'].pop(key)
    return value


def freeze(config):
    root = Path(config['run_dir'])
    path = root/'experiment.json'
    if path.exists():
        if identity(read(path)) != identity(config):
            raise ValueError('Experiment configuration changed; use a new run_dir')
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError('run_dir is not a new experiment directory')
        atomic_json(path, config)
    sources = root/'sources'
    sources.mkdir(exist_ok=True)
    index = {}
    for name in ('training_pipeline.py','prepare_training_data.py','sft_teacher.py','training_checkpoint.py',
                 'evaluate_teacher.py','evaluate_likelihood.py','report_teacher_eval.py','plot_teacher_eval.py'):
        source = Path(__file__).with_name(name)
        digest = sha256(source)
        target = sources/f'{digest[:12]}-{name}'
        if not target.exists(): target.write_bytes(source.read_bytes())
        index[name] = {'sha256':digest,'snapshot':str(target.relative_to(root))}
    atomic_json(root/'source-index.json',index)
    return root


def locations(config):
    root = Path(config['run_dir'])
    for name in ('prepared','training','evaluation','report','logs'):
        if not (root/name).resolve().is_relative_to(root):
            raise ValueError('Experiment output points outside run_dir')
    paths = {'training':root/'training','base':root/'evaluation/base',
             'sft':root/'evaluation/sft','likelihood':root/'evaluation/likelihood'}
    if config['artifacts']:
        paths = {k:Path(v) for k,v in config['artifacts'].items()}
    return root, paths


def event(root, **fields):
    entry = {'at_utc':datetime.now(timezone.utc).isoformat(), **fields}
    with (root/'events.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(entry,ensure_ascii=False)+'\n')


def invoke(config, stage, module, args, *, model=True):
    root = Path(config['run_dir'])
    python = config['runtime']['model_python' if model else 'report_python']
    command = [python,'-u','-X','utf8','-m',module,*map(str,args)]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    log_path = root/'logs'/f'{stamp}-{stage}.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    import os
    env = {**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    atomic_json(root/'state.json',{'state':'running','stage':stage,'log':str(log_path)})
    with log_path.open('w',encoding='utf-8') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        event(root,stage=stage,event='started',pid=process.pid,command=command,log=str(log_path))
        try:
            for line in process.stdout:
                log.write(line); log.flush()
                print(line,end='',flush=True)
            code = process.wait()
        except BaseException:
            # Do not leave an untracked child updating the same run after the controller exits.
            if process.poll() is None:
                process.terminate(); process.wait()
            raise
    event(root,stage=stage,event='finished',exit_code=code)
    if code:
        raise RuntimeError(f'{stage} failed with exit code {code}; see {log_path}')


def prepare(config):
    root,_ = locations(config)
    destination = root/'prepared'
    source, data = Path(config['data']['release']), Path(config['data']['evaluation'])
    validate_release(source,data)
    if (destination/'report.json').exists():
        report = read(destination/'report.json')
        if report['source_manifest_sha256'] != sha256(source/'manifest.json') or report['dataset_manifest_sha256'] != sha256(data/'manifest.json'):
            raise ValueError('Preparation inputs changed')
        for name,digest in report['frozen_files_sha256'].items():
            if sha256(destination/'release'/name) != digest:
                raise ValueError('Prepared snapshot changed')
        return report
    invoke(config,'prepare','finance_agent.prepare_training_data',[
        '--release',source,'--data',data,'--snapshot',config['model']['snapshot'],
        '--output',destination,'--max-length',config['training']['max_length']])
    return read(destination/'report.json')


def update_count(config, prepared):
    maximum = math.ceil(prepared['rows']/config['training']['accumulation'])
    count = config['training']['max_updates'] or maximum
    if count > maximum:
        raise ValueError('Current trainer supports at most one full data pass per experiment')
    if not 0 <= config['training']['speed_warmup_updates'] < count:
        raise ValueError('speed_warmup_updates must be less than total updates (this is not LR warmup)')
    return count


def training_args(config, prepared, resume=False, segment=None):
    root,_ = locations(config)
    train, method = config['training'],config['method']
    args = ['--release',root/'prepared/release','--output',root/'training',
        '--snapshot',config['model']['snapshot'],'--steps',update_count(config,prepared),
        '--accumulation',train['accumulation'],'--max-length',train['max_length'],
        '--learning-rate',train['learning_rate'],'--seed',train['seed'],'--selection',train['selection'],
        '--warmup-steps',train['speed_warmup_updates'],'--checkpoint-every',train['checkpoint_every'],
        '--rank',method['rank'],'--alpha',method['alpha'],'--dropout',method['dropout'],
        '--target-modules',*method['target_modules']]
    budget = train['segment_updates'] if segment is None else segment
    if budget is not None:
        args += ['--segment-updates',budget]
    if resume:
        args += ['--resume']
    return args


def train(config, *, require_resume=False, segment=None):
    if config['artifacts']:
        raise ValueError('Historical artifacts are read-only; use a new experiment config for training')
    prepared = prepare(config)
    root, paths = locations(config)
    directory = paths['training']
    if (directory/'report.json').exists():
        report = read(directory/'report.json')
        if report['adapter_sha256'] != sha256(directory/'adapter/adapter_model.safetensors'):
            raise ValueError('Completed training adapter changed')
        return {'state':'complete','completed_updates':report['completed_updates']}
    resume = (directory/'latest-checkpoint.json').exists()
    if require_resume and not resume:
        raise ValueError('No committed training checkpoint to resume')
    if (directory/'run.lock').exists():
        raise ValueError('Training run is locked; confirm the existing process has stopped')
    marker = directory/'pause.requested'
    if resume:
        marker.unlink(missing_ok=True)
    invoke(config,'train','finance_agent.sft_teacher',training_args(config,prepared,resume,segment))
    return read(directory/'status.json')


def evaluate(config):
    if config['artifacts']:
        raise ValueError('Historical artifact config only supports prepare/status/report/plan')
    root, paths = locations(config)
    training = paths['training']
    if not (training/'report.json').exists():
        raise ValueError('Finish training before evaluation')
    digest = sha256(training/'adapter/adapter_model.safetensors')
    expected_dataset = sha256(Path(config['data']['evaluation'])/'manifest.json')
    for name in ('base','sft'):
        output = paths[name]
        if (output/'report.json').exists():
            saved = read(output/'report.json')['config']
            if saved['dataset_manifest_sha256'] != expected_dataset or saved['split'] != config['evaluation']['split'] or saved['snapshot'] != config['model']['snapshot'] or saved['adapter_sha256'] != (digest if name=='sft' else None):
                raise ValueError('Completed evaluation inputs changed')
            continue
        args = ['--snapshot',config['model']['snapshot'],'--data',config['data']['evaluation'],
                '--split',config['evaluation']['split'],'--output',output]
        if name == 'sft': args += ['--adapter',training/'adapter']
        if config['evaluation']['empty_cache']: args += ['--empty-cache']
        if (output/'config.json').exists(): args += ['--resume']
        invoke(config,name+'-rollout','finance_agent.evaluate_teacher',args)
    likelihood = paths['likelihood']
    if not (likelihood/'report.json').exists():
        args = ['--snapshot',config['model']['snapshot'],'--adapter',training/'adapter',
                '--data',config['data']['evaluation'],'--split',config['evaluation']['split'],'--output',likelihood]
        if (likelihood/'config.json').exists(): args += ['--resume']
        invoke(config,'likelihood','finance_agent.evaluate_likelihood',args)


def report(config):
    root, paths = locations(config)
    if any(not (path/'report.json').exists() for path in paths.values()):
        raise ValueError('Training, both rollouts and likelihood must finish before reporting')
    for name in ('base','sft','likelihood'):
        saved = read(paths[name]/'report.json')['config']
        if saved['split'] != config['evaluation']['split'] or saved['snapshot'] != config['model']['snapshot']:
            raise ValueError('Report inputs do not match experiment model/split')
    args = ['--sft',paths['sft'],'--base',paths['base'],'--likelihood',paths['likelihood'],
            '--training',paths['training'],'--data',config['data']['evaluation'],'--output',root/'report']
    invoke(config,'report','finance_agent.report_teacher_eval',args,model=False)
    invoke(config,'charts','finance_agent.plot_teacher_eval',[
        '--comparison',root/'report','--training',paths['training'],'--likelihood',paths['likelihood']],model=False)
    atomic_json(root/'artifacts.json',{'inputs':{k:{'path':str(v),'report_sha256':sha256(v/'report.json')} for k,v in paths.items()},
        'report':str(root/'report/README.md'),'metrics_sha256':sha256(root/'report/metrics.json'),
        'figures_manifest_sha256':sha256(root/'report/figures/manifest.json')})
    return root/'report/README.md'


def status(config):
    root, paths = locations(config)
    result = {'run_dir':str(root),'pipeline':read(root/'state.json') if (root/'state.json').exists() else {'state':'not_started'}}
    for name,path in paths.items():
        if (path/'status.json').exists(): result[name] = read(path/'status.json')
    pointer = paths['training']/'latest-checkpoint.json'
    if pointer.exists(): result['checkpoint'] = read(pointer)
    return result


def execute(config, action, segment=None):
    root, paths = locations(config)
    if action == 'plan':
        return {'config':config,'stages':['prepare','train (segmented)','base rollout','SFT rollout','likelihood','report + figures'],
            'note':'No training or API calls performed by plan. all continues automatic segment boundaries; manual pause stops it.'}
    if action == 'status': return status(config)
    if action == 'pause':
        current = read(paths['training']/'status.json')
        if current['state'] not in ('training','preparing'):
            return {'state':current['state'],'message':'Training is not active'}
        atomic_text(paths['training']/'pause.requested','pause requested\n')
        return {'state':'pause_requested','message':'Checkpoint and exit at the next optimizer-step boundary'}
    root = freeze(config)
    with run_lock(root):
        event(root,event='invocation',action=action,segment_override=segment,config=config,sources=read(root/'source-index.json'))
        try:
            if action == 'prepare': prepare(config)
            elif action in ('train','resume'):
                result = train(config,require_resume=action=='resume',segment=segment)
                atomic_json(root/'state.json',{'state':'training_complete' if result['state']=='complete' else result['state'],
                                              'stage':'train','training':result})
                return result
            elif action == 'evaluate': evaluate(config)
            elif action == 'report': report(config)
            elif action == 'all':
                if config['artifacts']: raise ValueError('Use report for historical artifacts')
                while True:
                    result = train(config,segment=segment)
                    if result['state'] == 'complete': break
                    if result.get('pause_reason') != 'segment_limit':
                        atomic_json(root/'state.json',{'state':'paused','stage':'train','training':result})
                        return result
                evaluate(config); report(config)
            atomic_json(root/'state.json',{'state':{'prepare':'prepared','evaluate':'evaluated'}.get(action,'complete'),
                                          'stage':action,'report':str(root/'report/README.md') if action in ('all','report') else None})
            return status(config)
        except BaseException as error:
            atomic_json(root/'state.json',{'state':'failed','stage':action,'error_type':type(error).__name__,'error':str(error)})
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','prepare','train','resume','pause','status','evaluate','report','all'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--segment-updates',type=int)
    args=parser.parse_args()
    if args.segment_updates is not None and args.segment_updates<=0:
        parser.error('segment-updates must be positive')
    result=execute(load_config(args.config.resolve()),args.action,args.segment_updates)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
