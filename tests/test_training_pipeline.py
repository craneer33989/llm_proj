from copy import deepcopy
import json
from pathlib import Path

import pytest

from finance_agent import training_pipeline as pipeline
from finance_agent.deepseek_trajectories import atomic_json
from finance_agent.sft_teacher import segment_end
from finance_agent.training_checkpoint import validate_resume


@pytest.fixture
def config(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    path=tmp_path/'config.json'
    atomic_json(path,{'run_dir':'runs/one'})
    return pipeline.load_config(path)


def test_segments_keep_one_fixed_total_schedule():
    assert segment_end(0,789,100)==100
    assert segment_end(100,789,100)==200
    assert segment_end(750,789,100)==789
    assert segment_end(300,789,None)==789
    with pytest.raises(ValueError): segment_end(0,789,0)


def test_frozen_experiment_allows_control_budget_but_rejects_method_changes(config):
    pipeline.freeze(config)
    changed=deepcopy(config)
    changed['training']['segment_updates']=25
    changed['training']['checkpoint_every']=5
    pipeline.freeze(changed)
    changed['method']['rank']=8
    with pytest.raises(ValueError,match='configuration changed'):
        pipeline.freeze(changed)
    with pytest.raises(ValueError,match='rank'):
        validate_resume({'rank':16},{'rank':8})


def test_cli_arguments_use_total_updates_and_separate_segment_budget(config):
    args=pipeline.training_args(config,{'rows':9},resume=True,segment=1)
    assert args[args.index('--steps')+1]==3
    assert args[args.index('--segment-updates')+1]==1
    assert '--resume' in args
    config['training']['max_updates']=4
    with pytest.raises(ValueError,match='one full data pass'):
        pipeline.training_args(config,{'rows':9})


def test_all_continues_segment_limit_but_stops_manual_pause(config,monkeypatch):
    results=iter([{'state':'paused','pause_reason':'segment_limit'}, {'state':'paused','pause_reason':'requested'}])
    monkeypatch.setattr(pipeline,'train',lambda *a,**k:next(results))
    monkeypatch.setattr(pipeline,'evaluate',lambda _:pytest.fail('Manual pause must not start evaluation'))
    assert pipeline.execute(config,'all')['pause_reason']=='requested'
    assert pipeline.status(config)['pipeline']['state']=='paused'


def test_all_finishes_training_before_evaluation_and_reporting(config,monkeypatch):
    calls=[]
    results=iter([{'state':'paused','pause_reason':'segment_limit'},{'state':'complete'}])
    monkeypatch.setattr(pipeline,'train',lambda *a,**k:next(results))
    monkeypatch.setattr(pipeline,'evaluate',lambda _:calls.append('evaluate'))
    monkeypatch.setattr(pipeline,'report',lambda _:calls.append('report'))
    pipeline.execute(config,'all')
    assert calls==['evaluate','report']


def test_pause_only_writes_request_without_killing_process(config):
    path=Path(config['run_dir'])/'training'
    atomic_json(path/'status.json',{'state':'training'})
    assert pipeline.execute(config,'pause')['state']=='pause_requested'
    assert (path/'pause.requested').exists()
    assert json.loads((path/'status.json').read_text())['state']=='training'


@pytest.mark.parametrize('payload',[{'training':{'segement_updates':10}}, {'method':{'name':'grpo'}}, {'run_dir':'../outside'}])
def test_invalid_configs_fail_early(tmp_path,monkeypatch,payload):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    file=tmp_path/'x.json'
    atomic_json(file,{'run_dir':'runs/one',**payload})
    with pytest.raises(ValueError): pipeline.load_config(file)


def test_historical_sources_are_read_only_for_training(config):
    config['artifacts']={k:str(Path(config['run_dir'])/'old'/k) for k in ('training','base','sft','likelihood')}
    with pytest.raises(ValueError,match='read-only'):
        pipeline.train(config)
