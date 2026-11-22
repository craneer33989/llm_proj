"""Transient Windows sharing locks must not discard completed training updates."""
import json

import pytest

from finance_agent import deepseek_trajectories as data


def test_atomic_write_retries_transient_permission_errors(tmp_path, monkeypatch):
    original = data.os.replace
    calls, waits = [], []

    def replacing(source, destination):
        calls.append((source, destination))
        if len(calls) < 4:
            raise PermissionError("simulated Windows sharing lock")
        return original(source, destination)

    monkeypatch.setattr(data.os, "replace", replacing)
    monkeypatch.setattr(data.time, "sleep", waits.append)
    target = tmp_path / "status.json"
    data.atomic_json(target, {"completed_updates": 100})
    assert json.loads(target.read_text(encoding="utf-8"))["completed_updates"] == 100
    assert len(calls) == 4 and len(waits) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_keeps_previous_state_if_lock_persists(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    data.atomic_json(target, {"completed_updates": 50})

    def blocked(*args):
        raise PermissionError("persistent permission failure")

    monkeypatch.setattr(data.os, "replace", blocked)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        data.atomic_json(target, {"completed_updates": 100})
    assert json.loads(target.read_text(encoding="utf-8"))["completed_updates"] == 50
