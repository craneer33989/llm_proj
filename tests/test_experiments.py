import json

import pytest

from finance_agent.experiments import PILOT, training_experiment

pytestmark = pytest.mark.skipif(not (PILOT / "report.json").exists(), reason="Run the LoRA pilot first.")


def test_task_chart_counts_come_from_the_unassisted_case_records():
    payload = training_experiment()
    assert payload["groups"]["训练前"]["bridge"]["second_search"] == 0
    assert payload["groups"]["训练后（适配器重载）"]["bridge"]["second_hop_hit"] == 10
    assert payload["groups"]["训练前"]["direct"]["cases"] == 7
    assert payload["groups"]["训练后（适配器重载）"]["bridge"]["cases"] == 10
    assert payload["transitions"] == {"改善": 11, "保持通过": 6}
    assert len(payload["report"]["training"]["losses"]) == 10


def test_chart_rejects_assisted_or_unmatched_evaluation(tmp_path):
    for name in ("report.json", "before.json", "after.json"):
        (tmp_path / name).write_text((PILOT / name).read_text(encoding="utf-8"), encoding="utf-8")
    after_file = tmp_path / "after.json"
    after = json.loads(after_file.read_text(encoding="utf-8"))
    after["rows"][0]["answer"]["trace"][0]["fallback_reason"] = "premature_finish"
    after_file.write_text(json.dumps(after), encoding="utf-8")
    with pytest.raises(ValueError, match="rule-assisted"):
        training_experiment(tmp_path)
    after["rows"] = after["rows"][1:]
    after_file.write_text(json.dumps(after), encoding="utf-8")
    with pytest.raises(ValueError, match="same unique cases"):
        training_experiment(tmp_path)
