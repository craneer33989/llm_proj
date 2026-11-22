import json

import pytest

from hotpot_search.multihop import dependency_links
from hotpot_search.teacher_select import BudgetStop, Teacher, export, load_section_key


def test_only_requested_section_is_read(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text("other\nsk-" + "x"*24 + "\nsft data\nsk-" + "y"*24 + "\n", encoding="utf-8")
    assert load_section_key(path) == "sk-" + "y"*24
    path.write_text("sft data\nnot a key\nsk-" + "y"*24, encoding="utf-8")
    with pytest.raises(ValueError, match="next nonempty"):
        load_section_key(path)


def test_dependency_requires_new_entity_and_new_provider_turn():
    gold = {"supporting_facts": [{"doc_id": "director", "title": "Mira Finch"}]}
    events = [{"name": "local_search", "arguments": {"query": "Film Atlas"}, "provider_turn": 1,
               "observation": {"results": [{"doc_id": "film", "title": "Film Atlas", "text": "Directed by Mira Finch."}]}},
              {"name": "local_search", "arguments": {"query": "Mira Finch born"}, "provider_turn": 2,
               "observation": {"results": [{"doc_id": "director", "title": "Mira Finch", "text": "Born in Cedar."}]}}]
    assert dependency_links("Where was Film Atlas director born?", gold, events)
    assert not dependency_links("Where was Mira Finch born?", gold, events)
    events[1]["provider_turn"] = 1
    assert not dependency_links("Where was Film Atlas director born?", gold, events)


def test_budget_stops_before_any_paid_request(tmp_path):
    teacher = Teacher("sk-not-a-real-key", tmp_path, budget_cny=0)
    with pytest.raises(BudgetStop):
        teacher.complete([{"role": "user", "content": "q"}], [], lambda event: None)
    assert teacher.usage["requests"] == 0
    assert not (tmp_path / "usage.json").exists()


def test_rejected_episode_is_never_exported(tmp_path):
    (tmp_path / "episodes").mkdir()
    rejected = {"id": "bad", "question": "q", "accepted": False, "rejections": ["no_verified_new_entity_dependency"],
                "sft_messages": [{"role": "assistant", "content": "bad"}], "tools": []}
    (tmp_path / "episodes" / "bad.json").write_text(json.dumps(rejected), encoding="utf-8")
    summary = export(tmp_path)
    assert summary["accepted"] == 0
    assert (tmp_path / "teacher_sft.jsonl").read_text(encoding="utf-8") == ""
