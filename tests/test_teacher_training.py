"""Check teacher loss boundaries, non-truncation and benchmark sample selection."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from finance_agent.sft_teacher import batches, encode_example, select_examples, target_window


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, prompt, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "prompt"

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False
        return SimpleNamespace(input_ids=[1, 2, 3] if text == "prompt" else [4, 5])


def row():
    return {"case_id": "train-one", "turn": 2, "group": "direct", "split": "train", "gold_visible": False,
            "prompt": [{"role": "assistant", "content": "past action"}, {"role": "user", "content": "tool observation"}],
            "completion": [{"role": "assistant", "content": "target"}]}


def test_only_current_completion_and_eos_are_targets():
    encoded = encode_example(Tokenizer(), row())
    assert encoded["sequence"] == [1, 2, 3, 4, 5, 99]
    assert encoded["target"] == [4, 5, 99]
    assert encoded["prefix_length"] == 3
    assert target_window(encoded) == (2, 5)


def test_long_context_is_rejected_instead_of_truncated():
    with pytest.raises(ValueError, match="no truncation"):
        encode_example(Tokenizer(), row(), max_length=5)


@pytest.mark.parametrize("mutation", [{"split": "test"}, {"gold_visible": True}, {"expected": {"value": "secret"}}])
def test_holdouts_and_gold_labels_cannot_enter_training(mutation):
    data = {**row(), **mutation}
    with pytest.raises(ValueError):
        encode_example(Tokenizer(), data)


def test_benchmark_covers_longest_groups_and_largest_target_without_replacement():
    examples = [{"case_id": str(i), "sequence": list(range(10 + i)), "target": list(range(2 if i != 0 else 8)),
                 "group": str(i % 3)} for i in range(20)]
    picked = select_examples(examples, steps=3, accumulation=4)
    assert [r["case_id"] for r in picked[:4]] == ["19", "18", "17", "16"]
    assert len({r["case_id"] for r in picked}) == 12
    assert {r["group"] for r in picked} == {"0", "1", "2"}
    assert any(r["case_id"] == "0" for r in picked)
    assert picked == select_examples(examples, steps=3, accumulation=4)


def test_one_pass_batches_never_repeat_or_drop_the_last_partial_batch():
    examples = [{"case_id": str(i), "sequence": [1] * (i + 2), "target": [1], "group": "direct"} for i in range(9)]
    picked = select_examples(examples, steps=3, accumulation=4, selection="random")
    assert len(picked) == 9 and len({r["case_id"] for r in picked}) == 9
    assert [len(batch) for batch in batches(picked, 4)] == [4, 4, 1]
    with pytest.raises(ValueError, match="one pass"):
        select_examples(examples, steps=4, accumulation=4)
