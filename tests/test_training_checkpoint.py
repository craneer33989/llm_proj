"""Checkpoint resumption must reject changes that alter gradients or data order."""
import pytest

from finance_agent.training_checkpoint import validate_resume


def test_resume_rejects_changed_data_batching_and_learning_rate():
    original = {"model": "Qwen3", "train_actions_sha256": "a", "gradient_accumulation": 4,
                "learning_rate": 1e-4, "steps": 789, "selection": "random"}
    for field, value in (("train_actions_sha256", "b"), ("gradient_accumulation", 2),
                         ("learning_rate", 2e-4), ("selection", "mixed")):
        with pytest.raises(ValueError, match=field):
            validate_resume(original, {**original, field: value})


def test_checkpoint_frequency_can_change_without_changing_training():
    original = {"model": "Qwen3", "seed": 42, "checkpoint_every": 50}
    validate_resume(original, {**original, "checkpoint_every": 25})
