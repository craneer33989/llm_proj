"""Atomic, hashed LoRA/optimizer/RNG checkpoints at optimizer-step boundaries."""
from pathlib import Path
import random
from uuid import uuid4

from .deepseek_trajectories import atomic_json
from .prepare_sft import sha256

RESUME_FIELDS = ("model", "snapshot", "train_actions_sha256", "steps", "microbatch_size",
                 "gradient_accumulation", "max_length", "learning_rate", "seed", "selection",
                 "rank", "alpha", "dropout", "target_modules", "base_dtype", "attention",
                 "gradient_checkpointing", "loss_scope", "example_weighting", "versions")


def validate_resume(saved, current):
    changed = [field for field in RESUME_FIELDS if saved.get(field) != current.get(field)]
    if changed:
        raise ValueError(f"Resume training configuration changed: {', '.join(changed)}")


def save_checkpoint(output, model, optimizer, torch, settings, records, progress):
    step = records[-1]["update"]
    folder = output / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"step-{step:06d}"
    if target.exists():
        raise ValueError(f"Checkpoint already exists: {target}")
    staging = folder / (f".saving-{step:06d}-" + uuid4().hex)
    staging.mkdir()
    model.save_pretrained(staging / "adapter", safe_serialization=True)
    state = {"optimizer": optimizer.state_dict(), "torch_cpu_rng": torch.get_rng_state(),
             "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "python_rng": random.getstate()}
    torch.save(state, staging / "training-state.pt")
    atomic_json(staging / "runtime.json", {"format_version": 1, "settings": settings,
                "trainable_parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
                "completed_updates": step, "records": records, "progress": progress})
    files = {str(path.relative_to(staging)).replace("\\", "/"): sha256(path)
             for path in sorted(staging.rglob("*")) if path.is_file()}
    atomic_json(staging / "manifest.json", {"format_version": 1, "step": step, "files_sha256": files})
    # Destination never exists: interrupted writes remain in .saving-* and are
    # ignored. The pointer changes only after the entire checkpoint is committed.
    staging.rename(target)
    atomic_json(output / "latest-checkpoint.json", {"path": str(target.relative_to(output)).replace("\\", "/"),
                "completed_updates": step, "manifest_sha256": sha256(target / "manifest.json")})
    return target


def read_checkpoint(output, settings):
    import json
    pointer = json.loads((output / "latest-checkpoint.json").read_text(encoding="utf-8"))
    target = (output / pointer["path"]).resolve()
    checkpoint_root = (output / "checkpoints").resolve()
    if not target.is_relative_to(checkpoint_root):
        raise ValueError("Checkpoint pointer is outside this run")
    manifest_file = target / "manifest.json"
    if sha256(manifest_file) != pointer["manifest_sha256"]:
        raise ValueError("Checkpoint manifest changed")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for name, digest in manifest["files_sha256"].items():
        path = (target / name).resolve()
        if not path.is_relative_to(target) or sha256(path) != digest:
            raise ValueError(f"Checkpoint file changed: {name}")
    runtime = json.loads((target / "runtime.json").read_text(encoding="utf-8"))
    if runtime["format_version"] != 1 or runtime["completed_updates"] != pointer["completed_updates"]:
        raise ValueError("Unsupported or inconsistent checkpoint")
    validate_resume(runtime["settings"], settings)
    return target, runtime


def restore_training_state(checkpoint, optimizer, torch):
    # Only read locally created files after read_checkpoint verifies hashes.
    state = torch.load(checkpoint / "training-state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if state["torch_cuda_rng"]:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    random.setstate(state["python_rng"])
