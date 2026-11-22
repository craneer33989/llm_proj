"""Local training progress/ETA monitor; does not modify model or checkpoint state."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import time

from .deepseek_trajectories import atomic_json, atomic_text, read_jsonl


def snapshot(run):
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    rows = read_jsonl(run / "steps.jsonl") if (run / "steps.jsonl").exists() else []
    # Append-only raw logs can contain rerun updates after a crash; newest wins.
    records = sorted({r["update"]: r for r in rows}.values(), key=lambda r: r["update"])
    if status["state"] == "preparing" and (run / "latest-checkpoint.json").exists():
        committed = json.loads((run / "latest-checkpoint.json").read_text(encoding="utf-8"))["completed_updates"]
        records = [r for r in records if r["update"] <= committed]
    elif "completed_updates" in status:
        records = [r for r in records if r["update"] <= status["completed_updates"]]
    sampling = json.loads((run / "sampling.json").read_text(encoding="utf-8")) if (run / "sampling.json").exists() else {}
    now = datetime.now(timezone(timedelta(hours=8)))
    recent = records[-50:]
    seconds = sum(r["seconds"] for r in recent)
    tokens = sum(r["sequence_tokens"] for r in recent)
    throughput = tokens / seconds if seconds else None
    done = int(status.get("completed_updates", len(records)))
    total = int(status.get("total_updates", json.loads((run / "config.json").read_text(encoding="utf-8"))["steps"]))
    remaining = max(0, sampling.get("source_sequence_tokens", 0) - sum(r["sequence_tokens"] for r in records))
    latest = json.loads((run / "latest-checkpoint.json").read_text(encoding="utf-8")) if (run / "latest-checkpoint.json").exists() else {}
    gpu = None
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits", "-i", "0"], text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        used, size, utilization = [int(s.strip()) for s in result.strip().split(",")]
        gpu = {"used_mib": used, "total_mib": size, "utilization_percent": utilization}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    progress = {"updated_at_cst": now.isoformat(), "state": status["state"], "completed_updates": done,
                "total_updates": total, "percent": round(done / total * 100, 2),
                "latest_checkpoint_step": latest.get("completed_updates"),
                "recent_sequence_tokens_per_second": throughput,
                "estimated_training_end_cst": (now + timedelta(seconds=remaining / throughput)).isoformat()
                    if throughput and status["state"] not in ("complete", "failed", "paused") else None,
                "completed_actions": sum(r["actions"] for r in records), "last_step": records[-1] if records else None,
                "gpu_all_processes": gpu, "note": "ETA is a rolling token-throughput estimate; no task-quality evaluation is performed."}
    atomic_json(run / "progress.json", progress)
    eta = progress["estimated_training_end_cst"] or "—"
    gpu_text = f"{gpu['used_mib']:,} / {gpu['total_mib']:,} MiB（含其他进程）" if gpu else "不可用"
    atomic_text(run / "progress.md", f"""# 教师 SFT 训练进度

更新时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（北京时间）

| 项目 | 当前值 |
|---|---|
| 状态 | {status['state']} |
| 更新进度 | {done} / {total}（{progress['percent']}%） |
| 已处理动作 | {progress['completed_actions']} |
| 最近断点 | {progress['latest_checkpoint_step']} |
| 滚动估计训练结束时间 | {eta} |
| 当前整卡显存 | {gpu_text} |

这只是训练运行状态，不代表问答效果。估计时间根据最近 50 步的 token 吞吐滚动更新。

在当前聊天中说“暂停”可请求在完整更新之后保存断点并退出；也可运行项目根目录的 `Control-TeacherTraining.ps1 -Action Pause`。
""")
    return progress


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("interval must be at least five seconds")
    while True:
        progress = snapshot(args.run.resolve())
        if progress["state"] in ("complete", "paused", "failed"):
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            break
        time.sleep(args.interval)
