"""Collect blind teacher rollouts in filing-tools-v1, then grade and export SFT.

Uses only the standard library for HTTPS. Never sends gold answers or oracle
actions to the provider. Run with --dry-run first; see docs/deepseek-trajectories.md.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import random
from threading import Event, local
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from .prepare_sft import sha256
from .trajectory_runtime import Environment, PROTOCOL, dumps, initial_messages, score_episode
from .trajectory_sources import OUTPUT

ENDPOINT = "https://api.deepseek.com/chat/completions"
VERSION = 1


def now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def atomic_writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with staging.open("x", encoding="utf-8", newline="\n") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        # On Windows a viewer, scanner or short-lived reader can temporarily
        # deny replacement even though the destination is writable. A telemetry
        # write must not abort hours of GPU work on a transient sharing lock.
        for attempt in range(20):
            try:
                os.replace(staging, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(.025 * 2 ** attempt, .5))
    finally:
        try:
            staging.unlink(missing_ok=True)
        except PermissionError:
            pass  # An uncommitted staging file is never selected for recovery.


def atomic_text(path, value):
    with atomic_writer(path) as handle:
        handle.write(value)


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError(f"Run is locked: {lock}. Remove it only after confirming the previous process stopped.") from None
    try:
        with handle:
            json.dump({"pid": os.getpid(), "started_at": now()}, handle)
        yield
    finally:
        lock.unlink(missing_ok=True)


@dataclass(frozen=True)
class Settings:
    model: str = "deepseek-v4-pro"
    thinking: bool = True
    reasoning_effort: str = "high"
    max_tokens: int = 8192
    timeout: float = 180
    retries: int = 3


class APIError(Exception):
    def __init__(self, message, *, fatal=False):
        super().__init__(message)
        self.fatal = fatal


class NoRedirect(HTTPRedirectHandler):
    # Do not forward the Authorization header to another endpoint.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DeepSeekClient:
    def __init__(self, api_key, settings):
        self.api_key = api_key
        self.settings = settings
        self.local = local()

    def open(self, request):
        if not hasattr(self.local, "opener"):
            self.local.opener = build_opener(NoRedirect())
        return self.local.opener.open(request, timeout=self.settings.timeout)

    def complete(self, messages, record):
        settings = self.settings
        payload = {"model": settings.model, "messages": messages, "stream": False,
                   "max_tokens": settings.max_tokens, "response_format": {"type": "json_object"},
                   "thinking": {"type": "enabled" if settings.thinking else "disabled"}}
        if settings.thinking:
            payload["reasoning_effort"] = settings.reasoning_effort
        # There is deliberately no `tools` parameter: local JSON actions and
        # user-role observations exactly match the student runtime. In this API
        # mode DeepSeek does not require previous reasoning_content to be sent.
        for attempt in range(settings.retries + 1):
            request_id = uuid4().hex
            record({"event": "request", "request_id": request_id, "retry": attempt, "payload": payload})
            req = Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key},
                          method="POST")
            try:
                with self.open(req) as response:
                    raw = response.read().decode("utf-8")
                # Save even malformed provider responses before parsing them.
                record({"event": "response", "request_id": request_id, "body": raw})
                try:
                    result = json.loads(raw)
                except (ValueError, TypeError):
                    raise APIError("Provider returned invalid JSON") from None
                if not isinstance(result, dict) or not isinstance(result.get("choices"), list) or not result["choices"]:
                    raise APIError("Provider response has no completion choices")
                return result
            except HTTPError as error:
                status = error.code
                retry_after = error.headers.get("Retry-After") if error.headers else None
                error.close()
                record({"event": "request_error", "request_id": request_id, "http_status": status})
                retryable = status in (408, 429) or 500 <= status <= 599
                if not retryable or attempt == settings.retries:
                    raise APIError(f"DeepSeek HTTP {status}", fatal=not retryable) from None
                try:
                    delay = float(retry_after)
                except (ValueError, TypeError):
                    delay = 2 ** attempt + random.random()
            except (URLError, TimeoutError, OSError, HTTPException) as error:
                record({"event": "request_error", "request_id": request_id, "error_type": type(error).__name__})
                if attempt == settings.retries:
                    raise APIError("DeepSeek network request failed after retries") from None
                delay = 2 ** attempt + random.random()
            time.sleep(min(30, max(0, delay)))


def job_key(case_id, sample):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:24] + f"-{sample:03d}"


def generate_episode(case, sample, data, output, client, settings):
    """No oracle repair; model/format errors remain rejected episodes."""
    key, invocation = job_key(case["id"], sample), uuid4().hex
    events_path = output / "events" / (key + ".jsonl")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path.exists() and events_path.stat().st_size:
        with events_path.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
    messages, env = initial_messages(case), Environment(case, data)
    actions, observations, steps, invalid = [], [], [], []
    final, api_error = None, None
    usage, api_requests, fatal = Counter(), 0, False
    started_at, start = now(), time.perf_counter()
    with events_path.open("a", encoding="utf-8", newline="\n") as log:
        def record(event):
            nonlocal api_requests
            if event["event"] == "request":
                api_requests += 1
            if event["event"] == "response":
                try:
                    body = json.loads(event["body"])
                    for name, value in (body.get("usage") or {}).items():
                        if isinstance(value, int) and not isinstance(value, bool):
                            usage[name] += value
                except (ValueError, AttributeError, TypeError):
                    pass
            log.write(dumps({"at": now(), "invocation": invocation, **event}) + "\n")
            log.flush()
            os.fsync(log.fileno())

        record({"event": "episode_start", "case_id": case["id"], "sample": sample, "gold_visible": False})
        for turn in range(1, case["max_actions"] + 1):
            step = {"turn": turn, "prompt": list(messages)}
            steps.append(step)
            try:
                response = client.complete(messages, record)
                step["response"] = response  # Includes model, id, usage and reasoning_content.
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(f"Completion did not finish normally: {choice.get('finish_reason')}")
                message = choice["message"]
                if message.get("tool_calls"):
                    raise ValueError("Expected a filing-tools-v1 JSON action, not native tool calls")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty action content")
                action = json.loads(content)
                if not isinstance(action, dict):
                    raise ValueError("Action must be a JSON object")
                step["action"] = action
                actions.append(action)
                if action.get("tool") == "search" and env.searches >= case["max_searches"]:
                    raise ValueError("Search budget exceeded")
                observation = env.execute(action)
                messages.append({"role": "assistant", "content": dumps(action)})
                if action.get("done") is True:
                    final = action
                    break
                step["observation"] = observation
                observations.append(observation)
                messages.append({"role": "user", "content": "工具观察：" + dumps(observation)})
            except APIError as error:
                api_error, fatal = str(error), error.fatal
                step["api_error"] = api_error
                break
            except (ValueError, KeyError, TypeError, ArithmeticError, AttributeError, IndexError) as error:
                invalid.append({"turn": turn, "error": str(error)})
                break
        checks = score_episode(case, final, actions, observations)
        checks["valid_actions"] = not invalid
        checks["passed"] = bool(checks["passed"] and not invalid and not api_error)
        row = {"case_id": case["id"], "sample": sample, "job_key": key, "invocation": invocation,
               "split": case["split"], "group": case["group"], "protocol": PROTOCOL,
               "question": case["question"], "as_of": case["as_of"], "companies": case["companies"],
               "teacher": asdict(settings), "gold_visible": False, "rule_fallback": False,
               "started_at": started_at, "completed_at": now(),
               "elapsed_seconds": round(time.perf_counter() - start, 3),
               "outcome": "api_error" if api_error else "passed" if checks["passed"] else "rejected",
               "api_error": api_error, "fatal_api_error": fatal,
               "answer": final, "expected": case["expected"], "checks": checks, "invalid": invalid,
               "actions": actions, "observations": observations, "messages": messages, "steps": steps,
               "usage": dict(usage), "api_requests": api_requests}
        record({"event": "episode_result", "result": row})
    atomic_json(output / "episodes" / (key + ".json"), row)
    return row


def export_results(output, cases, samples):
    rows, seen, sft_count, accepted_count = [], set(), 0, 0
    # Stream large prompts/responses rather than retaining all trajectories in RAM.
    with ExitStack() as stack:
        files = {name: stack.enter_context(atomic_writer(output / (name + ".jsonl")))
                 for name in ("trajectories", "answers", "sft-actions", "sft-trajectories")}

        def write(name, value):
            files[name].write(dumps(value) + "\n")

        for case in cases:
            for sample in range(samples):
                path = output / "episodes" / (job_key(case["id"], sample) + ".json")
                if not path.exists():
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                rows.append({name: row[name] for name in ("case_id", "sample", "group", "checks", "outcome")})
                write("trajectories", row)
                write("answers", {name: row[name] for name in (
                    "case_id", "sample", "split", "group", "question", "answer", "expected",
                    "checks", "outcome", "api_error", "gold_visible")})
                signature = (row["case_id"], hashlib.sha256(dumps(row["actions"]).encode("utf-8")).hexdigest())
                if row["split"] != "train" or not row["checks"]["passed"] or signature in seen:
                    continue
                seen.add(signature)
                metadata = {name: row[name] for name in ("case_id", "sample", "split", "group", "protocol", "gold_visible")}
                write("sft-trajectories", {**metadata, "messages": row["messages"], "answer": row["answer"]})
                accepted_count += 1
                for step in row["steps"]:
                    write("sft-actions", {**metadata, "turn": step["turn"], "action_type": step["action"].get("tool", "finish"),
                                          "prompt": step["prompt"],
                                          "completion": [{"role": "assistant", "content": dumps(step["action"])}]})
                    sft_count += 1
    completed = [r for r in rows if r["outcome"] != "api_error"]
    solved = {r["case_id"] for r in rows if r["checks"]["passed"]}
    first = [r for r in completed if r["sample"] == 0]
    groups = sorted({c["group"] for c in cases})
    # Include superseded API-error runs and interrupted invocations in cost accounting.
    all_usage, all_requests, response_count, request_errors = Counter(), 0, 0, 0
    for path in sorted((output / "events").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # A hard interruption may leave a partial last line.
                all_requests += event.get("event") == "request"
                request_errors += event.get("event") == "request_error"
                if event.get("event") == "response":
                    response_count += 1
                    try:
                        for name, value in (json.loads(event["body"]).get("usage") or {}).items():
                            if isinstance(value, int) and not isinstance(value, bool):
                                all_usage[name] += value
                    except (ValueError, AttributeError, TypeError):
                        pass
    summary = {"planned_questions": len(cases), "planned_episodes": len(cases) * samples,
               "saved_episodes": len(rows), "completed_episodes": len(completed),
               "passed_episodes": sum(r["checks"]["passed"] for r in rows),
               "rejected_episodes": sum(r["outcome"] == "rejected" for r in rows),
               "api_error_episodes": sum(r["outcome"] == "api_error" for r in rows),
               "solved_questions": len(solved),
               "first_sample": {"completed": len(first), "passed": sum(r["checks"]["passed"] for r in first)},
               "sft_trajectories": accepted_count, "sft_actions": sft_count,
               "api_requests_including_retries": all_requests, "logged_responses": response_count,
               "request_errors": request_errors, "reported_usage_all_invocations": dict(all_usage),
               "usage_note": "Only reported tokens are counted; requests without a response may still be billed.",
               "by_group": {group: {"questions": sum(c["group"] == group for c in cases),
                                    "solved_questions": len({r["case_id"] for r in rows if r["group"] == group and r["checks"]["passed"]}),
                                    "completed": sum(r["group"] == group for r in completed),
                                    "passed": sum(r["group"] == group and r["checks"]["passed"] for r in rows)} for group in groups}}
    atomic_json(output / "summary.json", summary)
    return summary


def prepare(data, questions, split, limit, groups, samples, settings):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Unsupported dataset protocol")
    inputs = {"questions": sha256(questions)}
    for name in ("sources.json", "corpus.sqlite3"):
        digest = sha256(data / name)
        if manifest["files_sha256"].get(name) != digest:
            raise ValueError(f"Dataset manifest mismatch: {name}. Rebuild the dataset first.")
        inputs[name] = digest
    if questions.parent == data and questions.name == f"{split}-questions.jsonl":
        if manifest["files_sha256"].get(questions.name) != inputs["questions"]:
            raise ValueError("Question file differs from dataset manifest")
    cases = read_jsonl(questions)
    if any(case["split"] != split or case["protocol"] != PROTOCOL for case in cases):
        raise ValueError("Questions must have the requested split and filing-tools-v1 protocol")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate question IDs")
    if groups:
        unknown = set(groups) - {c["group"] for c in cases}
        if unknown:
            raise ValueError(f"Unknown groups: {sorted(unknown)}")
        cases = [c for c in cases if c["group"] in groups]
    cases = cases[:limit] if limit is not None else cases
    if not cases:
        raise ValueError("No questions selected")
    for case in cases:
        initial_messages(case)
        expected = case.get("expected")
        if not isinstance(expected, dict) or not isinstance(expected.get("citations"), list):
            raise ValueError(f"Missing grading labels: {case['id']}")
        if (expected.get("status") not in ("answered", "insufficient", "incomparable")
                or any(not isinstance(ref, str) for ref in expected["citations"])
                or len(set(expected["citations"])) != len(expected["citations"])):
            raise ValueError(f"Invalid grading labels: {case['id']}")
        if expected["status"] == "answered":
            try:
                valid_value = Decimal(str(expected.get("value"))).is_finite()
            except InvalidOperation:
                valid_value = False
            if not valid_value or expected.get("unit") not in ("元", "%") or not expected["citations"]:
                raise ValueError(f"Invalid numeric answer label: {case['id']}")
        date.fromisoformat(case["as_of"])
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"Empty question: {case['id']}")
        for field in ("max_actions", "max_searches"):
            if type(case.get(field)) is not int or case[field] < 1:
                raise ValueError(f"Invalid {field}: {case['id']}")
    modules = ("deepseek_trajectories.py", "trajectory_runtime.py", "trajectory_corpus.py", "corpus.py")
    config = {"version": VERSION, "protocol": PROTOCOL, "endpoint": ENDPOINT,
              "gold_visible": False, "teacher": asdict(settings), "split": split, "samples": samples,
              "case_ids": [c["id"] for c in cases], "inputs_sha256": inputs,
              "code_sha256": {name: sha256(Path(__file__).parent / name) for name in modules}}
    return cases, config


def run_batch(cases, config, data, output, client, settings, workers=2, export_only=False):
    samples = config["samples"]
    with run_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8"))["config"] != config:
                raise ValueError("Run configuration/data/code changed. Use a new --output directory.")
        else:
            if export_only:
                raise ValueError("No saved run manifest to export")
            if any(output.iterdir()):
                if set(p.name for p in output.iterdir()) != {"run.lock"}:
                    raise ValueError("Output is not an empty run directory")
            atomic_json(manifest_path, {"created_at": now(), "config": config})
        if export_only:
            return export_results(output, cases, samples), False
        pending = []
        for case in cases:
            for sample in range(samples):
                path = output / "episodes" / (job_key(case["id"], sample) + ".json")
                if path.exists() and json.loads(path.read_text(encoding="utf-8"))["outcome"] != "api_error":
                    continue
                pending.append((case, sample))
        stop = Event()

        def work(case, sample):
            if stop.is_set():
                return None
            row = generate_episode(case, sample, data, output, client, settings)
            if row["fatal_api_error"]:
                stop.set()
            return row

        interrupted = False
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, case, sample) for case, sample in pending]
                try:
                    for future in as_completed(futures):
                        row = future.result()
                        if row is not None:
                            print(f"{row['case_id']} sample={row['sample']} {row['outcome']}"
                                  + (f" ({row['api_error']})" if row["api_error"] else ""), flush=True)
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
        except KeyboardInterrupt:
            interrupted = True
        finally:
            summary = export_results(output, cases, samples)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary, interrupted


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--questions", type=Path, help="Optional custom filing-tools-v1 question JSONL")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output", type=Path, default=Path("data/runtime/deepseek-teacher"))
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--samples", type=positive, default=3, help="Independent rollouts per question")
    parser.add_argument("--limit", type=positive, help="Select the first N matching questions; default: all")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--workers", type=positive, default=2)
    parser.add_argument("--max-tokens", type=positive, default=8192, help="Per-turn output budget, including thinking")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="high")
    parser.add_argument("--timeout", type=positive, default=180, help="HTTPS timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries after transient HTTP/network errors")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print plan without API calls or output writes")
    parser.add_argument("--export-only", action="store_true", help="Rebuild exports from saved episodes without API calls")
    args = parser.parse_args(argv)
    if args.retries < 0 or args.retries > 10:
        parser.error("--retries must be between 0 and 10")
    data, output = args.data.resolve(), args.output.resolve()
    questions = args.questions.resolve() if args.questions else data / f"{args.split}-questions.jsonl"
    settings = Settings(args.model, args.thinking == "enabled", args.reasoning_effort,
                        args.max_tokens, args.timeout, args.retries)
    try:
        cases, config = prepare(data, questions, args.split, args.limit, args.groups, args.samples, settings)
        turns = sum(c["max_actions"] for c in cases) * args.samples
        plan = {"questions": len(cases), "episodes": len(cases) * args.samples,
                "maximum_requests_including_retries": turns * (settings.retries + 1),
                "maximum_output_tokens_including_retries": turns * (settings.retries + 1) * settings.max_tokens,
                "model": settings.model, "gold_visible": False, "output": str(output),
                "note": "Upper bounds, not expected usage; each episode makes multiple API calls."}
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            return 0
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() if not args.export_only else ""
        if not api_key and not args.export_only:
            raise ValueError("Set DEEPSEEK_API_KEY in the process environment; do not put it in source files.")
        # Each worker has its own opener; the immutable client is otherwise stateless.
        client = DeepSeekClient(api_key, settings)
        summary, interrupted = run_batch(cases, config, data, output, client, settings, args.workers, args.export_only)
        if interrupted:
            return 130
        return 0 if args.export_only or summary["completed_episodes"] == summary["planned_episodes"] else 2
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
