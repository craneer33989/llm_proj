"""Train one evidence-selected adapter and compare it to the saved dev baseline."""
import argparse
from collections import Counter, defaultdict, deque
import json
from math import comb
from pathlib import Path
import random
import time
from types import SimpleNamespace

from .course_view import answer_evidence, answer_grounded
from .reopd import Rollout, clean, digest, environment_lock, initial_messages, parse
from .reopd_model import Actor, MODELS, checkpoint
from .reopd_run import collect, verify_lock, write_json, write_rows
from .download import sha256
from .reopd_train import action_logprobs, parameter_hash
from .scoring import answer_metrics, sentence_key
from .search import DEFAULT_DATA, SearchEngine, rows
from .training import TaskStore


SOURCES = ("rl-recovery", "rl-recovery-wrong-without-search", "rl-teacher", "rl-sft")


def query_key(query):
    return " ".join(query.casefold().split())


def visible_count(env, facts):
    return sum(any(sentence_key(f["text"]) in sentence_key(text)
                   for text in env.exposed.get(f["doc_id"], ())) for f in facts)


def prepare(run, out, engine, store, mixed=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint("sft"), local_files_only=True)
    baseline = {e["qid"]: e for e in rows(run / "rl-sft/episodes.jsonl")}
    baseline_scores = {s["qid"]: s for s in rows(run / "rl-sft/scores.jsonl")}
    candidates = {}
    successes = 0
    for source in SOURCES:
        directory = run / source
        good = {(s["qid"], s["attempt"]) for s in rows(directory / "scores.jsonl") if s["supported_success"]}
        for ep in rows(directory / "episodes.jsonl"):
            qid = ep["qid"]
            if (qid, ep["attempt"]) not in good:
                continue
            successes += 1
            if ep["split"] != "rl" or qid in store.tasks["dev"]:
                raise ValueError("Unexpected training split")
            env = Rollout(engine, store.task("rl", qid), "rl")
            facts = store.gold["rl"][qid]["supporting_facts"]
            seen = 0
            for turn, step in enumerate(ep["steps"]):
                env.advance(step["raw"])
                if env.steps[-1] != step:
                    raise ValueError(f"Replay mismatch: {qid}/{turn}")
                previous, seen = seen, visible_count(env, facts)
                if source == "rl-recovery" and turn == 0:
                    continue
                if step["kind"] == "finish" and seen == len(facts):
                    kind = "finish"
                elif step["kind"] == "search":
                    past_queries = {query_key(s["value"]) for s in ep["steps"][:turn]}
                    if query_key(step["value"]) in past_queries:
                        continue
                    kind = "continue" if turn else "initial"
                    if kind == "continue" and seen <= previous:
                        continue
                else:
                    continue
                prefix = ep["steps"][:turn]
                student_steps = baseline[qid]["steps"]
                failure = baseline_scores[qid]
                priority = None
                if (kind == "initial" and source in {"rl-teacher", "rl-recovery-wrong-without-search"}
                        and not failure["answer_em"] and [s["kind"] for s in student_steps] == ["finish"]):
                    priority = "wrong_without_search"
                elif (kind == "continue" and turn == 1 and source == "rl-recovery"
                      and failure["early_stop"] and prefix == student_steps[:1]):
                    priority = "early_stop"
                if kind == "initial" and not seen and priority is None:
                    continue
                key = digest([qid, prefix, step["raw"]])
                if key in candidates:
                    continue
                length = len(tokenizer.encode(step["raw"], add_special_tokens=False)) + int(kind == "finish")
                if not 0 < length <= 512:
                    continue
                candidates[key] = dict(qid=qid, split="rl", source=source, attempt=ep["attempt"],
                    turn=turn, prefix=prefix, teacher_target=step["raw"], kind=kind,
                    visible_before=previous, visible_after=seen, target_tokens=length, priority=priority)

    pool = list(candidates.values())
    write_rows(out / "pool.jsonl", pool)
    rng = random.Random(42)
    by_kind = defaultdict(dict)
    shuffled = rng.sample(pool, len(pool))
    ordered = sorted(shuffled, key=lambda r: (not r["priority"],
                     r["source"] != "rl-sft" if r["kind"] == "finish" else False))
    for row in ordered:
        by_kind[row["kind"]].setdefault(row["qid"], row)
    targets = {"initial": 48, "continue": 48, "finish": 24}
    selected = []
    for kind, count in targets.items():
        queue = list(by_kind[kind].values())
        priorities = sum(bool(r["priority"]) for r in queue)
        selected.extend(queue[:max(count, priorities)])
    rng.shuffle(selected)
    epoch = sorted(selected, key=lambda r: not r["priority"])
    schedule, tokens = [], 0
    for row in epoch * 3:
        if len(schedule) >= 2 * len(epoch) and tokens + row["target_tokens"] > 30000:
            break
        schedule.append(row)
        tokens += row["target_tokens"]
    if mixed:
        interleaved = []
        pattern = ("initial", "continue", "finish", "initial", "continue")
        for offset in range(0, len(schedule), len(epoch)):
            chunk = schedule[offset:offset + len(epoch)]
            rng.shuffle(chunk)
            groups = defaultdict(deque)
            for row in chunk:
                groups[row["kind"]].append(row)
            while any(groups.values()):
                for kind in pattern:
                    if groups[kind]:
                        interleaved.append(groups[kind].popleft())
        schedule = interleaved
    write_rows(out / "schedule.jsonl", schedule)
    write_json(out / "selection.json", {"successful_episodes": successes, "pool_actions": len(pool),
               "pool_qids": len({r["qid"] for r in pool}), "pool_by_kind": dict(Counter(r["kind"] for r in pool)),
               "epoch_actions": dict(Counter(r["kind"] for r in epoch)),
               "priority_qids": dict(Counter(r["priority"] for r in epoch if r["priority"])),
               "scheduled_tokens": tokens, "scheduled_actions": len(schedule),
               "scheduled_priority_actions": dict(Counter(r["priority"] for r in schedule if r["priority"])),
               "schedule_sha256": digest(schedule)})
    return schedule


def evaluate(run, out, engine, store):
    reference = run / "ablation-dev-12000tokens-42-wrong-without-search"
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    specs = manifest["specs"]
    if manifest["student"] != list(MODELS["sft"]) or manifest["max_tokens"] != 512 or manifest["batch"] != 4:
        raise ValueError("Saved baseline does not match the evaluation settings")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    verify_lock(out / "dev-config.json", {"specs": specs, "reference": str(reference),
                "adapter_sha256": metrics["adapter_sha256"], "seed": manifest["seed"]})
    actor = Actor("sft", 512, adapter=out / "adapter")
    base = {r["qid"]: r for r in rows(reference / "base/scores.jsonl")}
    original = {e["qid"]: e for e in rows(reference / "base/episodes.jsonl")}
    fixed = [(qid, 0, original[qid]["steps"][:1]) for qid, _, _ in specs
             if original[qid]["steps"][0]["kind"] == "search"]
    for name, cohort in (("dev", specs), ("dev-first-search", fixed)):
        episodes, scores = collect(engine, store, actor, "dev", cohort, out / name, 4, 0, manifest["seed"])
        if len(scores) != len(cohort):
            raise ValueError("Incomplete dev scoring")
        pairs = []
        for ep, score in zip(episodes, scores, strict=True):
            qid = ep["qid"]
            pair = dict(qid=qid, before=base[qid]["supported_success"], after=score["supported_success"],
                        before_em=base[qid]["answer_em"], after_em=score["answer_em"],
                        before_calls=original[qid]["calls"], after_calls=ep["calls"])
            if name == "dev-first-search":
                prefix = original[qid]["steps"][:1]
                state = Rollout(engine, store.task("dev", qid), "dev", prefix)
                facts = store.gold["dev"][qid]["supporting_facts"]
                pair.update(first_evidence_complete=visible_count(state, facts) == len(facts),
                            baseline_next=original[qid]["steps"][1]["kind"], adapter_next=ep["steps"][1]["kind"])
                initial_count = visible_count(state, facts)
                for label, steps in (("baseline", original[qid]["steps"]), ("adapter", ep["steps"])):
                    step = steps[1]
                    useful_query = step["kind"] == "search" and query_key(step["value"]) != query_key(prefix[0]["value"])
                    state = Rollout(engine, store.task("dev", qid), "dev", prefix)
                    if useful_query:
                        state.advance(step["raw"])
                    pair[f"{label}_new_query"] = useful_query
                    pair[f"{label}_added_evidence"] = visible_count(state, facts) > initial_count
            pairs.append(pair)
        write_rows(out / name / "paired.jsonl", pairs)
        report = {"n": len(pairs),
            **{k: sum(r[k] for r in pairs) for k in ("before", "after", "before_em", "after_em","before_calls","after_calls")},
            "gains": sum(r["after"] and not r["before"] for r in pairs),
            "losses": sum(r["before"] and not r["after"] for r in pairs)}
        if name == "dev-first-search":
            early = [r for r in pairs if base[r["qid"]]["early_stop"]]
            report["baseline_early_stop"] = {"n": len(early),
                "adapter_search": sum(r["adapter_next"] == "search" for r in early),
                "adapter_new_query": sum(r["adapter_new_query"] for r in early),
                "adapter_added_evidence": sum(r["adapter_added_evidence"] for r in early),
                "adapter_supported": sum(r["after"] for r in early)}
            report["added_evidence"] = {label: sum(r[f"{label}_added_evidence"] for r in pairs)
                                         for label in ("baseline", "adapter")}
        write_json(out / name / "comparison.json", report)
    oracle_qids = [qid for qid, _, _ in specs if base[qid]["early_stop"]]
    oracle_adapter = oracle_answers(actor, store, oracle_qids)
    del actor
    import torch

    torch.cuda.empty_cache()
    oracle_base = oracle_answers(Actor("sft", 512), store, oracle_qids)
    oracle_pairs = [{"qid": qid, "baseline": oracle_base[qid], "adapter": oracle_adapter[qid]} for qid in oracle_qids]
    write_rows(out / "oracle.jsonl", oracle_pairs)
    write_json(out / "oracle-summary.json", {"cohort": "baseline first-search early stop",
               "n": len(oracle_pairs), **{label: sum(r[label]["answer_em"] for r in oracle_pairs)
                                         for label in ("baseline", "adapter")}})


def oracle_answers(actor, store, qids):
    results = {}
    for offset in range(0, len(qids), 4):
        batch = qids[offset:offset + 4]
        histories = []
        for qid in batch:
            history = initial_messages(store.task("dev", qid)["question"])
            facts = store.gold["dev"][qid]["supporting_facts"]
            passages = "\n".join(clean(fact["text"]) for fact in facts)
            history.append({"role": "user", "content": "Relevant passages:\n" + passages +
                            "\nUse these passages to answer now. Submit the short answer only in the specified boxed format."})
            histories.append(history)
        for qid, raw in zip(batch, actor.generate(histories), strict=True):
            kind, value = parse(raw)
            gold = store.gold["dev"][qid]
            refs = [gold["answer"], *gold.get("aliases", [])]
            results[qid] = {"raw": raw, "kind": kind, "answer_em":
                            max(answer_metrics(value, ref)[0] for ref in refs) if kind == "finish" else 0}
        print(f"oracle: {offset + len(batch)}/{len(qids)}", flush=True)
    return results


def report(run, out, store):
    reference = run / "ablation-dev-12000tokens-42-wrong-without-search"

    def table(path, qids):
        scores = {s["qid"]: s for s in rows(path / "scores.jsonl")}
        return {ep["qid"]: {**ep, **scores[ep["qid"]]} for ep in rows(path / "episodes.jsonl") if ep["qid"] in qids}

    qids = {qid for qid, _, _ in json.loads((reference / "manifest.json").read_text(encoding="utf-8"))["specs"]}
    gold = store.gold["dev"]
    models = {name: table(path, qids) for name, path in
              (("baseline", reference / "base"), ("old_A", reference / "A"), ("adapter", out / "dev"))}
    if any(set(model) != qids for model in models.values()):
        raise ValueError("Expected a complete paired development cohort")

    def metrics(model, ids):
        return {"answer_em": int(sum(model[qid]["answer_em"] for qid in ids)),
                "answer_grounded": sum(answer_grounded(model[qid], gold[qid]) for qid in ids),
                "strict": sum(model[qid]["supported_success"] for qid in ids)}

    full = {name: metrics(model, qids) for name, model in models.items()}
    baseline, adapted = models["baseline"], models["adapter"]
    early = {qid for qid in qids if baseline[qid]["early_stop"]}
    enough = {qid for qid in early if answer_evidence(baseline[qid], gold[qid], prefix=1)}
    fixed = table(out / "dev-first-search", early)
    diagnostics = {r["qid"]: r for r in rows(out / "dev-first-search/paired.jsonl")}
    buckets = {}
    for name, ids in (("single_hop_enough", enough), ("true_early_stop", early - enough)):
        buckets[name] = {"n": len(ids), "baseline": metrics(baseline, ids), "adapter": metrics(fixed, ids),
                         "new_query": sum(diagnostics[qid]["adapter_new_query"] for qid in ids),
                         "added_evidence": sum(diagnostics[qid]["adapter_added_evidence"] for qid in ids),
                         "grounded_gains": sum(answer_grounded(fixed[qid], gold[qid])
                                               and not answer_grounded(baseline[qid], gold[qid]) for qid in ids),
                         "grounded_losses": sum(answer_grounded(baseline[qid], gold[qid])
                                                and not answer_grounded(fixed[qid], gold[qid]) for qid in ids)}
    paired = [{"qid": qid, **{name: dict(em=ep["answer_em"], grounded=answer_grounded(ep, gold[qid]),
                strict=ep["supported_success"]) for name, ep in
                (("baseline", baseline[qid]), ("old_A", models["old_A"][qid]), ("adapter", adapted[qid]))}}
              for qid in sorted(qids)]
    write_rows(out / "grounded-paired.jsonl", paired)
    decision = full["adapter"]["strict"] > 24 and full["adapter"]["answer_grounded"] > 41
    comparisons = {}
    for reference_model in ("baseline", "old_A"):
        comparisons[reference_model] = {}
        for metric in ("grounded", "strict"):
            gains = sum(r["adapter"][metric] and not r[reference_model][metric] for r in paired)
            losses = sum(r[reference_model][metric] and not r["adapter"][metric] for r in paired)
            discordant = gains + losses
            comparisons[reference_model][metric] = {"gains": gains, "losses": losses,
                "mcnemar_exact_p": min(1.0, 2 * sum(comb(discordant, i) for i in range(min(gains, losses) + 1))
                                        / (2 ** discordant))}
    result = {"n": len(qids), "full": full, "first_search": buckets, "passes_gate": decision,
              "paired": comparisons}
    write_json(out / "grounded-summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def train_schedule(args, schedule, output, engine, store):
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    student = Actor("sft", args.max_tokens)
    student.model = get_peft_model(student.model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                                   target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    student.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    initial = parameter_hash(student.model, trainable_only=True)
    optimizer = torch.optim.AdamW([param for param in student.model.parameters() if param.requires_grad], lr=args.lr)
    accumulation = args.gradient_accumulation
    optimizer_steps = 0
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    updates = []
    used = 0
    for index, row in enumerate(schedule):
        env = Rollout(engine, store.tasks["rl"][row["qid"]], "rl", row["prefix"])
        prompt = torch.tensor(student.tokenizer.encode(render(env.messages), add_special_tokens=False), device=device)
        raw = row["teacher_target"]
        kind, _ = parse(raw)
        tokens = student.tokenizer.encode(raw, add_special_tokens=False)
        if args.finish_eos and kind == "finish":
            tokens.append(student.tokenizer.eos_token_id)
        if len(tokens) != row["target_tokens"] or len(tokens) > args.max_tokens:
            raise ValueError("SFT action length differs from the scheduled length")
        action = torch.tensor(tokens, device=device)
        student.model.train()
        student_lp = action_logprobs(student.model, prompt, action)
        loss = -student_lp.mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite student loss")
        (loss / accumulation).backward()
        used += len(action)
        pending += 1
        complete = index + 1 == len(schedule)
        grad = None
        if pending == accumulation or complete:
            if pending < accumulation:
                for parameter in student.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(accumulation / pending)
            grad = float(torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1.0, error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending = 0
        updates.append({"step": index + 1, "qid": row["qid"], "loss": loss.item(), "grad_norm": grad,
                        "source": row["source"], "turn": row["turn"], "action_tokens_before_budget_clip": len(action),
                        "student_update_tokens": len(action), "prompt_masked_tokens": len(prompt),
                        "teacher_scoring_input_tokens": 0, "sampled_reverse_kl": None,
                        "sampled_action_kind": kind, "policy_search_calls": 0,
                        "prefix_replay_calls": len(env.env.session.trace)})
        if index == 0:
            write_json(output / "mask-example.json", {"prefix_tokens": prompt.tolist(), "action_tokens": action.tolist(),
                                                       "loss_mask": [0] * len(prompt) + [1] * len(action)})
        write_rows(output / "updates.jsonl", updates)
        print(updates[-1], flush=True)

    final = parameter_hash(student.model, trainable_only=True)
    if final == initial:
        raise RuntimeError("LoRA weights did not change")
    checkpoint_dir = output / "adapter"
    student.model.save_pretrained(checkpoint_dir)
    expected = {name: weight.detach().cpu().clone() for name, weight in get_peft_model_state_dict(student.model).items()}
    del student, optimizer, loss, student_lp
    torch.cuda.empty_cache()
    loaded = Actor("sft", args.max_tokens, adapter=checkpoint_dir)
    restored = get_peft_model_state_dict(loaded.model)
    if restored.keys() != expected.keys() or any(not torch.equal(value, restored[name].cpu()) for name, value in expected.items()):
        raise RuntimeError("Reloaded adapter differs from saved weights")
    write_json(output / "metrics.json", {"steps": len(updates), "student_update_tokens": used,
                                          "optimizer_steps": optimizer_steps,
                                          "student_sampled_tokens": 0,
                                          "prefix_replay_calls": sum(u["prefix_replay_calls"] for u in updates),
                                          "policy_search_calls": sum(u["policy_search_calls"] for u in updates),
                                          "teacher_scoring_input_tokens": sum(u["teacher_scoring_input_tokens"] for u in updates),
                                          "seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "adapter_reloaded": True,
                                          "adapter_sha256": {name: sha256(checkpoint_dir / name) for name in
                                                             ("adapter_config.json", "adapter_model.safetensors")},
                                          "adapter_changed": True, "teacher_unchanged": None})



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "eval", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixed", action="store_true", help="Interleave action types and use the v2.1 learning rate")
    args = parser.parse_args()
    engine, store = SearchEngine(DEFAULT_DATA), TaskStore(DEFAULT_DATA)
    try:
        lock = environment_lock(engine)
        verify_lock(args.run / "environment.json", lock)
        if args.stage == "train":
            args.output.mkdir(parents=True, exist_ok=False)
            schedule = prepare(args.run, args.output, engine, store, args.mixed)
            recipe = SimpleNamespace(seed=42, max_tokens=512, group="A", lr=8e-5 if args.mixed else 2e-5,
                                     gradient_accumulation=8, finish_eos=True,
                                     update_tokens=sum(row["target_tokens"] for row in schedule))
            write_json(args.output / "recipe.json", {**vars(recipe), "environment": lock, "student": MODELS["sft"]})
            train_schedule(recipe, schedule, args.output, engine, store)
        elif args.stage == "eval":
            evaluate(args.run, args.output, engine, store)
            report(args.run, args.output, store)
        else:
            report(args.run, args.output, store)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
