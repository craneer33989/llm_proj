"""Trainer-only terminal scoring. Never return reference answers to the actor."""
from collections import Counter
import re
import string


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_metrics(prediction, reference):
    predicted, expected = normalize_answer(prediction), normalize_answer(reference)
    exact = float(predicted == expected)
    if predicted in ("yes", "no", "noanswer") or expected in ("yes", "no", "noanswer"):
        return exact, exact
    left, right = predicted.split(), expected.split()
    overlap = sum((Counter(left) & Counter(right)).values())
    if not overlap:
        return exact, 0.0
    precision, recall = overlap / len(left), overlap / len(right)
    return exact, 2 * precision * recall / (precision + recall)


def sentence_key(text):
    return " ".join(text.casefold().split())


def score_episode(gold, answer, citations, exposed_documents, invalid_calls=0):
    exact, f1 = answer_metrics(answer, gold["answer"])
    cited = set(citations)
    unseen = cited - set(exposed_documents)
    facts = gold["supporting_facts"]
    supported = 0
    supporting_citations = set()
    observed = 0
    for fact in facts:
        text = sentence_key(" ".join(exposed_documents.get(fact["doc_id"], [])))
        visible = bool(text) and sentence_key(fact["text"]) in text
        observed += bool(visible)
        supported += bool(visible and fact["doc_id"] in cited)
        if visible and fact["doc_id"] in cited:
            supporting_citations.add(fact["doc_id"])
    recall = supported / len(facts) if facts else 0.0
    precision = len(supporting_citations) / len(cited) if cited else 0.0
    evidence_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Reward correct answers with actual cited evidence; no bonus for extra hops.
    penalty = min(.2, .05 * len(unseen) + .01 * invalid_calls)
    reward = max(-.2, f1 * (.8 + .2 * evidence_f1) - penalty)
    return {"reward": round(reward, 6), "answer_em": exact, "answer_f1": f1,
            "cited_support_sentence_recall": recall,
            "citation_precision": precision, "evidence_f1": evidence_f1,
            "observed_support_sentence_recall": observed / len(facts) if facts else 0.0,
            "unseen_citations": len(unseen), "invalid_calls": invalid_calls,
            "supported_success": bool(exact and recall == 1 and precision == 1 and not unseen),
            "evidence_metric": "Exact normalized labeled sentences visible and cited; not semantic entailment"}
