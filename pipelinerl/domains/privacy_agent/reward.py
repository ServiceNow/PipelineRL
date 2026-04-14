import re
from collections import Counter


def normalize_answer(value: str | None) -> str:
    text = "" if value is None else str(value)
    text = text.lower().strip()
    text = re.sub(r"\$\s*", "", text)
    text = re.sub(r"(\d(?:\.\d+)?)\s*b\b", r"\1 billion", text)
    text = re.sub(r"(\d(?:\.\d+)?)\s*m\b", r"\1 million", text)
    text = re.sub(r"(\d(?:\.\d+)?)\s*k\b", r"\1 thousand", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1_score(predicted: str | None, truth: str | None) -> float:
    pred = normalize_answer(predicted)
    gold = normalize_answer(truth)
    if not pred or not gold:
        return float(pred == gold)

    pred_tokens = pred.split()
    gold_tokens = gold.split()
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answers_match(predicted: str | None, truth: str | None, f1_threshold: float = 0.75) -> tuple[bool, float]:
    pred = normalize_answer(predicted)
    gold = normalize_answer(truth)
    if not pred or not gold:
        score = float(pred == gold)
        return bool(score), score

    if pred == gold or gold in pred or pred in gold:
        return True, 1.0

    score = token_f1_score(pred, gold)
    return score >= f1_threshold, score


def score_chain_answers(
    problem: dict,
    answers: dict[str, str],
    f1_threshold: float = 0.75,
) -> dict:
    hops = list(problem.get("hops") or [])
    per_hop = []
    correct_count = 0
    for hop in hops:
        hop_num = str(hop["hop_number"])
        predicted = answers.get(hop_num, "")
        truth = hop.get("answer", "")
        correct, match_score = answers_match(predicted, truth, f1_threshold=f1_threshold)
        correct_count += int(correct)
        per_hop.append(
            {
                "hop": int(hop_num),
                "agent_answer": predicted,
                "ground_truth": truth,
                "correct": correct,
                "match_score": match_score,
                "doc_id": hop.get("doc_id"),
            }
        )

    total_hops = max(len(hops), 1)
    final_key = "FINAL" if "FINAL" in answers else str(len(hops))
    final_correct, final_match_score = answers_match(
        answers.get(final_key, ""),
        problem.get("global_answer", ""),
        f1_threshold=f1_threshold,
    )
    chain_complete = all(answers.get(str(hop["hop_number"])) not in ("", "NOT_FOUND", None) for hop in hops)
    reward = correct_count / total_hops

    return {
        "per_hop": per_hop,
        "correct_hops": correct_count,
        "total_hops": len(hops),
        "hop_accuracy": reward,
        "reward": reward,
        "final_correct": final_correct,
        "final_match_score": final_match_score,
        "chain_complete": chain_complete,
        "f1_threshold": f1_threshold,
    }
