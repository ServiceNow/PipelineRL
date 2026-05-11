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


def _accepted_list(primary: str | None, variants: list | tuple | None = None, alternates: list | tuple | None = None) -> list[str]:
    values: list[str] = []
    for value in [primary, *(variants or []), *(alternates or [])]:
        text = "" if value is None else str(value)
        if text and normalize_answer(text) not in {normalize_answer(existing) for existing in values}:
            values.append(text)
    return values


def answers_match(predicted: str | None, truth: str | list[str] | tuple[str, ...] | None, f1_threshold: float = 0.75) -> tuple[bool, float]:
    pred = normalize_answer(predicted)
    golds = list(truth) if isinstance(truth, (list, tuple)) else [truth]
    normalized_golds = [normalize_answer(gold) for gold in golds if normalize_answer(gold)]
    if not pred or not normalized_golds:
        score = float(pred == "" and not normalized_golds)
        return bool(score), score

    if any(pred == gold or gold in pred or pred in gold for gold in normalized_golds):
        return True, 1.0

    best_score = max(token_f1_score(pred, gold) for gold in normalized_golds)
    return best_score >= f1_threshold, best_score


def score_chain_answers(
    problem: dict,
    answers: dict[str, str],
    f1_threshold: float = 0.75,
    reward_mode: str = "all_hops_correct",
) -> dict:
    hops = list(problem.get("hops") or [])
    per_hop = []
    correct_count = 0
    for hop in hops:
        hop_num = str(hop["hop_number"])
        predicted = answers.get(hop_num, "")
        accepted_answers = _accepted_list(
            hop.get("answer", ""),
            hop.get("accepted_answer_variants"),
            hop.get("alternate_valid_answers"),
        )
        correct, match_score = answers_match(predicted, accepted_answers, f1_threshold=f1_threshold)
        correct_count += int(correct)
        per_hop.append(
            {
                "hop": int(hop_num),
                "agent_answer": predicted,
                "ground_truth": hop.get("answer", ""),
                "accepted_answers": accepted_answers,
                "correct": correct,
                "match_score": match_score,
                "doc_id": hop.get("doc_id"),
            }
        )

    total_hops = max(len(hops), 1)
    final_key = "FINAL" if "FINAL" in answers else str(len(hops))
    final_accepted_answers = _accepted_list(
        problem.get("global_answer", ""),
        problem.get("global_answer_variants"),
        problem.get("global_alternate_valid_answers"),
    )
    final_correct, final_match_score = answers_match(
        answers.get(final_key, ""),
        final_accepted_answers,
        f1_threshold=f1_threshold,
    )
    chain_complete = all(answers.get(str(hop["hop_number"])) not in ("", "NOT_FOUND", None) for hop in hops)
    hop_accuracy = correct_count / total_hops
    if reward_mode == "all_hops_correct":
        reward = 1.0 if correct_count == len(hops) and len(hops) > 0 else 0.0
    elif reward_mode == "hop_accuracy":
        reward = hop_accuracy
    else:
        raise ValueError(f"unknown privacy_hopqa reward_mode: {reward_mode}")

    return {
        "per_hop": per_hop,
        "correct_hops": correct_count,
        "total_hops": len(hops),
        "hop_accuracy": hop_accuracy,
        "reward": reward,
        "final_correct": final_correct,
        "final_match_score": final_match_score,
        "final_accepted_answers": final_accepted_answers,
        "chain_complete": chain_complete,
        "f1_threshold": f1_threshold,
    }
