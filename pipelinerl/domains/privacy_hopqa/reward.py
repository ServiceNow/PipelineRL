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


def _parse_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value or "").strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def _hop_number(hop: dict) -> int:
    return _parse_int(hop.get("hop_number") or hop.get("hop")) or 0


def _hop_dependencies(hop: dict) -> list[int]:
    hop_number = _hop_number(hop)
    template = str(hop.get("question_templated") or "")
    refs = {int(match) for match in re.findall(r"\((\d+)\)", template)}
    return sorted(ref for ref in refs if 0 < ref < hop_number)


def dependency_aware_metrics(hops: list[dict], per_hop: list[dict]) -> dict:
    """Compute task metrics that do not punish downstream dependent hops.

    Raw hop accuracy answers "how many gold hop answers did the agent produce?"
    Conditional hop accuracy answers "how many still-evaluable hops did it solve
    before a required prior answer broke the chain?"
    """
    per_hop_by_number = {}
    for item in per_hop:
        hop_number = _parse_int(item.get("hop"))
        if hop_number is not None:
            per_hop_by_number[hop_number] = item

    sorted_hops = sorted(hops, key=_hop_number)
    correct_evaluable_hops: set[int] = set()
    prefix_correct_hops = 0
    prefix_intact = True
    conditional_correct_hops = 0
    evaluable_hops = 0
    blocked_hops = 0
    first_incorrect_hop: int | None = None
    enriched_per_hop = []

    for hop in sorted_hops:
        hop_number = _hop_number(hop)
        hop_score = dict(per_hop_by_number.get(hop_number) or {})
        dependencies = _hop_dependencies(hop)
        blocked_by = [dep for dep in dependencies if dep not in correct_evaluable_hops]
        blocked = bool(blocked_by)
        hop_correct = bool(hop_score.get("correct"))
        if blocked:
            blocked_hops += 1
        else:
            evaluable_hops += 1
            if hop_correct:
                conditional_correct_hops += 1
                correct_evaluable_hops.add(hop_number)
            elif first_incorrect_hop is None:
                first_incorrect_hop = hop_number

        if prefix_intact and hop_correct and not blocked:
            prefix_correct_hops += 1
        else:
            prefix_intact = False

        hop_score.update(
            {
                "dependencies": dependencies,
                "dependency_blocked": blocked,
                "blocked_by": blocked_by,
                "conditional_correct": bool(hop_correct and not blocked),
            }
        )
        enriched_per_hop.append(hop_score)

    total_hops = len(sorted_hops)
    return {
        "per_hop": enriched_per_hop,
        "conditional_correct_hops": conditional_correct_hops,
        "evaluable_hops": evaluable_hops,
        "blocked_hops": blocked_hops,
        "conditional_hop_accuracy": conditional_correct_hops / evaluable_hops if evaluable_hops else 0.0,
        "prefix_correct_hops": prefix_correct_hops,
        "prefix_hop_accuracy": prefix_correct_hops / total_hops if total_hops else 0.0,
        "first_incorrect_hop": first_incorrect_hop,
    }


def answers_match(
    predicted: str | None,
    truth: str | list[str] | tuple[str, ...] | None,
    f1_threshold: float = 0.75,
    match_mode: str = "exact_or_f1",
) -> tuple[bool, float]:
    pred = normalize_answer(predicted)
    golds = list(truth) if isinstance(truth, (list, tuple)) else [truth]
    normalized_golds = [normalize_answer(gold) for gold in golds if normalize_answer(gold)]
    if not pred or not normalized_golds:
        score = float(pred == "" and not normalized_golds)
        return bool(score), score

    if any(pred == gold for gold in normalized_golds):
        return True, 1.0

    best_score = max(token_f1_score(pred, gold) for gold in normalized_golds)
    if match_mode == "accepted_exact":
        return False, best_score
    if match_mode == "exact_or_f1":
        return best_score >= f1_threshold, best_score
    if match_mode == "f1_or_substring":
        if any(gold in pred or pred in gold for gold in normalized_golds):
            return True, 1.0
        return best_score >= f1_threshold, best_score
    raise ValueError(f"unknown privacy_hopqa answer_match_mode: {match_mode}")


def score_chain_answers(
    problem: dict,
    answers: dict[str, str],
    f1_threshold: float = 0.75,
    reward_mode: str = "all_hops_correct",
    answer_match_mode: str = "exact_or_f1",
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
        correct, match_score = answers_match(
            predicted,
            accepted_answers,
            f1_threshold=f1_threshold,
            match_mode=answer_match_mode,
        )
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
        match_mode=answer_match_mode,
    )
    chain_complete = all(answers.get(str(hop["hop_number"])) not in ("", "NOT_FOUND", None) for hop in hops)
    hop_accuracy = correct_count / total_hops
    dependency_metrics = dependency_aware_metrics(hops, per_hop)
    per_hop = dependency_metrics.pop("per_hop")
    if reward_mode == "all_hops_correct":
        reward = 1.0 if correct_count == len(hops) and len(hops) > 0 else 0.0
    elif reward_mode == "hop_accuracy":
        reward = hop_accuracy
    else:
        raise ValueError(f"unknown privacy_hopqa reward_mode: {reward_mode}")
    strict_chain_success = bool(correct_count == len(hops) and len(hops) > 0 and final_correct)

    return {
        "per_hop": per_hop,
        "correct_hops": correct_count,
        "total_hops": len(hops),
        "hop_accuracy": hop_accuracy,
        "raw_hop_accuracy": hop_accuracy,
        **dependency_metrics,
        "strict_chain_success": strict_chain_success,
        "reward": reward,
        "final_correct": final_correct,
        "final_match_score": final_match_score,
        "final_accepted_answers": final_accepted_answers,
        "chain_complete": chain_complete,
        "f1_threshold": f1_threshold,
        "answer_match_mode": answer_match_mode,
    }
