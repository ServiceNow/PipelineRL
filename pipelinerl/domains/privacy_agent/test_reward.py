from pipelinerl.domains.privacy_agent.reward import normalize_answer, score_chain_answers, token_f1_score


def test_normalize_answer_handles_currency_suffixes():
    assert normalize_answer("$2.5M") == "25 million"
    assert normalize_answer("  500K ") == "500 thousand"


def test_score_chain_answers_counts_total_questions():
    problem = {
        "global_answer": "Lee's Market",
        "hops": [
            {"hop_number": 1, "answer": "1976", "doc_id": "local/DR0005/IN0001/company-history.md"},
            {"hop_number": 2, "answer": "Lee's Market", "doc_id": "web/123"},
        ],
    }
    answers = {
        "1": "1976",
        "2": "Lee's Market",
        "FINAL": "Lee's Market",
    }
    score = score_chain_answers(problem, answers)
    assert score["correct_hops"] == 2
    assert score["total_hops"] == 2
    assert score["reward"] == 1.0
    assert score["final_correct"] is True


def test_score_chain_answers_accepts_high_overlap_matches():
    assert token_f1_score("Lee's Market revenue", "Lee's Market") >= 0.75

    problem = {
        "global_answer": "Lee's Market",
        "hops": [
            {"hop_number": 1, "answer": "1976", "doc_id": "local/DR0005/IN0001/company-history.md"},
        ],
    }
    answers = {"1": "The answer is 1976", "FINAL": "Lee's Market"}
    score = score_chain_answers(problem, answers, f1_threshold=0.75)
    assert score["correct_hops"] == 1
    assert score["per_hop"][0]["match_score"] >= 0.75
