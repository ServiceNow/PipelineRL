from pipelinerl.domains.privacy_agent.dataset import load_problems


def test_load_seed20_problems():
    problems = load_problems(["seed20"])
    assert len(problems) == 20
    first = problems[0]
    assert first["domain"] == "privacy_agent"
    assert first["dataset"] == "seed20"
    assert first["chain_id"] == "51277225"
    assert first["task_id"] == "DR0005"
    assert first["company"] == "Lee's Market"
    assert first["n_hops"] == len(first["hops"]) > 0
    assert first["task"] == first["numbered_questions"]
