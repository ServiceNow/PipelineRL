import numpy as np

from pipelinerl.domains.terminal.turn_probes import (
    auroc,
    cross_validated_metric,
    group_fold,
    is_prefix,
    loo_centered_returns,
    pre_gen_position,
    prompt_length,
    turn_bucket,
)


def test_prompt_length_and_pre_gen_position():
    record = {"input_ids": [1, 2, 3, 4, 5], "n_predicted": 2}
    assert prompt_length(record) == 3
    assert pre_gen_position(record) == 2


def test_is_prefix():
    full = [1, 2, 3, 4, 5]
    assert is_prefix([1, 2, 3, 99], full, 3)
    assert not is_prefix([1, 9, 3], full, 3)
    assert not is_prefix([1, 2, 3], [1, 2], 3)


def test_loo_centered_returns_matches_manual_computation():
    group_ids = ["g", "g", "g", "h"]
    rollout_ids = [0, 0, 1, 0]
    rewards = [1.0, 1.0, -1.0, 0.5]
    z = loo_centered_returns(group_ids, rollout_ids, rewards)
    # g: rollout 0 (R=1) vs rollout 1 (R=-1) -> z = 1-(-1) = 2 and -1-1 = -2
    assert z[0] == 2.0 and z[1] == 2.0
    assert z[2] == -2.0
    # singleton group -> 0
    assert z[3] == 0.0


def test_group_fold_is_deterministic_and_in_range():
    folds = [group_fold(f"task-{i}", 5) for i in range(100)]
    assert folds == [group_fold(f"task-{i}", 5) for i in range(100)]
    assert set(folds) <= set(range(5))
    assert len(set(folds)) > 1


def test_turn_bucket_boundaries():
    assert turn_bucket(0) == "0-2"
    assert turn_bucket(3) == "3-7"
    assert turn_bucket(15) == "8-15"
    assert turn_bucket(40) == "16+"


def test_auroc_known_values():
    labels = np.array([1, 1, 0, 0])
    assert auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert auroc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(auroc(np.array([1, 1]), np.array([0.1, 0.2])))


def test_cross_validated_metric_recovers_linear_signal():
    rng = np.random.default_rng(0)
    n, d = 400, 8
    x = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    logits = x @ w
    y_bin = (logits + 0.1 * rng.normal(size=n) > 0).astype(float)
    folds = rng.integers(0, 5, size=n)
    score = cross_validated_metric(x, y_bin, folds, l2=1.0, binary=True)
    assert score > 0.9

    y_reg = logits + 0.1 * rng.normal(size=n)
    r2 = cross_validated_metric(x, y_reg, folds, l2=1.0, binary=False)
    assert r2 > 0.9

    y_noise = rng.normal(size=n)
    r2_noise = cross_validated_metric(x, y_noise, folds, l2=1.0, binary=False)
    assert r2_noise < 0.1
