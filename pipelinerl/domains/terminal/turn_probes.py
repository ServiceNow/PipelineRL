"""P0 probe study: how early do turn-boundary activations encode terminal-rollout outcomes?

Replicates the pre-generation linear-probe methodology of arXiv:2604.01202 on our
terminal domain, as groundwork for critic-free credit assignment. Two stages:

  extract (GPU): read per-turn TrainingTexts from an experiment's raw actor stream
      and save the residual-stream hidden state at each turn's pre-generation
      position (the last prompt token) for a strided subset of layers. Two modes:
      the default runs ONE forward pass per rollout over the last turn's sequence
      and keeps only turns whose prompt is a token-exact prefix of it. In practice
      that keeps ONLY the final turn (each turn's generation-prompt suffix is
      rewritten in the next turn's history), so probes fit on it describe terminal
      states. --all-turns instead runs one forward per sampled turn over that
      turn's own recorded prompt (exactly the state the policy generated from),
      giving true mid-rollout coverage at ~turns_per_rollout the compute.

  fit (CPU): train linear probes on the saved activations with GroupKFold by
      group_id (so the same task never spans train and test) and report AUROC /
      R^2 per (layer, turn-position bucket) for three targets: the turn's action
      is submit, final rollout success, and the LOO-centered rollout return z.

Usage:
    python -m pipelinerl.domains.terminal.turn_probes extract \
        --exp-dir <exp> --model-path <ckpt> [--max-rollouts 400]
    python -m pipelinerl.domains.terminal.turn_probes fit \
        --activations <exp>/probe_analysis/activations.pt
    python -m pipelinerl.domains.terminal.turn_probes export \
        --activations <exp>/probe_analysis/activations.pt --layer 31
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
TURN_BUCKETS = [(0, 2), (3, 7), (8, 15), (16, 10_000)]


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def prompt_length(record: dict) -> int:
    """Prompt token count of a turn record.

    Uses the explicit ``prompt_tokens`` field; ``n_predicted`` is a CHARACTER
    count (``len(output_text)``) and must not be used for token positions.
    """
    return int(record["prompt_tokens"])


def pre_gen_position(record: dict) -> int:
    """Index of the last prompt token: the state from which the first completion token is generated."""
    return prompt_length(record) - 1


def is_prefix(turn_ids, full_ids, length: int) -> bool:
    if length > len(full_ids):
        return False
    return np.array_equal(np.asarray(turn_ids[:length]), np.asarray(full_ids[:length]))


def loo_centered_returns(group_ids: list[str], rollout_ids: list[int], rewards: list[float]) -> np.ndarray:
    """z = R - mean(R of the OTHER rollouts in the group), per (group, rollout).

    Rollouts are deduplicated on (group_id, rollout_id); singleton groups get z=0.
    Returns one z per input row (rows of the same rollout share the z).
    """
    rollout_reward: dict[tuple[str, int], float] = {}
    for g, r, rew in zip(group_ids, rollout_ids, rewards):
        rollout_reward[(g, r)] = rew
    by_group: dict[str, list[float]] = defaultdict(list)
    for (g, _), rew in rollout_reward.items():
        by_group[g].append(rew)
    z = np.zeros(len(group_ids))
    for i, (g, r) in enumerate(zip(group_ids, rollout_ids)):
        rewards_g = by_group[g]
        if len(rewards_g) < 2:
            continue
        own = rollout_reward[(g, r)]
        z[i] = own - (sum(rewards_g) - own) / (len(rewards_g) - 1)
    return z


def group_fold(group_id: str, n_folds: int) -> int:
    """Deterministic fold assignment by group so a task never spans train and test."""
    digest = hashlib.sha1(group_id.encode()).hexdigest()
    return int(digest[:8], 16) % n_folds


def sample_turn_indices(n_turns: int, limit: int) -> list[int]:
    """Evenly spaced turn indices, always including the first and last turn."""
    if n_turns <= limit:
        return list(range(n_turns))
    return sorted(set(np.linspace(0, n_turns - 1, max(limit, 2)).round().astype(int).tolist()))


def turn_bucket(step_index: int) -> str:
    for lo, hi in TURN_BUCKETS:
        if lo <= step_index <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return "unknown"


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (ties get midranks)."""
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# stage 1: extract
# ---------------------------------------------------------------------------

def load_rollouts(exp_dir: Path, max_rollouts: int) -> list[list[dict]]:
    """Read the raw actor stream and return per-rollout turn lists sorted by step_index."""
    rollouts: dict[tuple[str, int], list[dict]] = defaultdict(list)
    files = sorted((exp_dir / "streams" / "actor").rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no actor stream files under {exp_dir}")
    done = False
    for path in files:
        with open(path) as f:
            for line in f:
                for rec in json.loads(line):
                    # Compact the token arrays immediately: full-context turns are
                    # ~2MB each as Python int lists but ~260KB as int32 arrays, and
                    # a single stream line carries a whole group of them.
                    rec["input_ids"] = np.asarray(rec["input_ids"], dtype=np.int32)
                    rec.pop("labels", None)
                    rec.pop("logprobs", None)
                    rec.pop("ref_logprobs", None)
                    rec.pop("text", None)
                    key = (rec["group_id"], rec["metadata"]["rollout_index"])
                    rollouts[key].append(rec)
                if len(rollouts) >= max_rollouts:
                    done = True
                    break
        if done:
            break
    ordered = []
    for key in sorted(rollouts.keys(), key=str):
        turns = sorted(rollouts[key], key=lambda r: r["metadata"]["step_index"])
        ordered.append(turns)
        if len(ordered) >= max_rollouts:
            break
    logger.info("loaded %d rollouts from %d stream files", len(ordered), len(files))
    return ordered


def find_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    for path in ("model.language_model.layers", "model.layers", "transformer.h"):
        node: torch.nn.Module | None = model
        for name in path.split("."):
            node = getattr(node, name, None)
            if node is None:
                break
        if node is not None:
            return list(node)
    raise ValueError("could not locate decoder layer list on the model")


def extract(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    exp_dir = Path(args.exp_dir)
    out_path = Path(args.out) if args.out else exp_dir / "probe_analysis" / "activations.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rollouts = load_rollouts(exp_dir, args.max_rollouts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.to(args.device).eval()
    layers = find_decoder_layers(model)
    layer_indices = sorted(set(range(0, len(layers), args.layer_stride)) | {len(layers) - 1})
    logger.info("probing layers %s of %d", layer_indices, len(layers))

    captured: dict[int, torch.Tensor] = {}
    positions_holder: list[int] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0, positions_holder, :].detach().to("cpu", torch.float16)
        return hook

    handles = [layers[i].register_forward_hook(make_hook(i)) for i in layer_indices]

    features: list[torch.Tensor] = []
    meta: list[dict] = []
    skipped = 0
    try:
        for n_done, turns in enumerate(rollouts):
            # Each forward is (input_ids, probed positions, the turn records they belong to).
            forwards: list[tuple[np.ndarray, list[int], list[dict]]] = []
            if args.all_turns:
                for ti in sample_turn_indices(len(turns), args.turns_per_rollout):
                    rec = turns[ti]
                    p = prompt_length(rec)
                    if p - 1 < 0 or p > args.max_seq_len:
                        skipped += 1
                        continue
                    forwards.append((rec["input_ids"][:p], [p - 1], [rec]))
            else:
                full_ids = turns[-1]["input_ids"]
                if len(full_ids) > args.max_seq_len:
                    full_ids = full_ids[: args.max_seq_len]
                valid_turns, positions = [], []
                for rec in turns:
                    p = prompt_length(rec)
                    if p - 1 < 0 or p > len(full_ids) or not is_prefix(rec["input_ids"], full_ids, p):
                        skipped += 1
                        continue
                    valid_turns.append(rec)
                    positions.append(p - 1)
                if valid_turns:
                    forwards.append((full_ids, positions, valid_turns))
            rollout_reward = turns[-1]["reward"]
            for ids, positions, recs in forwards:
                positions_holder[:] = positions
                input_ids = torch.tensor([ids], device=args.device)
                with torch.no_grad():
                    model(input_ids=input_ids, use_cache=False)
                stacked = torch.stack([captured[i] for i in layer_indices], dim=1)  # [turns, layers, hidden]
                features.append(stacked)
                for rec, pos in zip(recs, positions):
                    completion = tokenizer.decode(rec["input_ids"][prompt_length(rec):].tolist())
                    meta.append({
                        "group_id": rec["group_id"],
                        "rollout_index": rec["metadata"]["rollout_index"],
                        "step_index": rec["metadata"]["step_index"],
                        "n_turns": len(turns),
                        "turn_reward": rec["reward"],
                        "rollout_reward": rollout_reward,
                        "is_submit": SUBMIT_MARKER in completion,
                        "position": pos,
                    })
            if (n_done + 1) % 20 == 0:
                logger.info("extracted %d/%d rollouts (%d turns)", n_done + 1, len(rollouts), len(meta))
    finally:
        for h in handles:
            h.remove()

    torch.save(
        {
            "features": torch.cat(features, dim=0),
            "layer_indices": layer_indices,
            "meta": meta,
            "model_path": str(args.model_path),
            "exp_dir": str(exp_dir),
        },
        out_path,
    )
    logger.info(
        "saved %d turn activations x %d layers to %s (skipped %d turns: %s)",
        len(meta), len(layer_indices), out_path, skipped,
        "over max-seq-len" if args.all_turns else "prefix mismatches",
    )


# ---------------------------------------------------------------------------
# stage 2: fit
# ---------------------------------------------------------------------------

def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-6
    return (train - mean) / std, (test - mean) / std


def fold_standardization(
    w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fold standardization into raw-feature logistic weights.

    ``std`` is the raw feature std; the same +1e-6 epsilon as ``_standardize``
    is applied here.
    """
    w64 = w.astype(np.float64)
    mean64 = mean.astype(np.float64)
    scale = std.astype(np.float64) + 1e-6
    w_prime = w64 / scale
    b_prime = float(b - (mean64 / scale) @ w64)
    return w_prime.astype(np.float32), b_prime


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt) + l2 * (w * w).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return np.concatenate([w.detach().numpy(), b.detach().numpy()])


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    gram = xb.T @ xb + l2 * np.eye(xb.shape[1])
    return np.linalg.solve(gram, xb.T @ y)


def cross_validated_metric(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, l2: float, binary: bool
) -> float:
    """GroupKFold AUROC (binary) or R^2 (regression), pooled over held-out folds."""
    scores = np.zeros(len(y))
    for fold in np.unique(folds):
        test_mask = folds == fold
        if test_mask.all() or not test_mask.any():
            return float("nan")
        x_train, x_test = _standardize(x[~test_mask], x[test_mask])
        y_train = y[~test_mask]
        if binary:
            if len(np.unique(y_train)) < 2:
                return float("nan")
            wb = _fit_logistic(x_train, y_train, l2)
        else:
            wb = _fit_ridge(x_train, y_train, l2)
        scores[test_mask] = x_test @ wb[:-1] + wb[-1]
    if binary:
        return auroc(y.astype(bool), scores)
    ss_res = float(((y - scores) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def fit(args: argparse.Namespace) -> None:
    blob = torch.load(args.activations, map_location="cpu", weights_only=False)
    features: torch.Tensor = blob["features"]  # [N, L, H]
    layer_indices: list[int] = blob["layer_indices"]
    meta: list[dict] = blob["meta"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.activations).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    group_ids = [m["group_id"] for m in meta]
    rollout_ids = [m["rollout_index"] for m in meta]
    step_indices = np.array([m["step_index"] for m in meta])
    rollout_rewards = [m["rollout_reward"] for m in meta]
    folds = np.array([group_fold(g, args.folds) for g in group_ids])
    buckets = np.array([turn_bucket(int(s)) for s in step_indices])

    targets: dict[str, tuple[np.ndarray, bool]] = {
        "submit_action": (np.array([m["is_submit"] for m in meta], dtype=float), True),
        "rollout_success": (np.array([r >= 0.999 for r in rollout_rewards], dtype=float), True),
        "z_centered_return": (loo_centered_returns(group_ids, rollout_ids, rollout_rewards), False),
    }

    l2_values = [float(v) for v in args.l2_sweep.split(",")]
    rows = []
    bucket_names = ["all"] + sorted(set(buckets.tolist()))

    # Control probe: turn position only. If the activation probes do not beat this,
    # the "signal" is positional (late turns of long rollouts fail more), not
    # representational.
    x_control = np.stack([step_indices, step_indices ** 2], axis=1).astype(float)
    for bucket in bucket_names:
        mask = np.ones(len(meta), dtype=bool) if bucket == "all" else buckets == bucket
        if mask.sum() < args.min_bucket_size:
            continue
        for name, (y, binary) in targets.items():
            value = cross_validated_metric(x_control[mask], y[mask], folds[mask], 1.0, binary)
            rows.append({
                "target": name, "layer": -1, "bucket": bucket, "l2": 1.0,
                "n": int(mask.sum()), "n_pos": int(y[mask].sum()) if binary else -1,
                "metric": "auroc" if binary else "r2", "value": round(value, 4),
            })

    for li, layer in enumerate(layer_indices):
        x_layer = features[:, li, :].float().numpy()
        for bucket in bucket_names:
            mask = np.ones(len(meta), dtype=bool) if bucket == "all" else buckets == bucket
            if mask.sum() < args.min_bucket_size:
                continue
            for name, (y, binary) in targets.items():
                for l2 in l2_values:
                    value = cross_validated_metric(x_layer[mask], y[mask], folds[mask], l2, binary)
                    rows.append({
                        "target": name, "layer": layer, "bucket": bucket, "l2": l2,
                        "n": int(mask.sum()), "n_pos": int(y[mask].sum()) if binary else -1,
                        "metric": "auroc" if binary else "r2", "value": round(value, 4),
                    })
                logger.info("layer %d bucket %-5s %-18s done (n=%d)", layer, bucket, name, int(mask.sum()))

    csv_path = out_dir / "probe_results.csv"
    with open(csv_path, "w") as f:
        f.write("target,layer,bucket,l2,n,n_pos,metric,value\n")
        for r in rows:
            f.write(f"{r['target']},{r['layer']},{r['bucket']},{r['l2']},{r['n']},{r['n_pos']},{r['metric']},{r['value']}\n")

    lines = ["# P0 turn-probe results", "", f"activations: {args.activations}", ""]
    for name in targets:
        best = {}
        for r in rows:
            if r["target"] == name and (r["bucket"] not in best or r["value"] > best[r["bucket"]]["value"]):
                best[r["bucket"]] = r
        lines.append(f"## {name} (best layer+l2 per bucket vs turn-index control; best is max over the sweep, mildly optimistic)")
        lines.append("")
        lines.append("| bucket | layer | l2 | metric | value | control (turn-index only) | n |")
        lines.append("|---|---|---|---|---|---|---|")
        controls = {r["bucket"]: r for r in rows if r["target"] == name and r["layer"] == -1}
        for bucket in bucket_names:
            if bucket in best and best[bucket]["layer"] != -1:
                r = best[bucket]
                c = controls.get(bucket, {}).get("value", "n/a")
                lines.append(f"| {bucket} | {r['layer']} | {r['l2']} | {r['metric']} | {r['value']} | {c} | {r['n']} |")
        lines.append("")
    (out_dir / "probe_summary.md").write_text("\n".join(lines))
    logger.info("wrote %s and probe_summary.md (%d rows)", csv_path, len(rows))


def export(args: argparse.Namespace) -> None:
    activations = Path(args.activations)
    blob = torch.load(activations, map_location="cpu", weights_only=False)
    features: torch.Tensor = blob["features"]
    layer_indices: list[int] = blob["layer_indices"]
    meta: list[dict] = blob["meta"]
    if args.layer not in layer_indices:
        raise ValueError(f"layer {args.layer} not found in activations; available layers: {layer_indices}")

    layer_pos = layer_indices.index(args.layer)
    x = features[:, layer_pos, :].float().numpy()
    y = np.array([m["rollout_reward"] >= 0.999 for m in meta], dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("rollout_success target needs both positive and negative rows")

    mean, std = x.mean(axis=0), x.std(axis=0)
    x_standardized = (x - mean) / (std + 1e-6)
    wb = _fit_logistic(x_standardized, y, args.l2)
    w_prime, b_prime = fold_standardization(wb[:-1], float(wb[-1]), mean, std)
    in_sample_auroc = auroc(y.astype(bool), x @ w_prime + b_prime)

    out_path = Path(args.out) if args.out else activations.parent / f"frozen_probe_layer{args.layer}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_prime_tensor = torch.as_tensor(w_prime, dtype=torch.float32, device=torch.device("cpu")).contiguous()
    torch.save(
        {
            "version": 1,
            "target": "rollout_success",
            "model_path": blob["model_path"],
            "layer_index": int(args.layer),
            "l2": float(args.l2),
            "activations": str(activations),
            "w_prime": w_prime_tensor,
            "b_prime": float(b_prime),
            "n_rows": int(len(y)),
            "n_pos": int(y.sum()),
            "in_sample_auroc": float(in_sample_auroc),
        },
        out_path,
    )
    logger.info(
        "exported frozen rollout_success probe layer=%d l2=%s n=%d n_pos=%d auroc=%.4f to %s",
        args.layer, args.l2, len(y), int(y.sum()), in_sample_auroc, out_path,
    )


# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="dump turn-start activations (needs a GPU)")
    p_ext.add_argument("--exp-dir", required=True)
    p_ext.add_argument("--model-path", required=True)
    p_ext.add_argument("--out", default=None)
    p_ext.add_argument("--max-rollouts", type=int, default=400)
    p_ext.add_argument("--all-turns", action="store_true",
                       help="one forward per sampled turn over its own prompt (mid-rollout coverage) "
                            "instead of one forward per rollout (final turn only)")
    p_ext.add_argument("--turns-per-rollout", type=int, default=12)
    p_ext.add_argument("--layer-stride", type=int, default=4)
    p_ext.add_argument("--max-seq-len", type=int, default=65536)
    p_ext.add_argument("--device", default="cuda")
    p_ext.set_defaults(func=extract)

    p_fit = sub.add_parser("fit", help="fit probes on saved activations (CPU)")
    p_fit.add_argument("--activations", required=True)
    p_fit.add_argument("--out-dir", default=None)
    p_fit.add_argument("--folds", type=int, default=5)
    p_fit.add_argument("--l2-sweep", default="1,10,100,1000")
    p_fit.add_argument("--min-bucket-size", type=int, default=50)
    p_fit.set_defaults(func=fit)

    p_export = sub.add_parser("export", help="export a frozen rollout_success probe artifact")
    p_export.add_argument("--activations", required=True)
    p_export.add_argument("--layer", type=int, default=31)
    p_export.add_argument("--l2", type=float, default=1000.0)
    p_export.add_argument("--out", default=None)
    p_export.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
