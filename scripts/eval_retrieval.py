#!/usr/bin/env python3
"""Evaluate retrieval quality on a persisted memory bank, independent of the
LLM reasoner and aggregation.

For every entry we do leave-one-out k-NN over the statistic view (the STAT
retriever's representation): retrieve its k nearest neighbours among all other
entries of the same channel and measure how well neighbour labels match the
query label. This isolates "is retrieval finding same-label neighbours?" from
the rest of the pipeline, so it can be used to compare distance/normalization
settings quickly (no LLM calls).

Normalization is fit on the full gallery (matching real inference, where the
memory bank is the gallery and normalization stats are computed over it); the
query itself is only excluded from the neighbour search, not from the fit.

Metrics (per channel and macro-averaged over channels):
  - knn_acc          : k-NN majority-vote prediction == true label
  - label_prec@k     : fraction of the k neighbours that share the query label
  - positive P/R/F1  : treating `--positive-label` as the positive class
  - baseline_acc     : majority-class prior (sanity floor)

Example:
  python scripts/eval_retrieval.py --compare \
    --memory-bank-path outputs/memory/SKAB_memory_build/memory_bank.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np

from memory.memory_store import load_memory_bank_jsonl
from retrieval.scoring import apply_normalization


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--memory-bank-path",
        type=str,
        default="outputs/memory/SKAB_memory_build/memory_bank.jsonl",
    )
    p.add_argument("--k", type=int, default=5, help="number of neighbours")
    p.add_argument("--distance", type=str, default="cosine", choices=["cosine", "l2"])
    p.add_argument(
        "--normalize",
        type=str,
        default="zscore",
        choices=["none", "zscore", "robust", "log1p_robust"],
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Also evaluate normalize=none and print both, to show the delta.",
    )
    p.add_argument(
        "--channels",
        nargs="*",
        type=int,
        default=None,
        help="Restrict to these channel ids (default: all).",
    )
    p.add_argument(
        "--positive-label",
        type=str,
        default="1",
        help="Label treated as the positive class for P/R/F1 (compared as string).",
    )
    p.add_argument("--save-json", type=str, default=None)
    return p


def _load_by_channel(path: str) -> dict[int, list[tuple[str, Any, dict]]]:
    bank = load_memory_bank_jsonl(path)
    by_ch: dict[int, list[tuple[str, Any, dict]]] = defaultdict(list)
    for e in bank.get_all():
        stat = e.statistic_view
        if not isinstance(stat, dict) or not stat:
            continue
        by_ch[int(e.channel_id)].append((e.sample_id, e.label, stat))
    return by_ch


def _pairwise_distances(Xn: np.ndarray, distance: str) -> np.ndarray:
    """Full distance matrix mirroring retrieval.scoring semantics."""
    if distance == "cosine":
        norms = np.linalg.norm(Xn, axis=1)
        denom = np.outer(norms, norms)
        sim = Xn @ Xn.T
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where(denom == 0.0, 0.0, sim / denom)
        return 1.0 - cos
    if distance == "l2":
        sq = np.sum(Xn * Xn, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (Xn @ Xn.T)
        return np.sqrt(np.maximum(d2, 0.0))
    raise ValueError(f"Unsupported distance: {distance}")


def _eval_channel(
    rows: list[tuple[str, Any, dict]],
    k: int,
    distance: str,
    normalize: str,
    positive_label: str,
) -> dict[str, float]:
    feats = sorted(rows[0][2].keys())
    X = np.array([[float(r[2][f]) for f in feats] for r in rows], dtype=float)
    y = np.array([str(r[1]) for r in rows])

    Xn, _ = apply_normalization(X, X[:1], normalize)
    D = _pairwise_distances(Xn, distance)
    np.fill_diagonal(D, np.inf)  # leave-one-out: never retrieve self

    n = len(rows)
    kk = min(k, n - 1)
    nn_idx = np.argsort(D, axis=1)[:, :kk]

    preds = []
    label_prec = np.zeros(n)
    for i in range(n):
        neigh = y[nn_idx[i]]
        vals, counts = np.unique(neigh, return_counts=True)
        preds.append(vals[np.argmax(counts)])
        label_prec[i] = float(np.mean(neigh == y[i]))
    preds = np.array(preds)

    acc = float(np.mean(preds == y))
    pos = positive_label
    tp = int(np.sum((preds == pos) & (y == pos)))
    fp = int(np.sum((preds == pos) & (y != pos)))
    fn = int(np.sum((preds != pos) & (y == pos)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    _, class_counts = np.unique(y, return_counts=True)
    baseline = float(class_counts.max() / n)

    return {
        "n": n,
        "knn_acc": acc,
        "label_prec_at_k": float(np.mean(label_prec)),
        "pos_precision": prec,
        "pos_recall": rec,
        "pos_f1": f1,
        "baseline_acc": baseline,
    }


def _run(by_ch, args, normalize: str) -> dict[str, Any]:
    channels = args.channels if args.channels else sorted(by_ch.keys())
    per_channel: dict[str, Any] = {}
    for ch in channels:
        per_channel[str(ch)] = _eval_channel(
            by_ch[ch], args.k, args.distance, normalize, args.positive_label
        )
    keys = ["knn_acc", "label_prec_at_k", "pos_precision", "pos_recall", "pos_f1", "baseline_acc"]
    macro = {kk: float(np.mean([per_channel[str(c)][kk] for c in channels])) for kk in keys}
    return {"normalize": normalize, "distance": args.distance, "k": args.k,
            "macro": macro, "per_channel": per_channel}


def _print_result(res: dict[str, Any]) -> None:
    m = res["macro"]
    print(f"\n[normalize={res['normalize']} | distance={res['distance']} | k={res['k']}]")
    print(f"  macro  knn_acc={m['knn_acc']:.3f}  label_prec@k={m['label_prec_at_k']:.3f}  "
          f"pos_P={m['pos_precision']:.3f} pos_R={m['pos_recall']:.3f} pos_F1={m['pos_f1']:.3f}  "
          f"(baseline_acc={m['baseline_acc']:.3f})")
    for ch, s in res["per_channel"].items():
        print(f"    ch{ch}: acc={s['knn_acc']:.3f} prec@k={s['label_prec_at_k']:.3f} "
              f"F1={s['pos_f1']:.3f} (n={s['n']})")


def main() -> None:
    args = _build_parser().parse_args()
    by_ch = _load_by_channel(args.memory_bank_path)
    if not by_ch:
        raise SystemExit(f"No statistic-view entries found in {args.memory_bank_path}")
    print(f"Loaded {sum(len(v) for v in by_ch.values())} entries "
          f"across {len(by_ch)} channels from {args.memory_bank_path}")

    results = []
    if args.compare and args.normalize != "none":
        results.append(_run(by_ch, args, "none"))
    results.append(_run(by_ch, args, args.normalize))

    for res in results:
        _print_result(res)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump({"memory_bank_path": args.memory_bank_path, "results": results}, f, indent=2)
        print(f"\nSaved -> {args.save_json}")


if __name__ == "__main__":
    main()
