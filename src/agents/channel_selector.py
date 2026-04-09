# src/agents/channel_selector.py

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from agents.agent_base import BaseAgent
from agents.schemas import ChannelSelectorInput, ChannelSelectorOutput
from core.schemas import TimeSeriesSample
from utils.math_utils import cosine_sim, euclidean_sq, zscore_list


class ChannelSelectorAgent(BaseAgent):
    """
    Dataset-level channel selector.

    Purpose
    -------
    This module selects top-k informative channels from the training set and is
    intended to run before:
    - memory_build_pipeline
    - inference_pipeline

    Core scoring
    ------------
    For each channel d:
      B) prototype margin:
         mean_pairwise_centroid_distance / mean_within_class_spread
      C) 1NN leave-one-out accuracy (optionally probed on a subset)

    Final fused score:
      alpha * zscore(B) + (1 - alpha) * zscore(C)

    Notes
    -----
    - For univariate data, the selector degenerates naturally to channel [0].
    - Diversity pruning is supported as an optional post-processing step.
    - This agent is dataset-level rather than sample-level.
    """

    def validate_input(self, input_data: Any) -> None:
        if not isinstance(input_data, ChannelSelectorInput):
            raise TypeError(
                f"{self.name}: input_data must be ChannelSelectorInput, "
                f"but got {type(input_data).__name__}."
            )

    def _run_impl(
        self,
        input_data: ChannelSelectorInput,
        context: dict[str, Any] | None = None,
    ) -> ChannelSelectorOutput:
        samples = input_data.train_samples

        self.log_info(
            context,
            "ChannelSelector '%s': start selection on %d training samples",
            self.name,
            len(samples),
        )
        self.log_event(
            context,
            event_type="channel_selector_start",
            payload={
                "agent_name": self.name,
                "num_train_samples": len(samples),
                "top_k": input_data.top_k,
                "max_len": input_data.max_len,
                "z_norm": input_data.z_norm,
                "alpha": input_data.alpha,
                "nn_eval_samples": input_data.nn_eval_samples,
                "diversity_threshold": input_data.diversity_threshold,
                "task_type": input_data.task_spec.task_type.value,
            },
        )

        data_mode = samples[0].data_mode
        num_channels = samples[0].num_channels

        self._validate_samples_consistency(samples)

        # Univariate shortcut
        if data_mode.value == "univariate" or num_channels == 1:
            self.log_info(
                context,
                "ChannelSelector '%s': univariate bypass, selecting channel [0]",
                self.name,
            )
            self.log_event(
                context,
                event_type="channel_selector_bypass",
                payload={
                    "agent_name": self.name,
                    "reason": "univariate",
                    "selected_channel_ids": [0],
                },
            )

            return ChannelSelectorOutput(
                selected_channel_ids=[0],
                ranked_channel_ids=[0],
                channel_scores={0: 1.0},
                score_details={
                    0: {
                        "score_fused": 1.0,
                        "score_B_prototype_margin": 1.0,
                        "score_C_1nn_acc": 1.0,
                        "debug_B": {"note": "univariate_bypass"},
                        "debug_C": {"note": "univariate_bypass"},
                    }
                },
                selection_applied=False,
                metadata={
                    "num_total_channels": 1,
                    "top_k": 1,
                    "data_mode": data_mode.value,
                    "task_type": input_data.task_spec.task_type.value,
                },
            )

        per_channel_values, labels = self._preprocess_train_per_channel(
            train_samples=samples,
            max_len=input_data.max_len,
            z_norm=input_data.z_norm,
        )

        dims = sorted(per_channel_values.keys())

        self.log_info(
            context,
            "ChannelSelector '%s': evaluating %d channels",
            self.name,
            len(dims),
        )

        score_b_list: list[float] = []
        score_c_list: list[float] = []
        debug_b: dict[int, dict[str, Any]] = {}
        debug_c: dict[int, dict[str, Any]] = {}
        centroid_stacks: dict[int, list[float]] = {}

        for dim in dims:
            channel_data = per_channel_values[dim]

            score_b, dbg_b = self._score_prototype_margin(channel_data, labels)
            score_b_list.append(score_b)
            debug_b[dim] = dbg_b

            score_c, dbg_c = self._score_1nn_accuracy(
                channel_data=channel_data,
                labels=labels,
                nn_eval_samples=input_data.nn_eval_samples,
                seed=input_data.random_seed + dim,
            )
            score_c_list.append(score_c)
            debug_c[dim] = dbg_c

            centroid_stacks[dim] = self._build_centroid_stack(channel_data, labels)

        z_b = zscore_list(score_b_list)
        z_c = zscore_list(score_c_list)

        fused_scores: list[tuple[int, float]] = []
        score_details: dict[int, dict[str, Any]] = {}

        for idx, dim in enumerate(dims):
            fused = float(input_data.alpha * z_b[idx] + (1.0 - input_data.alpha) * z_c[idx])
            fused_scores.append((dim, fused))

            score_details[dim] = {
                "score_fused": fused,
                "score_B_prototype_margin": float(score_b_list[idx]),
                "score_C_1nn_acc": float(score_c_list[idx]),
                "debug_B": debug_b[dim],
                "debug_C": debug_c[dim],
            }

        fused_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_channel_ids = [dim for dim, _ in fused_scores]

        top_k = min(input_data.top_k, len(ranked_channel_ids))

        if input_data.diversity_threshold is None or input_data.diversity_threshold <= 0:
            selected_channel_ids = ranked_channel_ids[:top_k]
        else:
            selected_channel_ids = self._apply_diversity_pruning(
                ranked_channel_ids=ranked_channel_ids,
                centroid_stacks=centroid_stacks,
                budget_k=top_k,
                diversity_threshold=float(input_data.diversity_threshold),
            )

        channel_scores = {dim: score_details[dim]["score_fused"] for dim in ranked_channel_ids}

        self.log_info(
            context,
            "ChannelSelector '%s': selected top-%d channels %s",
            self.name,
            len(selected_channel_ids),
            selected_channel_ids,
        )
        self.log_event(
            context,
            event_type="channel_selector_end",
            payload={
                "agent_name": self.name,
                "num_total_channels": num_channels,
                "num_selected_channels": len(selected_channel_ids),
                "selected_channel_ids": selected_channel_ids,
                "ranked_channel_ids": ranked_channel_ids,
                "task_type": input_data.task_spec.task_type.value,
                "data_mode": data_mode.value,
            },
        )

        return ChannelSelectorOutput(
            selected_channel_ids=selected_channel_ids,
            ranked_channel_ids=ranked_channel_ids,
            channel_scores=channel_scores,
            score_details=score_details,
            selection_applied=True,
            metadata={
                "num_total_channels": num_channels,
                "num_selected_channels": len(selected_channel_ids),
                "top_k": top_k,
                "alpha": float(input_data.alpha),
                "max_len": input_data.max_len,
                "z_norm": input_data.z_norm,
                "nn_eval_samples": input_data.nn_eval_samples,
                "diversity_threshold": input_data.diversity_threshold,
                "random_seed": input_data.random_seed,
                "task_type": input_data.task_spec.task_type.value,
                "data_mode": data_mode.value,
            },
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_samples_consistency(self, samples: list[TimeSeriesSample]) -> None:
        if len(samples) == 0:
            raise ValueError(f"{self.name}: train_samples must not be empty.")

        first_mode = samples[0].data_mode
        first_num_channels = samples[0].num_channels

        for sample in samples:
            if sample.data_mode != first_mode:
                raise ValueError(
                    f"{self.name}: mixed data_mode detected in training samples."
                )
            if sample.num_channels != first_num_channels:
                raise ValueError(
                    f"{self.name}: inconsistent num_channels detected across samples."
                )
            if sample.y is None:
                raise ValueError(
                    f"{self.name}: channel selection requires labeled training samples."
                )

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    def _preprocess_train_per_channel(
        self,
        train_samples: list[TimeSeriesSample],
        max_len: int | None,
        z_norm: bool,
    ) -> tuple[dict[int, list[list[float]]], list[int]]:
        """
        Return
        ------
        per_channel_values:
            dict[channel_id] -> list of sample vectors
        labels:
            list[int], aligned with sample order
        """
        num_channels = train_samples[0].num_channels
        per_channel_values: dict[int, list[list[float]]] = {
            dim: [] for dim in range(num_channels)
        }
        labels: list[int] = []

        for sample in train_samples:
            labels.append(int(sample.y))

            values_2d = self._ensure_2d(sample.x)
            for dim in range(num_channels):
                seq = values_2d[:, dim].tolist()
                seq = self._downsample_1d(seq, max_len)
                if z_norm:
                    seq = self._z_norm_1d(seq)
                per_channel_values[dim].append(seq)

        return per_channel_values, labels

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_prototype_margin(
        self,
        channel_data: list[list[float]],
        labels: list[int],
    ) -> tuple[float, dict[str, Any]]:
        by_cls: dict[int, list[list[float]]] = defaultdict(list)
        for x, y in zip(channel_data, labels):
            by_cls[int(y)].append(x)

        classes = sorted(by_cls.keys())
        if len(classes) <= 1:
            return 0.0, {"note": "single_class", "classes": classes}

        centroids: dict[int, list[float]] = {}
        for cls_id, xs in by_cls.items():
            if len(xs) == 0:
                centroids[cls_id] = []
                continue
            arr = np.asarray(xs, dtype=float)
            centroids[cls_id] = arr.mean(axis=0).tolist()

        spreads: list[float] = []
        for cls_id, xs in by_cls.items():
            mu = centroids[cls_id]
            if len(xs) == 0 or len(mu) == 0:
                continue
            dsum = 0.0
            for v in xs:
                dsum += math.sqrt(euclidean_sq(v, mu))
            spreads.append(dsum / max(1, len(xs)))

        mean_within = sum(spreads) / max(1, len(spreads))

        pair_dists: list[float] = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                ci, cj = classes[i], classes[j]
                ai, aj = centroids[ci], centroids[cj]
                if ai and aj:
                    pair_dists.append(math.sqrt(euclidean_sq(ai, aj)))

        mean_between = sum(pair_dists) / max(1, len(pair_dists))
        score = mean_between / (mean_within + 1e-8)

        dbg = {
            "classes": classes,
            "mean_between": float(mean_between),
            "mean_within": float(mean_within),
            "num_pairs": len(pair_dists),
        }
        return float(score), dbg

    def _score_1nn_accuracy(
        self,
        channel_data: list[list[float]],
        labels: list[int],
        nn_eval_samples: int,
        seed: int = 42,
    ) -> tuple[float, dict[str, Any]]:
        n = len(channel_data)
        if n <= 1:
            return 0.0, {"note": "too_few_samples"}

        idxs = list(range(n))
        rnd = random.Random(seed)

        if nn_eval_samples <= 0 or nn_eval_samples >= n:
            probe = idxs
        else:
            probe = rnd.sample(idxs, nn_eval_samples)

        correct = 0
        for i in probe:
            xi = channel_data[i]
            yi = labels[i]

            best_d = None
            best_j = None

            for j in idxs:
                if j == i:
                    continue
                d = euclidean_sq(xi, channel_data[j])
                if best_d is None or d < best_d:
                    best_d = d
                    best_j = j

            if best_j is not None and labels[best_j] == yi:
                correct += 1

        acc = correct / max(1, len(probe))
        return float(acc), {"N_probe": len(probe)}

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------
    def _build_centroid_stack(
        self,
        channel_data: list[list[float]],
        labels: list[int],
    ) -> list[float]:
        by_cls: dict[int, list[list[float]]] = defaultdict(list)
        for x, y in zip(channel_data, labels):
            by_cls[int(y)].append(x)

        classes = sorted(by_cls.keys())
        stack: list[float] = []

        for cls_id in classes:
            xs = by_cls[cls_id]
            if not xs:
                continue
            arr = np.asarray(xs, dtype=float)
            mu = arr.mean(axis=0).tolist()
            stack.extend(mu)

        return stack

    def _apply_diversity_pruning(
        self,
        ranked_channel_ids: list[int],
        centroid_stacks: dict[int, list[float]],
        budget_k: int,
        diversity_threshold: float,
    ) -> list[int]:
        selected: list[int] = []

        for dim in ranked_channel_ids:
            vec_d = centroid_stacks[dim]
            ok = True

            for kept_dim in selected:
                sim = cosine_sim(vec_d, centroid_stacks[kept_dim])
                if sim > diversity_threshold:
                    ok = False
                    break

            if ok:
                selected.append(dim)

            if len(selected) >= budget_k:
                break

        return selected

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------
    def _ensure_2d(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"{self.name}: sample.x must be 1D or 2D, got ndim={arr.ndim}.")
        return arr

    def _downsample_1d(self, values: Sequence[float], max_len: int | None) -> list[float]:
        if max_len is None or max_len <= 0 or len(values) <= max_len:
            return [float(v) for v in values]

        n = len(values)
        if max_len == 1:
            return [float(values[0])]

        idxs = [round(i * (n - 1) / (max_len - 1)) for i in range(max_len)]
        return [float(values[i]) for i in idxs]

    def _z_norm_1d(self, values: Sequence[float]) -> list[float]:
        n = len(values)
        if n == 0:
            return []

        mu = sum(values) / n
        var = sum((v - mu) * (v - mu) for v in values) / n

        if var <= 1e-12:
            return [float(v - mu) for v in values]

        sd = math.sqrt(var)
        return [float((v - mu) / sd) for v in values]