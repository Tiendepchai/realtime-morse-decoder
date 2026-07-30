from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .morse_rules import MorseRun


@dataclass(frozen=True)
class DotEstimationConfig:
    lower_quantile: float = 0.35
    min_unit_ms: float = 20.0
    max_unit_ms: float = 400.0
    search_ratio_low: float = 0.5
    search_ratio_high: float = 1.75
    search_steps: int = 41
    tone_weight: float = 1.0
    gap_weight: float = 0.75


def _lower_cluster_center(values: np.ndarray, lower_quantile: float) -> Optional[float]:
    if values.size == 0:
        return None

    quantile = float(np.clip(lower_quantile, 0.05, 0.95))
    cutoff = float(np.quantile(values, quantile))
    cluster = values[values <= cutoff + 1e-12]
    if cluster.size == 0:
        cluster = np.array([float(np.min(values))], dtype=np.float64)
    return float(np.median(cluster))


def _score_unit(
    unit_s: float,
    tone_durations: np.ndarray,
    gap_durations: np.ndarray,
    config: DotEstimationConfig,
) -> float:
    tone_loss = 0.0
    gap_loss = 0.0

    if tone_durations.size:
        tone_units = tone_durations / unit_s
        tone_loss = float(
            np.mean(np.minimum(np.abs(tone_units - 1.0), np.abs(tone_units - 3.0)))
        )

    if gap_durations.size:
        gap_units = gap_durations / unit_s
        gap_loss = float(
            np.mean(
                np.minimum.reduce(
                    [
                        np.abs(gap_units - 1.0),
                        np.abs(gap_units - 3.0),
                        np.abs(gap_units - 7.0),
                    ]
                )
            )
        )

    return config.tone_weight * tone_loss + config.gap_weight * gap_loss


def estimate_dot_unit(
    runs: Sequence[MorseRun],
    config: DotEstimationConfig,
) -> tuple[Optional[float], dict[str, float | int | str | None]]:
    tone_durations = np.asarray(
        [run.duration_s for run in runs if run.is_tone and run.duration_s > 0.0],
        dtype=np.float64,
    )
    gap_durations = np.asarray(
        [run.duration_s for run in runs if not run.is_tone and run.duration_s > 0.0],
        dtype=np.float64,
    )

    debug: dict[str, float | int | str | None] = {
        "num_tone_runs": int(tone_durations.size),
        "num_gap_runs": int(gap_durations.size),
        "initial_unit_s": None,
        "refined_unit_s": None,
        "score": None,
        "reason": None,
    }

    if tone_durations.size == 0 and gap_durations.size == 0:
        debug["reason"] = "no_positive_runs"
        return None, debug

    initial_candidates: list[float] = []
    tone_initial = _lower_cluster_center(tone_durations, config.lower_quantile)
    if tone_initial is not None:
        initial_candidates.append(tone_initial)

    if gap_durations.size:
        gap_core = gap_durations
        if gap_durations.size > 3:
            gap_cutoff = float(np.quantile(gap_durations, 0.75))
            gap_core = gap_durations[gap_durations <= gap_cutoff + 1e-12]
        gap_initial = _lower_cluster_center(gap_core, config.lower_quantile)
        if gap_initial is not None:
            initial_candidates.append(gap_initial)

    if not initial_candidates:
        debug["reason"] = "failed_to_find_initial_candidate"
        return None, debug

    min_unit_s = config.min_unit_ms / 1000.0
    max_unit_s = config.max_unit_ms / 1000.0
    initial_unit_s = float(np.clip(np.median(initial_candidates), min_unit_s, max_unit_s))
    debug["initial_unit_s"] = initial_unit_s

    if tone_durations.size <= 1 and gap_durations.size == 0:
        debug["refined_unit_s"] = initial_unit_s
        debug["score"] = 0.0
        debug["reason"] = "single_tone_fallback"
        return initial_unit_s, debug

    search_low = max(min_unit_s, initial_unit_s * config.search_ratio_low)
    search_high = min(max_unit_s, initial_unit_s * config.search_ratio_high)
    if search_high <= search_low:
        debug["refined_unit_s"] = initial_unit_s
        debug["score"] = 0.0
        debug["reason"] = "degenerate_search_window"
        return initial_unit_s, debug

    candidates = np.linspace(search_low, search_high, max(3, int(config.search_steps)))
    gap_core = gap_durations
    if gap_durations.size > 3:
        gap_cutoff = float(np.quantile(gap_durations, 0.90))
        gap_core = gap_durations[gap_durations <= gap_cutoff + 1e-12]

    scores = np.asarray(
        [_score_unit(candidate, tone_durations, gap_core, config) for candidate in candidates],
        dtype=np.float64,
    )
    best_index = int(np.argmin(scores))
    refined_unit_s = float(candidates[best_index])

    debug["refined_unit_s"] = refined_unit_s
    debug["score"] = float(scores[best_index])
    return refined_unit_s, debug
