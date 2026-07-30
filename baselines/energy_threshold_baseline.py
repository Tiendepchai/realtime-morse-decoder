from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .morse_rules import (
    FAILURE_TONE_DETECTION,
    DurationRules,
    decode_runs_to_text,
)
from .signal_utils import (
    bandpass_filter,
    estimate_dominant_frequency,
    finalize_detection_scores,
    frame_signal,
    load_audio_mono,
    mask_to_runs,
    compute_short_time_energy,
)
from .timing_estimation import DotEstimationConfig, estimate_dot_unit


@dataclass(frozen=True)
class EnergyThresholdBaselineConfig:
    sample_rate: int = 16000
    lowcut: float = 400.0
    highcut: float = 1200.0
    filter_order: int = 4
    frame_length_ms: float = 20.0
    hop_length_ms: float = 10.0
    smoothing_window: int = 5
    threshold_mode: str = "adaptive_percentile"
    hysteresis_high: float = 0.55
    hysteresis_low: float = 0.35
    min_tone_duration_ms: float = 30.0
    fill_gap_duration_ms: float = 20.0
    auto_frequency: bool = True
    frequency_search_low_hz: float = 400.0
    frequency_search_high_hz: float = 1200.0
    band_margin_hz: float = 140.0
    target_frequency_hz: Optional[float] = None
    use_manifest_frequency: bool = False
    duration_rules: DurationRules = field(default_factory=DurationRules)
    dot_estimation: DotEstimationConfig = field(default_factory=DotEstimationConfig)


class EnergyThresholdBaseline:
    method_name = "energy_threshold"

    def __init__(self, config: Optional[EnergyThresholdBaselineConfig] = None):
        self.config = config or EnergyThresholdBaselineConfig()

    def _resolve_target_frequency(
        self,
        audio: np.ndarray,
        sample_metadata: Optional[dict[str, Any]],
    ) -> float | None:
        if self.config.use_manifest_frequency and sample_metadata is not None:
            manifest_freq = sample_metadata.get("freq")
            if manifest_freq is not None and str(manifest_freq) != "":
                return float(manifest_freq)

        if self.config.target_frequency_hz is not None:
            return float(self.config.target_frequency_hz)

        if self.config.auto_frequency:
            return estimate_dominant_frequency(
                audio,
                sample_rate=self.config.sample_rate,
                search_low_hz=self.config.frequency_search_low_hz,
                search_high_hz=self.config.frequency_search_high_hz,
            )

        return None

    def _resolve_band(self, target_frequency_hz: float | None) -> tuple[float, float]:
        if target_frequency_hz is None:
            return float(self.config.lowcut), float(self.config.highcut)

        lowcut = max(20.0, target_frequency_hz - self.config.band_margin_hz)
        highcut = min((self.config.sample_rate / 2.0) - 20.0, target_frequency_hz + self.config.band_margin_hz)
        if lowcut >= highcut:
            return float(self.config.lowcut), float(self.config.highcut)
        return float(lowcut), float(highcut)

    def decode_audio(
        self,
        audio: np.ndarray,
        sample_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return {
                "prediction": "",
                "is_failure": True,
                "failure_type": FAILURE_TONE_DETECTION,
                "failure_details": ["empty_audio"],
                "estimated_frequency_hz": None,
            }

        frame_length = max(1, int(round(self.config.sample_rate * self.config.frame_length_ms / 1000.0)))
        hop_length = max(1, int(round(self.config.sample_rate * self.config.hop_length_ms / 1000.0)))
        min_tone_frames = max(1, int(round(self.config.min_tone_duration_ms / self.config.hop_length_ms)))
        max_gap_frames = max(1, int(round(self.config.fill_gap_duration_ms / self.config.hop_length_ms)))

        estimated_frequency_hz = self._resolve_target_frequency(audio, sample_metadata)
        lowcut, highcut = self._resolve_band(estimated_frequency_hz)
        filtered = bandpass_filter(
            audio,
            sample_rate=self.config.sample_rate,
            lowcut=lowcut,
            highcut=highcut,
            order=self.config.filter_order,
        )

        frames = frame_signal(filtered, frame_length=frame_length, hop_length=hop_length)
        energies = compute_short_time_energy(frames)
        detection = finalize_detection_scores(
            raw_scores=energies,
            threshold_mode=self.config.threshold_mode,
            hysteresis_high=self.config.hysteresis_high,
            hysteresis_low=self.config.hysteresis_low,
            smoothing_window=self.config.smoothing_window,
            min_tone_frames=min_tone_frames,
            max_gap_frames=max_gap_frames,
        )
        runs = mask_to_runs(detection.mask, hop_seconds=float(hop_length) / self.config.sample_rate)

        if not np.any(detection.mask):
            return {
                "prediction": "",
                "is_failure": True,
                "failure_type": FAILURE_TONE_DETECTION,
                "failure_details": ["no_tone_frames_after_thresholding"],
                "estimated_frequency_hz": estimated_frequency_hz,
                "dot_unit_ms": None,
                "high_threshold": detection.high_threshold,
                "low_threshold": detection.low_threshold,
            }

        dot_unit_s, timing_debug = estimate_dot_unit(runs, self.config.dot_estimation)
        outcome = decode_runs_to_text(runs, dot_unit_s=dot_unit_s, rules=self.config.duration_rules)

        return {
            "prediction": outcome.prediction,
            "is_failure": outcome.is_failure,
            "failure_type": outcome.failure_type,
            "failure_details": outcome.failure_details,
            "estimated_frequency_hz": estimated_frequency_hz,
            "dot_unit_ms": None if outcome.dot_unit_s is None else outcome.dot_unit_s * 1000.0,
            "high_threshold": detection.high_threshold,
            "low_threshold": detection.low_threshold,
            "num_frames": int(frames.shape[0]),
            "num_tone_frames": int(np.sum(detection.mask)),
            "band_lowcut_hz": lowcut,
            "band_highcut_hz": highcut,
            "timing_debug": timing_debug,
            "morse_tokens": outcome.morse_tokens,
        }

    def decode_file(
        self,
        audio_path: str,
        sample_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        audio = load_audio_mono(audio_path, sample_rate=self.config.sample_rate)
        return self.decode_audio(audio, sample_metadata=sample_metadata)
