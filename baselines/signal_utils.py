from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.signal
from scipy.io import wavfile

from .morse_rules import MorseRun


@dataclass(frozen=True)
class ThresholdTrace:
    raw_scores: np.ndarray
    normalized_scores: np.ndarray
    smoothed_scores: np.ndarray
    high_threshold: float
    low_threshold: float
    mask: np.ndarray


def load_audio_mono(file_path: str, sample_rate: int) -> np.ndarray:
    source_sample_rate, audio = wavfile.read(file_path)
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if audio.dtype.kind in {"i", "u"}:
        max_value = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max(max_value, 1.0)
    else:
        audio = audio.astype(np.float32)

    if source_sample_rate != sample_rate and audio.size > 0:
        gcd = int(np.gcd(int(source_sample_rate), int(sample_rate)))
        up = int(sample_rate // gcd)
        down = int(source_sample_rate // gcd)
        audio = scipy.signal.resample_poly(audio, up=up, down=down).astype(np.float32)

    if audio.size == 0:
        return audio

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak
    return audio


def estimate_dominant_frequency(
    audio: np.ndarray,
    sample_rate: int,
    search_low_hz: float,
    search_high_hz: float,
) -> float | None:
    if audio.size == 0:
        return None

    n_fft = 4096
    while n_fft < audio.size and n_fft < 65536:
        n_fft *= 2

    usable = audio[: min(audio.size, n_fft)]
    if usable.size < n_fft:
        usable = np.pad(usable, (0, n_fft - usable.size))

    window = np.hanning(usable.size)
    spectrum = np.abs(np.fft.rfft(usable * window)) ** 2
    freqs = np.fft.rfftfreq(usable.size, d=1.0 / sample_rate)
    mask = (freqs >= search_low_hz) & (freqs <= search_high_hz)
    if not np.any(mask):
        return None

    band_power = spectrum[mask]
    if band_power.size == 0:
        return None

    peak_index = int(np.argmax(band_power))
    peak_power = float(band_power[peak_index])
    baseline_power = float(np.median(band_power))
    if peak_power <= max(1e-10, baseline_power * 2.0):
        return None

    band_freqs = freqs[mask]
    return float(band_freqs[peak_index])


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    lowcut: float,
    highcut: float,
    order: int,
) -> np.ndarray:
    if audio.size == 0:
        return audio

    nyquist = 0.5 * sample_rate
    low = max(0.0, float(lowcut)) / nyquist
    high = min(float(highcut), nyquist - 1.0) / nyquist
    if not (0.0 < low < high < 1.0):
        return audio

    b, a = scipy.signal.butter(int(order), [low, high], btype="band")
    try:
        return scipy.signal.filtfilt(b, a, audio).astype(np.float32)
    except ValueError:
        return scipy.signal.lfilter(b, a, audio).astype(np.float32)


def frame_signal(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")

    if audio.size == 0:
        return np.zeros((0, frame_length), dtype=np.float32)

    if audio.size < frame_length:
        padded = np.pad(audio, (0, frame_length - audio.size))
        return padded.reshape(1, frame_length).astype(np.float32)

    num_frames = 1 + int(np.ceil((audio.size - frame_length) / hop_length))
    total_length = frame_length + (num_frames - 1) * hop_length
    if total_length > audio.size:
        audio = np.pad(audio, (0, total_length - audio.size))

    frames = np.stack(
        [audio[start : start + frame_length] for start in range(0, total_length - frame_length + 1, hop_length)],
        axis=0,
    )
    return frames.astype(np.float32)


def compute_short_time_energy(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.mean(np.square(frames), axis=1).astype(np.float32)


def smooth_sequence(values: np.ndarray, window_size: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    if window_size <= 1:
        return values.astype(np.float32)

    kernel = np.ones(int(window_size), dtype=np.float32)
    kernel /= float(np.sum(kernel))
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def robust_scale_01(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)

    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    if high <= low + 1e-12:
        return np.zeros_like(values, dtype=np.float32)

    scaled = (values - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def compute_hysteresis_thresholds(
    scores: np.ndarray,
    mode: str,
    hysteresis_high: float,
    hysteresis_low: float,
) -> tuple[float, float]:
    if scores.size == 0:
        return 1.0, 0.0

    mode = mode.lower()
    if mode == "fixed":
        high = float(hysteresis_high)
        low = float(hysteresis_low)
    elif mode == "adaptive_median":
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        spread = max(1.4826 * mad, 1e-6)
        high = median + float(hysteresis_high) * spread
        low = median + float(hysteresis_low) * spread
    else:
        p10, p90 = np.percentile(scores, [10, 90])
        span = max(float(p90 - p10), 1e-6)
        high = float(p10) + float(hysteresis_high) * span
        low = float(p10) + float(hysteresis_low) * span

    high = float(np.clip(high, 0.0, 1.0))
    low = float(np.clip(low, 0.0, high))
    return high, low


def hysteresis_binarize(scores: np.ndarray, high_threshold: float, low_threshold: float) -> np.ndarray:
    if scores.size == 0:
        return np.zeros(0, dtype=bool)

    mask = np.zeros(scores.shape[0], dtype=bool)
    state = False
    for index, score in enumerate(scores):
        if not state and score >= high_threshold:
            state = True
        elif state and score <= low_threshold:
            state = False
        mask[index] = state
    return mask


def fill_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0 or mask.size == 0:
        return mask.copy()

    result = mask.copy()
    start = 0
    while start < result.size:
        state = bool(result[start])
        end = start + 1
        while end < result.size and bool(result[end]) == state:
            end += 1

        if not state and 0 < start < result.size and end < result.size and (end - start) <= max_gap_frames:
            result[start:end] = True
        start = end
    return result


def suppress_short_tones(mask: np.ndarray, min_tone_frames: int) -> np.ndarray:
    if min_tone_frames <= 1 or mask.size == 0:
        return mask.copy()

    result = mask.copy()
    start = 0
    while start < result.size:
        state = bool(result[start])
        end = start + 1
        while end < result.size and bool(result[end]) == state:
            end += 1

        if state and (end - start) < min_tone_frames:
            result[start:end] = False
        start = end
    return result


def finalize_detection_scores(
    raw_scores: np.ndarray,
    threshold_mode: str,
    hysteresis_high: float,
    hysteresis_low: float,
    smoothing_window: int,
    min_tone_frames: int,
    max_gap_frames: int,
) -> ThresholdTrace:
    raw_scores = np.asarray(raw_scores, dtype=np.float32)
    normalized = robust_scale_01(raw_scores)
    smoothed = smooth_sequence(normalized, smoothing_window)
    high_threshold, low_threshold = compute_hysteresis_thresholds(
        smoothed, threshold_mode, hysteresis_high, hysteresis_low
    )
    mask = hysteresis_binarize(smoothed, high_threshold, low_threshold)
    mask = fill_short_gaps(mask, max_gap_frames=max_gap_frames)
    mask = suppress_short_tones(mask, min_tone_frames=min_tone_frames)
    return ThresholdTrace(
        raw_scores=raw_scores,
        normalized_scores=normalized,
        smoothed_scores=smoothed,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        mask=mask,
    )


def mask_to_runs(mask: np.ndarray, hop_seconds: float) -> list[MorseRun]:
    if mask.size == 0:
        return []

    runs: list[MorseRun] = []
    current_state = bool(mask[0])
    current_length = 1

    for value in mask[1:]:
        if bool(value) == current_state:
            current_length += 1
            continue

        runs.append(
            MorseRun(
                is_tone=current_state,
                frames=current_length,
                duration_s=float(current_length * hop_seconds),
            )
        )
        current_state = bool(value)
        current_length = 1

    runs.append(
        MorseRun(
            is_tone=current_state,
            frames=current_length,
            duration_s=float(current_length * hop_seconds),
        )
    )
    return runs


def build_frequency_candidates(
    target_frequency_hz: float,
    tolerance_hz: float,
    neighboring_bins: int,
    sample_rate: int,
    frame_length: int,
) -> list[float]:
    if target_frequency_hz <= 0.0:
        return []

    bin_resolution = float(sample_rate) / float(frame_length)
    bin_radius = max(int(neighboring_bins), int(np.ceil(max(0.0, tolerance_hz) / bin_resolution)))
    offsets = np.arange(-bin_radius, bin_radius + 1, dtype=np.int32)
    candidates = target_frequency_hz + offsets * bin_resolution
    candidates = candidates[(candidates > 0.0) & (candidates < (sample_rate / 2.0))]
    return [float(value) for value in np.unique(np.round(candidates, 6))]


def goertzel_power(
    frames: np.ndarray,
    sample_rate: int,
    frequencies_hz: Sequence[float],
) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 1:
        frames = frames.reshape(1, -1)
    if frames.size == 0:
        return np.zeros((0, len(frequencies_hz)), dtype=np.float64)
    if len(frequencies_hz) == 0:
        return np.zeros((frames.shape[0], 0), dtype=np.float64)

    num_frames, frame_length = frames.shape
    output = np.zeros((num_frames, len(frequencies_hz)), dtype=np.float64)

    for freq_index, frequency_hz in enumerate(frequencies_hz):
        k = int(round((frame_length * float(frequency_hz)) / sample_rate))
        k = int(np.clip(k, 0, frame_length // 2))
        omega = (2.0 * np.pi * k) / frame_length
        coeff = 2.0 * np.cos(omega)

        s_prev = np.zeros(num_frames, dtype=np.float64)
        s_prev2 = np.zeros(num_frames, dtype=np.float64)
        for sample_index in range(frame_length):
            s = frames[:, sample_index] + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s

        output[:, freq_index] = s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2

    return output
