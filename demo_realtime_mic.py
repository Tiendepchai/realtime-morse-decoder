from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F

try:
    import sounddevice as sd
except ImportError:
    sd = None

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.energy_threshold_baseline import EnergyThresholdBaseline, EnergyThresholdBaselineConfig
from baselines.goertzel_baseline import GoertzelBaseline, GoertzelBaselineConfig
from src.features.audio_processor import AudioProcessor
from src.inference import _load_model_for_inference
from src.utils.device import DEVICE_CHOICES, resolve_torch_device, synchronize_device
from src.utils.text import CHARS, TextTransform

METHODS = ("energy_threshold", "goertzel", "crnn", "conformer")
DECODE_MODES = ("segment", "sliding")


def find_latest_checkpoint(model_type: str) -> str:
    candidates = sorted(Path("experiments/checkpoints").glob(f"{model_type}/*/best_model.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pth found for model_type={model_type}")
    return str(candidates[-1])


def resolve_input_device(device_text: str) -> str | int | None:
    raw = str(device_text).strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    return raw


def compute_rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float32)))


def compute_activity_peak(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    return float(np.percentile(np.abs(audio), 95))


def normalize_prediction_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def extract_incremental_text(previous_text: str, current_text: str) -> str:
    if not current_text:
        return ""
    if not previous_text:
        return current_text
    if current_text == previous_text:
        return ""
    if current_text.startswith(previous_text):
        return current_text[len(previous_text) :]

    max_overlap = min(len(previous_text), len(current_text))
    for overlap_size in range(max_overlap, 0, -1):
        if previous_text[-overlap_size:] == current_text[:overlap_size]:
            return current_text[overlap_size:]
    return current_text


def predictions_are_compatible(previous_text: str, current_text: str) -> bool:
    previous = normalize_prediction_text(previous_text)
    current = normalize_prediction_text(current_text)
    if not previous or not current:
        return False
    return current.startswith(previous) or previous.startswith(current)


def should_accept_deep_prediction(
    text: str,
    emitted_frame_count: int,
    emitted_frame_ratio: float,
    mean_emitted_confidence: float,
    mean_blank_probability: float,
    min_emitted_frames: int = 3,
    min_emitted_frame_ratio: float = 0.01,
    min_mean_emitted_confidence: float = 0.45,
    max_mean_blank_probability: float = 0.995,
) -> tuple[bool, str]:
    if not normalize_prediction_text(text):
        return False, "empty_text"
    if emitted_frame_count < min_emitted_frames:
        return False, "too_few_emitted_frames"
    if emitted_frame_ratio < min_emitted_frame_ratio:
        return False, "emitted_frame_ratio_too_low"
    if mean_emitted_confidence < min_mean_emitted_confidence:
        return False, "low_emitted_confidence"
    if mean_blank_probability > max_mean_blank_probability:
        return False, "blank_probability_too_high"
    return True, "ok"


def should_accept_traditional_prediction(
    text: str,
    meta: dict[str, Any],
    min_tone_frames: int = 4,
    min_tone_frame_ratio: float = 0.01,
) -> tuple[bool, str]:
    if not normalize_prediction_text(text):
        return False, "empty_text"
    if bool(meta.get("is_failure")):
        return False, "decoder_failure"

    num_tone_frames = int(meta.get("num_tone_frames", 0) or 0)
    num_frames = int(meta.get("num_frames", 0) or 0)
    tone_frame_ratio = float(num_tone_frames) / max(1, num_frames)
    if num_tone_frames < min_tone_frames:
        return False, "too_few_tone_frames"
    if tone_frame_ratio < min_tone_frame_ratio:
        return False, "tone_frame_ratio_too_low"
    return True, "ok"


@dataclass
class ActivityDecision:
    should_infer: bool
    rms: float
    activity_peak: float
    noise_floor_rms: float
    threshold_rms: float
    threshold_peak: float
    reason: str


@dataclass(frozen=True)
class DebugPaths:
    root_dir: Path
    session_wav: Path
    session_json: Path
    segments_dir: Path


def build_debug_paths(debug_dir: str | Path) -> DebugPaths:
    root_dir = Path(debug_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = root_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    return DebugPaths(
        root_dir=root_dir,
        session_wav=root_dir / "session.wav",
        session_json=root_dir / "session.json",
        segments_dir=segments_dir,
    )


@dataclass(frozen=True)
class TuningProfile:
    source_path: Path
    model_type: str
    pre_pad_s: float
    post_pad_s: float
    speed: float
    band_low_hz: float
    band_high_hz: float
    clip_threshold: float
    companding_exponent: float


def load_tuning_profile(path_text: str, preferred_method: str | None = None) -> TuningProfile:
    profile_path = Path(path_text).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    preferred_method = str(preferred_method).strip().lower() if preferred_method else ""

    selected = None
    if preferred_method:
        best_by_model = payload.get("best_by_model")
        if isinstance(best_by_model, dict):
            selected = best_by_model.get(preferred_method)
        if selected is None:
            best = payload.get("best")
            if isinstance(best, dict) and str(best.get("model_type", "")).strip().lower() == preferred_method:
                selected = best
    else:
        selected = payload.get("best")

    best = selected
    if not isinstance(best, dict):
        if preferred_method:
            raise ValueError(f"Missing tuning result for method={preferred_method!r} in {profile_path}")
        raise ValueError(f"Missing object field 'best' in {profile_path}")

    try:
        return TuningProfile(
            source_path=profile_path,
            model_type=str(best["model_type"]).strip().lower(),
            pre_pad_s=float(best["pre_pad_s"]),
            post_pad_s=float(best["post_pad_s"]),
            speed=float(best["speed"]),
            band_low_hz=float(best["band_low_hz"]),
            band_high_hz=float(best["band_high_hz"]),
            clip_threshold=float(best["clip_threshold"]),
            companding_exponent=float(best["companding_exponent"]),
        )
    except KeyError as error:
        missing_key = str(error).strip("'")
        raise ValueError(f"Missing key '{missing_key}' in {profile_path}") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid tuning profile data in {profile_path}: {error}") from error


def apply_tuning_profile(audio: np.ndarray, sample_rate: int, profile: TuningProfile | None) -> np.ndarray:
    processed = np.asarray(audio, dtype=np.float32).reshape(-1)
    if processed.size == 0:
        return processed
    if profile is None:
        return processed.copy()

    if profile.speed > 0.0 and not np.isclose(profile.speed, 1.0):
        new_length = max(1, int(round(processed.size / profile.speed)))
        processed = scipy.signal.resample(processed, new_length).astype(np.float32)
    else:
        processed = processed.astype(np.float32, copy=True)

    nyquist = float(sample_rate) / 2.0
    low = float(profile.band_low_hz) / nyquist
    high = float(profile.band_high_hz) / nyquist
    if 0.0 < low < high < 1.0:
        b, a = scipy.signal.butter(4, [low, high], btype="band")
        padlen = 3 * (max(len(a), len(b)) - 1)
        if processed.size > padlen:
            processed = scipy.signal.filtfilt(b, a, processed).astype(np.float32)

    clip_threshold = max(float(profile.clip_threshold), 1e-4)
    processed = np.tanh(processed / clip_threshold) * clip_threshold

    if not np.isclose(profile.companding_exponent, 1.0):
        processed = np.sign(processed) * np.power(np.abs(processed), profile.companding_exponent)

    peak = float(np.max(np.abs(processed)))
    if peak > 0.0:
        processed = processed / peak
    return processed.astype(np.float32, copy=False)


class AdaptiveSilenceGate:
    def __init__(
        self,
        base_threshold: float,
        activity_multiplier: float = 3.0,
        peak_multiplier: float = 1.5,
        min_peak_threshold: float = 0.015,
        warmup_steps: int = 4,
        noise_floor_decay: float = 0.15,
        noise_floor_rise: float = 0.03,
    ):
        self.base_threshold = float(base_threshold)
        self.activity_multiplier = float(activity_multiplier)
        self.peak_multiplier = float(peak_multiplier)
        self.min_peak_threshold = float(min_peak_threshold)
        self.warmup_steps = max(0, int(warmup_steps))
        self.noise_floor_decay = float(noise_floor_decay)
        self.noise_floor_rise = float(noise_floor_rise)

        self.noise_floor_rms = 0.0
        self.observed_steps = 0

    def _update_noise_floor(self, rms: float, should_infer: bool) -> None:
        if self.observed_steps == 0:
            self.noise_floor_rms = rms
            return

        if should_infer and rms >= self.noise_floor_rms:
            return

        alpha = self.noise_floor_decay if rms <= self.noise_floor_rms else self.noise_floor_rise
        self.noise_floor_rms = ((1.0 - alpha) * self.noise_floor_rms) + (alpha * rms)

    def evaluate(self, audio_chunk: np.ndarray) -> ActivityDecision:
        rms = compute_rms(audio_chunk)
        activity_peak = compute_activity_peak(audio_chunk)
        threshold_rms = max(self.base_threshold, self.noise_floor_rms * self.activity_multiplier)
        threshold_peak = max(self.min_peak_threshold, threshold_rms * self.peak_multiplier)

        in_warmup = self.observed_steps < self.warmup_steps
        if in_warmup:
            should_infer = False
            reason = "warmup"
        elif rms < threshold_rms:
            should_infer = False
            reason = "rms_below_threshold"
        elif activity_peak < threshold_peak:
            should_infer = False
            reason = "peak_below_threshold"
        else:
            should_infer = True
            reason = "active_signal"

        self._update_noise_floor(rms, should_infer=should_infer)
        decision = ActivityDecision(
            should_infer=should_infer,
            rms=rms,
            activity_peak=activity_peak,
            noise_floor_rms=self.noise_floor_rms,
            threshold_rms=threshold_rms,
            threshold_peak=threshold_peak,
            reason=reason,
        )
        self.observed_steps += 1
        return decision


@dataclass
class PredictionOutput:
    text: str
    meta: dict[str, Any]


class BaseRealtimePredictor:
    method_name: str
    segment_pre_padding_samples: int = 0
    segment_post_padding_samples: int = 0

    def predict(self, audio_chunk: np.ndarray) -> PredictionOutput:
        raise NotImplementedError


class RealtimeDebugRecorder:
    def __init__(
        self,
        sample_rate: int,
        session_wav_path: str = "",
        debug_dir: str = "",
    ):
        self.sample_rate = int(sample_rate)
        self.session_wav_path = Path(session_wav_path).expanduser().resolve() if str(session_wav_path).strip() else None
        self.debug_paths = build_debug_paths(debug_dir) if str(debug_dir).strip() else None
        if self.session_wav_path is None and self.debug_paths is not None:
            self.session_wav_path = self.debug_paths.session_wav
        if self.session_wav_path is not None:
            self.session_wav_path.parent.mkdir(parents=True, exist_ok=True)

        self.session_chunks: list[np.ndarray] = []
        self.segment_records: list[dict[str, Any]] = []
        self.segment_index = 0

    @property
    def enabled(self) -> bool:
        return self.session_wav_path is not None or self.debug_paths is not None

    def append_session_audio(self, chunk: np.ndarray) -> None:
        if not self.enabled:
            return
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        self.session_chunks.append(chunk.copy())

    def save_segment(
        self,
        audio_chunk: np.ndarray,
        prediction_text: str,
        activity: ActivityDecision,
        method_name: str,
    ) -> str | None:
        if self.debug_paths is None:
            return None

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if audio_chunk.size == 0:
            return None

        self.segment_index += 1
        segment_path = self.debug_paths.segments_dir / f"segment_{self.segment_index:03d}.wav"
        sf.write(segment_path, audio_chunk, self.sample_rate)
        self.segment_records.append(
            {
                "index": self.segment_index,
                "path": str(segment_path),
                "method": method_name,
                "prediction": prediction_text,
                "duration_s": float(audio_chunk.size) / float(self.sample_rate),
                "rms": activity.rms,
                "activity_peak": activity.activity_peak,
                "noise_floor_rms": activity.noise_floor_rms,
                "threshold_rms": activity.threshold_rms,
                "threshold_peak": activity.threshold_peak,
            }
        )
        return str(segment_path)

    def finalize(
        self,
        method_name: str,
        decode_mode: str,
    ) -> tuple[str | None, str | None]:
        session_wav = None
        if self.session_wav_path is not None and self.session_chunks:
            session_audio = (
                self.session_chunks[0]
                if len(self.session_chunks) == 1
                else np.concatenate(self.session_chunks).astype(np.float32, copy=False)
            )
            sf.write(self.session_wav_path, session_audio, self.sample_rate)
            session_wav = str(self.session_wav_path)

        session_json = None
        if self.debug_paths is not None:
            metadata = {
                "sample_rate": self.sample_rate,
                "method": method_name,
                "decode_mode": decode_mode,
                "session_wav": session_wav,
                "segments": self.segment_records,
            }
            self.debug_paths.session_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            session_json = str(self.debug_paths.session_json)

        return session_wav, session_json


class DeepModelRealtimePredictor(BaseRealtimePredictor):
    def __init__(
        self,
        model_type: str,
        model_path: str,
        sample_rate: int,
        device_name: str,
        tuning_profile: TuningProfile | None = None,
    ):
        self.method_name = model_type
        self.sample_rate = sample_rate
        self.device = resolve_torch_device(device_name)
        self.tuning_profile = tuning_profile
        if self.tuning_profile is not None and self.tuning_profile.model_type != model_type:
            raise ValueError(
                f"Tuning profile model_type={self.tuning_profile.model_type!r} does not match --method {model_type!r}"
            )

        self.segment_pre_padding_samples = (
            max(0, int(round(self.tuning_profile.pre_pad_s * sample_rate))) if self.tuning_profile is not None else 0
        )
        self.segment_post_padding_samples = (
            max(0, int(round(self.tuning_profile.post_pad_s * sample_rate))) if self.tuning_profile is not None else 0
        )
        low_cut = self.tuning_profile.band_low_hz if self.tuning_profile is not None else 400.0
        high_cut = self.tuning_profile.band_high_hz if self.tuning_profile is not None else 1200.0
        self.processor = AudioProcessor(sample_rate=sample_rate, n_mels=64, low_cut=low_cut, high_cut=high_cut)
        self.text_transform = TextTransform()

        vocab_size = len(CHARS)
        self.model = _load_model_for_inference(model_path, model_type, vocab_size, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_path = model_path

    def predict(self, audio_chunk: np.ndarray) -> PredictionOutput:
        tuned_audio = apply_tuning_profile(audio_chunk, self.sample_rate, self.tuning_profile)
        cleaned = self.processor.clean_audio(tuned_audio)
        if cleaned.size == 0:
            return PredictionOutput(text="", meta={"reason": "empty_cleaned_audio"})

        log_mel = self.processor.compute_log_mel(cleaned)
        if log_mel.size == 0:
            return PredictionOutput(text="", meta={"reason": "empty_log_mel"})

        features = self.processor.apply_cmvn(log_mel)
        features = np.clip(features, -10.0, 10.0)

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        features_tensor = features_tensor.permute(0, 1, 3, 2).to(self.device)

        with torch.no_grad():
            input_lengths = torch.tensor([features_tensor.shape[3]], dtype=torch.long, device=self.device)
            logits = self.model(features_tensor, input_lengths)
            output = F.log_softmax(logits, dim=2).permute(1, 0, 2)

        synchronize_device(self.device)

        probabilities = output.exp()
        frame_probabilities, arg_maxes = torch.max(probabilities, dim=2)
        frame_probabilities = frame_probabilities[:, 0]
        arg_maxes = arg_maxes[:, 0]
        blank_probabilities = probabilities[:, 0, 0]
        emitted_mask = arg_maxes != 0
        emitted_frame_count = int(emitted_mask.sum().item())
        emitted_frame_ratio = float(emitted_mask.float().mean().item())
        mean_blank_probability = float(blank_probabilities.mean().item())
        mean_emitted_confidence = (
            float(frame_probabilities[emitted_mask].mean().item())
            if emitted_frame_count > 0
            else 0.0
        )

        decoded_indices: list[int] = []
        prev_idx = 0
        for idx in arg_maxes:
            token = int(idx.item())
            if token != 0 and token != prev_idx:
                decoded_indices.append(token)
            prev_idx = token

        text = self.text_transform.int_to_text(decoded_indices)
        accepted, rejection_reason = should_accept_deep_prediction(
            text=text,
            emitted_frame_count=emitted_frame_count,
            emitted_frame_ratio=emitted_frame_ratio,
            mean_emitted_confidence=mean_emitted_confidence,
            mean_blank_probability=mean_blank_probability,
        )
        if not accepted:
            text = ""

        return PredictionOutput(
            text=text,
            meta={
                "num_frames": int(features.shape[0]),
                "device": str(self.device),
                "checkpoint_path": self.model_path,
                "tuning_profile": str(self.tuning_profile.source_path) if self.tuning_profile is not None else "",
                "tuning_speed": self.tuning_profile.speed if self.tuning_profile is not None else 1.0,
                "tuning_band_low_hz": self.tuning_profile.band_low_hz if self.tuning_profile is not None else self.processor.low_cut,
                "tuning_band_high_hz": self.tuning_profile.band_high_hz if self.tuning_profile is not None else self.processor.high_cut,
                "tuning_clip_threshold": self.tuning_profile.clip_threshold if self.tuning_profile is not None else 0.95,
                "emitted_frame_count": emitted_frame_count,
                "emitted_frame_ratio": emitted_frame_ratio,
                "mean_emitted_confidence": mean_emitted_confidence,
                "mean_blank_probability": mean_blank_probability,
                "accepted_prediction": accepted,
                "rejection_reason": rejection_reason,
            },
        )


class TraditionalRealtimePredictor(BaseRealtimePredictor):
    def __init__(self, method_name: str, sample_rate: int, tone_frequency_hz: float):
        self.method_name = method_name
        if method_name == "energy_threshold":
            self.decoder = EnergyThresholdBaseline(
                EnergyThresholdBaselineConfig(
                    sample_rate=sample_rate,
                    target_frequency_hz=tone_frequency_hz if tone_frequency_hz > 0 else None,
                    auto_frequency=tone_frequency_hz <= 0,
                )
            )
        elif method_name == "goertzel":
            self.decoder = GoertzelBaseline(
                GoertzelBaselineConfig(
                    sample_rate=sample_rate,
                    target_frequency_hz=tone_frequency_hz if tone_frequency_hz > 0 else None,
                    auto_frequency=tone_frequency_hz <= 0,
                )
            )
        else:
            raise ValueError(f"Unsupported traditional method: {method_name}")

    def predict(self, audio_chunk: np.ndarray) -> PredictionOutput:
        result = self.decoder.decode_audio(np.asarray(audio_chunk, dtype=np.float32))
        text = str(result.get("prediction", ""))
        accepted, rejection_reason = should_accept_traditional_prediction(text=text, meta=result)
        if not accepted:
            text = ""

        num_tone_frames = int(result.get("num_tone_frames", 0) or 0)
        num_frames = int(result.get("num_frames", 0) or 0)
        result = dict(result)
        result["tone_frame_ratio"] = float(num_tone_frames) / max(1, num_frames)
        result["accepted_prediction"] = accepted
        result["rejection_reason"] = rejection_reason
        return PredictionOutput(
            text=text,
            meta=result,
        )


def build_predictor(
    method: str,
    sample_rate: int,
    model_path: str,
    device_name: str,
    tone_frequency_hz: float,
    tuning_profile: TuningProfile | None = None,
) -> BaseRealtimePredictor:
    if method in {"crnn", "conformer"}:
        resolved_model_path = model_path or find_latest_checkpoint(method)
        return DeepModelRealtimePredictor(
            model_type=method,
            model_path=resolved_model_path,
            sample_rate=sample_rate,
            device_name=device_name,
            tuning_profile=tuning_profile,
        )

    return TraditionalRealtimePredictor(
        method_name=method,
        sample_rate=sample_rate,
        tone_frequency_hz=tone_frequency_hz,
    )


class RealtimeMicDemo:
    def __init__(
        self,
        predictor: BaseRealtimePredictor,
        sample_rate: int,
        buffer_duration: float,
        step_duration: float,
        chunk_duration: float,
        silence_threshold: float,
        reset_silence_steps: int,
        confirm_steps: int,
        decode_mode: str,
        min_segment_duration: float,
        max_segment_duration: float,
        debug_recorder: RealtimeDebugRecorder | None,
        input_device: str | int | None,
        show_full_predictions: bool,
        show_meta: bool,
    ):
        self.predictor = predictor
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.step_duration = step_duration
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.reset_silence_steps = max(1, int(reset_silence_steps))
        self.confirm_steps = max(1, int(confirm_steps))
        self.decode_mode = str(decode_mode).strip().lower()
        self.min_segment_samples = max(1, int(round(float(min_segment_duration) * sample_rate)))
        self.max_segment_samples = max(self.min_segment_samples, int(round(float(max_segment_duration) * sample_rate)))
        self.debug_recorder = debug_recorder
        self.input_device = input_device
        self.show_full_predictions = show_full_predictions
        self.show_meta = show_meta

        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self.buffer_size = max(1, int(round(buffer_duration * sample_rate)))
        self.block_size = max(1, int(round(chunk_duration * sample_rate)))
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        self.last_prediction = ""
        self.silent_step_count = 0
        self.last_inference_time = 0.0
        self.pending_prediction = ""
        self.pending_prediction_count = 0
        self.activity_gate = AdaptiveSilenceGate(base_threshold=silence_threshold)
        self.step_chunks: list[np.ndarray] = []
        self.segment_chunks: list[np.ndarray] = []
        self.segment_sample_count = 0
        self.segment_silence_steps = 0
        self.segment_trailing_chunks: list[np.ndarray] = []
        self.segment_trailing_sample_count = 0
        self.recent_step_chunks: list[np.ndarray] = []
        self.recent_step_sample_count = 0
        self.last_activity_decision: ActivityDecision | None = None

    def audio_callback(self, indata: np.ndarray, frames: int, callback_time: Any, status: Any) -> None:
        del frames, callback_time
        if status:
            print(status, file=sys.stderr)
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        self.queue.put(mono.copy())

    def _append_audio(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return
        if self.debug_recorder is not None:
            self.debug_recorder.append_session_audio(chunk)
        self.step_chunks.append(chunk.copy())
        if chunk.size >= self.buffer_size:
            self.audio_buffer = chunk[-self.buffer_size :].copy()
            return
        self.audio_buffer = np.roll(self.audio_buffer, -chunk.size)
        self.audio_buffer[-chunk.size :] = chunk

    def _consume_step_audio(self) -> np.ndarray:
        if not self.step_chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self.step_chunks) == 1:
            chunk = self.step_chunks[0]
            self.step_chunks = []
            return chunk
        combined = np.concatenate(self.step_chunks).astype(np.float32, copy=False)
        self.step_chunks = []
        return combined

    def _remember_recent_step_audio(self, audio: np.ndarray) -> None:
        max_samples = getattr(self.predictor, "segment_pre_padding_samples", 0)
        if max_samples <= 0:
            return

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return

        self.recent_step_chunks.append(audio.copy())
        self.recent_step_sample_count += int(audio.size)
        while self.recent_step_sample_count > max_samples and self.recent_step_chunks:
            overflow = self.recent_step_sample_count - max_samples
            oldest = self.recent_step_chunks[0]
            if oldest.size <= overflow:
                self.recent_step_chunks.pop(0)
                self.recent_step_sample_count -= int(oldest.size)
                continue
            self.recent_step_chunks[0] = oldest[overflow:].copy()
            self.recent_step_sample_count -= int(overflow)
            break

    def _take_recent_step_context(self, max_samples: int) -> np.ndarray:
        if max_samples <= 0 or not self.recent_step_chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self.recent_step_chunks) == 1:
            combined = self.recent_step_chunks[0]
        else:
            combined = np.concatenate(self.recent_step_chunks).astype(np.float32, copy=False)
        if combined.size <= max_samples:
            return combined.copy()
        return combined[-max_samples:].copy()

    def _take_buffer_pre_context(self, audio_chunk: np.ndarray, step_audio: np.ndarray, max_samples: int) -> np.ndarray:
        if max_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        step_audio = np.asarray(step_audio, dtype=np.float32).reshape(-1)
        if audio_chunk.size == 0:
            return np.zeros(0, dtype=np.float32)

        if 0 < step_audio.size < audio_chunk.size:
            context = audio_chunk[: -step_audio.size]
        elif step_audio.size >= audio_chunk.size:
            context = np.zeros(0, dtype=np.float32)
        else:
            context = audio_chunk

        if context.size <= max_samples:
            return context.copy()
        return context[-max_samples:].copy()

    def _append_segment_audio(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        self.segment_chunks.append(audio.copy())
        self.segment_sample_count += int(audio.size)

        while self.segment_sample_count > self.max_segment_samples and self.segment_chunks:
            overflow = self.segment_sample_count - self.max_segment_samples
            oldest = self.segment_chunks[0]
            if oldest.size <= overflow:
                self.segment_chunks.pop(0)
                self.segment_sample_count -= int(oldest.size)
                continue
            self.segment_chunks[0] = oldest[overflow:].copy()
            self.segment_sample_count -= int(overflow)
            break

    def _append_segment_trailing_audio(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        self.segment_trailing_chunks.append(audio.copy())
        self.segment_trailing_sample_count += int(audio.size)

    def _consume_segment_trailing_audio(self, max_samples: int | None = None) -> np.ndarray:
        if not self.segment_trailing_chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self.segment_trailing_chunks) == 1:
            combined = self.segment_trailing_chunks[0]
        else:
            combined = np.concatenate(self.segment_trailing_chunks).astype(np.float32, copy=False)
        self.segment_trailing_chunks = []
        self.segment_trailing_sample_count = 0
        if max_samples is None or max_samples <= 0 or combined.size <= max_samples:
            return combined.copy()
        return combined[:max_samples].copy()

    def _consume_segment_audio(self, post_context: np.ndarray | None = None) -> np.ndarray:
        if not self.segment_chunks:
            return np.zeros(0, dtype=np.float32)
        parts: list[np.ndarray] = []
        if len(self.segment_chunks) == 1:
            parts.append(self.segment_chunks[0])
        else:
            parts.append(np.concatenate(self.segment_chunks).astype(np.float32, copy=False))
        if post_context is not None:
            post_context = np.asarray(post_context, dtype=np.float32).reshape(-1)
            if post_context.size > 0:
                parts.append(post_context)
        audio = parts[0] if len(parts) == 1 else np.concatenate(parts).astype(np.float32, copy=False)
        self.segment_chunks = []
        self.segment_sample_count = 0
        self.segment_silence_steps = 0
        self.segment_trailing_chunks = []
        self.segment_trailing_sample_count = 0
        return audio

    def _reset_pending_prediction(self) -> None:
        self.pending_prediction = ""
        self.pending_prediction_count = 0

    def _maybe_reset_after_silence(self, activity: ActivityDecision) -> bool:
        if activity.should_infer:
            self.silent_step_count = 0
            return False

        self.silent_step_count += 1
        if self.silent_step_count >= self.reset_silence_steps:
            self.last_prediction = ""
            self._reset_pending_prediction()
        return True

    def _finalize_segment_prediction(self, audio_chunk: np.ndarray, activity: ActivityDecision) -> None:
        if audio_chunk.size < self.min_segment_samples:
            return

        started_at = time.perf_counter()
        prediction = self.predictor.predict(audio_chunk)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        current_text = normalize_prediction_text(prediction.text)
        if not current_text:
            return

        prediction.meta = dict(prediction.meta)
        prediction.meta["segment_duration_s"] = float(audio_chunk.size) / float(self.sample_rate)
        self._print_meta(current_text, latency_ms, prediction.meta, activity)

        segment_path = None
        if self.debug_recorder is not None:
            segment_path = self.debug_recorder.save_segment(
                audio_chunk=audio_chunk,
                prediction_text=current_text,
                activity=activity,
                method_name=self.predictor.method_name,
            )
            if segment_path:
                print(f"Saved segment debug WAV: {segment_path}", file=sys.stderr, flush=True)
        self._handle_segment_prediction_output(
            text=current_text,
            latency_ms=latency_ms,
            meta=prediction.meta,
            activity=activity,
            audio_chunk=audio_chunk,
            segment_path=segment_path,
        )

    def _promote_stable_prediction(self, current_text: str) -> str:
        current = current_text.strip()
        if not current:
            return ""

        if self.confirm_steps <= 1:
            self.pending_prediction = current
            self.pending_prediction_count = 1
            return current

        if not self.pending_prediction:
            self.pending_prediction = current
            self.pending_prediction_count = 1
            return ""

        if predictions_are_compatible(self.pending_prediction, current):
            if len(normalize_prediction_text(current)) >= len(normalize_prediction_text(self.pending_prediction)):
                self.pending_prediction = current
            self.pending_prediction_count += 1
        else:
            self.pending_prediction = current
            self.pending_prediction_count = 1
            return ""

        if self.pending_prediction_count < self.confirm_steps:
            return ""
        return self.pending_prediction

    def _print_meta(self, text: str, latency_ms: float, meta: dict[str, Any], activity: ActivityDecision) -> None:
        if not self.show_meta:
            return

        summary_parts = [f"latency={latency_ms:.1f}ms"]
        summary_parts.append(f"rms={activity.rms:.4f}")
        summary_parts.append(f"noise={activity.noise_floor_rms:.4f}")
        summary_parts.append(f"thr={activity.threshold_rms:.4f}")
        estimated_frequency = meta.get("estimated_frequency_hz")
        if estimated_frequency is not None:
            summary_parts.append(f"freq={float(estimated_frequency):.1f}Hz")
        dot_unit_ms = meta.get("dot_unit_ms")
        if dot_unit_ms is not None:
            summary_parts.append(f"dot={float(dot_unit_ms):.1f}ms")
        num_frames = meta.get("num_frames")
        if num_frames is not None:
            summary_parts.append(f"frames={int(num_frames)}")
        tone_frame_ratio = meta.get("tone_frame_ratio")
        if tone_frame_ratio is not None:
            summary_parts.append(f"tone_ratio={float(tone_frame_ratio):.3f}")
        mean_emitted_confidence = meta.get("mean_emitted_confidence")
        if mean_emitted_confidence is not None:
            summary_parts.append(f"conf={float(mean_emitted_confidence):.3f}")
        rejection_reason = meta.get("rejection_reason")
        if rejection_reason and rejection_reason != "ok":
            summary_parts.append(f"gate={rejection_reason}")
        segment_duration_s = meta.get("segment_duration_s")
        if segment_duration_s is not None:
            summary_parts.append(f"segment={float(segment_duration_s):.2f}s")

        print(
            f"\n[{self.predictor.method_name}] {text} | " + " | ".join(summary_parts),
            file=sys.stderr,
            flush=True,
        )

    def _handle_segment_prediction_output(
        self,
        text: str,
        latency_ms: float,
        meta: dict[str, Any],
        activity: ActivityDecision,
        audio_chunk: np.ndarray,
        segment_path: str | None,
    ) -> None:
        del audio_chunk, segment_path
        self._print_meta(text, latency_ms, meta, activity)
        print(text, flush=True)
        self.last_prediction = text
        self._reset_pending_prediction()

    def _handle_sliding_prediction_output(
        self,
        stable_text: str,
        latency_ms: float,
        meta: dict[str, Any],
        activity: ActivityDecision,
    ) -> None:
        self._print_meta(stable_text, latency_ms, meta, activity)

        if self.show_full_predictions:
            if stable_text != self.last_prediction:
                print(f"\n[{self.predictor.method_name}] {stable_text}", flush=True)
                self.last_prediction = stable_text
            return

        new_text = extract_incremental_text(self.last_prediction, stable_text)
        if new_text:
            print(new_text, end="", flush=True)
            self.last_prediction = stable_text

    def _run_segment_decode_step(
        self,
        activity: ActivityDecision,
        step_audio: np.ndarray,
        audio_chunk: np.ndarray | None = None,
    ) -> None:
        if activity.should_infer:
            self.silent_step_count = 0
            self.segment_silence_steps = 0
            if not self.segment_chunks:
                pre_context = np.zeros(0, dtype=np.float32)
                if audio_chunk is not None:
                    pre_context = self._take_buffer_pre_context(
                        audio_chunk=audio_chunk,
                        step_audio=step_audio,
                        max_samples=getattr(self.predictor, "segment_pre_padding_samples", 0),
                    )
                if pre_context.size == 0:
                    pre_context = self._take_recent_step_context(getattr(self.predictor, "segment_pre_padding_samples", 0))
                if pre_context.size > 0:
                    self._append_segment_audio(pre_context)
            elif self.segment_trailing_chunks:
                self._append_segment_audio(self._consume_segment_trailing_audio())
            self._append_segment_audio(step_audio)
            return

        if not self.segment_chunks:
            self._maybe_reset_after_silence(activity)
            return

        self.segment_silence_steps += 1
        self._append_segment_trailing_audio(step_audio)
        if self.segment_silence_steps < self.reset_silence_steps:
            return

        post_context = self._consume_segment_trailing_audio(
            max_samples=getattr(self.predictor, "segment_post_padding_samples", 0)
        )
        segment_audio = self._consume_segment_audio(post_context=post_context)
        self._finalize_segment_prediction(segment_audio, activity)

    def _run_inference_step(self) -> None:
        audio_chunk = self.audio_buffer.copy()
        step_audio = self._consume_step_audio()
        activity = self.activity_gate.evaluate(audio_chunk)
        self.last_activity_decision = activity

        try:
            if self.decode_mode == "segment":
                self._run_segment_decode_step(activity, step_audio, audio_chunk=audio_chunk)
                return

            if self._maybe_reset_after_silence(activity):
                return

            started_at = time.perf_counter()
            prediction = self.predictor.predict(audio_chunk)
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            current_text = prediction.text.strip()
            if not current_text:
                return

            stable_text = self._promote_stable_prediction(current_text)
            if not stable_text:
                return

            self._handle_sliding_prediction_output(
                stable_text=stable_text,
                latency_ms=latency_ms,
                meta=prediction.meta,
                activity=activity,
            )
        finally:
            self._remember_recent_step_audio(step_audio)

    def _finalize_pending_segment_before_shutdown(self) -> None:
        if self.decode_mode != "segment" or not self.segment_chunks:
            return

        pending_activity = self.last_activity_decision
        if pending_activity is None:
            pending_activity = self.activity_gate.evaluate(self.audio_buffer.copy())
        print("Finalizing pending segment before exit.", file=sys.stderr, flush=True)
        self._finalize_segment_prediction(self._consume_segment_audio(), pending_activity)

    def start(self) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed in the current Python environment. "
                "Install it with `pip install sounddevice` before using microphone realtime demo."
            )

        print(f"Realtime demo started with method={self.predictor.method_name}")
        print(f"Listening at {self.sample_rate} Hz. Press Ctrl+C to stop.")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.block_size,
                device=self.input_device,
                callback=self.audio_callback,
            ):
                while True:
                    try:
                        chunk = self.queue.get(timeout=0.1)
                        self._append_audio(chunk)
                    except queue.Empty:
                        pass

                    now = time.monotonic()
                    if now - self.last_inference_time >= self.step_duration:
                        self._run_inference_step()
                        self.last_inference_time = now
        except KeyboardInterrupt:
            print("\nStopped realtime demo.")
        except Exception as error:
            print(f"\nError while running realtime demo: {error}", file=sys.stderr)
        finally:
            self._finalize_pending_segment_before_shutdown()
            if self.debug_recorder is not None and self.debug_recorder.enabled:
                session_wav, session_json = self.debug_recorder.finalize(
                    method_name=self.predictor.method_name,
                    decode_mode=self.decode_mode,
                )
                if session_wav:
                    print(f"Saved session WAV: {session_wav}", file=sys.stderr, flush=True)
                if session_json:
                    print(f"Saved debug metadata: {session_json}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone microphone realtime demo for Morse decoding without touching benchmark code.",
    )
    parser.add_argument("--method", default="conformer", choices=METHODS)
    parser.add_argument("--model", default="", help="Checkpoint path for crnn/conformer. Empty means latest checkpoint.")
    parser.add_argument("--decode-mode", default="segment", choices=DECODE_MODES, help="segment waits for silence then decodes the full active region; sliding decodes a rolling window")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--input-device", default="", help="sounddevice input device id or name")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--buffer", type=float, default=3.0, help="Sliding window duration in seconds")
    parser.add_argument("--step", type=float, default=0.75, help="Inference interval in seconds")
    parser.add_argument("--chunk", type=float, default=0.25, help="Microphone read chunk in seconds")
    parser.add_argument("--silence-threshold", type=float, default=0.01, help="RMS threshold below which inference is skipped")
    parser.add_argument("--reset-silence-steps", type=int, default=2, help="How many silent inference steps before incremental text resets")
    parser.add_argument("--confirm-steps", type=int, default=2, help="How many compatible predictions are required before text is printed")
    parser.add_argument("--min-segment", type=float, default=1.2, help="Minimum active segment duration in seconds before a finalized decode is emitted")
    parser.add_argument("--max-segment", type=float, default=24.0, help="Maximum active segment audio kept for a finalized decode")
    parser.add_argument("--record-wav", default="", help="Optional path to save the full microphone session as WAV")
    parser.add_argument("--debug-dir", default="", help="Optional directory to save session.wav, per-segment WAVs, and session.json for later analysis")
    parser.add_argument("--tuning-json", default="", help="Optional tuning_result.json from scripts/tune_recording_to_target.py for deep realtime preprocessing")
    parser.add_argument("--tone-frequency", type=float, default=0.0, help="Fixed tone frequency for baselines in Hz. <=0 means auto-detect")
    parser.add_argument("--show-full-predictions", action="store_true", help="Print the full decoded buffer instead of only incremental text")
    parser.add_argument("--show-meta", action="store_true", help="Print latency and decoder metadata to stderr")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be > 0")
    if args.buffer <= 0:
        parser.error("--buffer must be > 0")
    if args.step <= 0:
        parser.error("--step must be > 0")
    if args.chunk <= 0:
        parser.error("--chunk must be > 0")
    if args.chunk > args.buffer:
        parser.error("--chunk must be <= --buffer")
    if args.confirm_steps <= 0:
        parser.error("--confirm-steps must be > 0")
    if args.min_segment <= 0:
        parser.error("--min-segment must be > 0")
    if args.max_segment < args.min_segment:
        parser.error("--max-segment must be >= --min-segment")
    if args.method in {"energy_threshold", "goertzel"} and args.model:
        parser.error("--model is only used for crnn/conformer")
    if args.tuning_json:
        if args.method not in {"crnn", "conformer"}:
            parser.error("--tuning-json is only supported for crnn/conformer")
        try:
            tuning_profile = load_tuning_profile(args.tuning_json, preferred_method=args.method)
        except Exception as error:
            parser.error(f"--tuning-json could not be loaded: {error}")
        if tuning_profile.model_type != args.method:
            parser.error(
                f"--tuning-json best profile was tuned for method={tuning_profile.model_type}, "
                f"but --method={args.method}"
            )
        setattr(args, "_loaded_tuning_profile", tuning_profile)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    if args.list_devices:
        if sd is None:
            raise SystemExit(
                "sounddevice is not installed in the current Python environment. "
                "Install it with `pip install sounddevice` to list microphone devices."
            )
        print(sd.query_devices())
        return

    predictor = build_predictor(
        method=args.method,
        sample_rate=args.sample_rate,
        model_path=args.model,
        device_name=args.device,
        tone_frequency_hz=args.tone_frequency,
        tuning_profile=getattr(args, "_loaded_tuning_profile", None),
    )
    debug_recorder = RealtimeDebugRecorder(
        sample_rate=args.sample_rate,
        session_wav_path=args.record_wav,
        debug_dir=args.debug_dir,
    )
    if args.method in {"crnn", "conformer"}:
        assert isinstance(predictor, DeepModelRealtimePredictor)
        print(f"Loaded {args.method} checkpoint: {predictor.model_path}")
        print(f"Inference device: {predictor.device}")
        if predictor.tuning_profile is not None:
            print(f"Loaded tuning profile: {predictor.tuning_profile.source_path}")
            print(
                "Tuning: "
                f"speed={predictor.tuning_profile.speed:.2f} "
                f"band={predictor.tuning_profile.band_low_hz:.0f}-{predictor.tuning_profile.band_high_hz:.0f}Hz "
                f"clip={predictor.tuning_profile.clip_threshold:.2f} "
                f"pre_pad={predictor.tuning_profile.pre_pad_s:.2f}s "
                f"post_pad={predictor.tuning_profile.post_pad_s:.2f}s"
            )
    elif args.tone_frequency > 0:
        print(f"Using fixed baseline tone frequency: {args.tone_frequency:.1f} Hz")
    else:
        print("Using automatic tone frequency detection for the baseline.")

    demo = RealtimeMicDemo(
        predictor=predictor,
        sample_rate=args.sample_rate,
        buffer_duration=args.buffer,
        step_duration=args.step,
        chunk_duration=args.chunk,
        silence_threshold=args.silence_threshold,
        reset_silence_steps=args.reset_silence_steps,
        confirm_steps=args.confirm_steps,
        decode_mode=args.decode_mode,
        min_segment_duration=args.min_segment,
        max_segment_duration=args.max_segment,
        debug_recorder=debug_recorder,
        input_device=resolve_input_device(args.input_device),
        show_full_predictions=args.show_full_predictions,
        show_meta=args.show_meta,
    )
    demo.start()


if __name__ == "__main__":
    main()
