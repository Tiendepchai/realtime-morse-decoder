import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import librosa
from src.features.audio_processor import AudioProcessor
from src.models.crnn import CRNN
from src.models.conformer import Conformer, infer_conformer_time_reduction_factor
from src.real_style import infer_gap_formatted_text
from src.utils.device import DEVICE_CHOICES, resolve_torch_device, synchronize_device
from src.utils.text import CHARS, TextTransform
import torch.nn.functional as F


ROUTE_CHOICES = ("direct", "realtime-segment", "realtime-sliding")


@dataclass(frozen=True)
class RouteEmission:
    kind: str
    stable_text: str
    emitted_text: str
    start_s: float
    end_s: float
    latency_ms: float
    meta: dict[str, Any]


@dataclass(frozen=True)
class RoutedInferenceResult:
    route: str
    method: str
    audio_path: str
    sample_rate: int
    final_text: str
    raw_final_text: str
    emissions: list[RouteEmission]


def _build_model(model_type: str, vocab_size: int, conformer_time_reduction: int = 1):
    """Build model by type. No blank_bias at inference (bias=0.0)."""
    num_classes = vocab_size + 1
    if model_type == "conformer":
        return Conformer(
            num_classes=num_classes,
            input_dim=64,
            d_model=256,
            num_layers=8,
            time_reduction_factor=conformer_time_reduction,
        )
    if model_type == "crnn":
        return CRNN(num_classes=num_classes)
    raise ValueError(f"Unknown model_type: {model_type}")


def _load_state_dict(model, model_path, device):
    """Load checkpoint, stripping DataParallel 'module.' prefix if present."""
    checkpoint = torch.load(model_path, map_location="cpu")
    state = {}
    for k, v in checkpoint.items():
        state[k.removeprefix("module.")] = v
    model.load_state_dict(state)
    return model.to(device)


def _load_model_for_inference(model_path, model_type, vocab_size, device):
    checkpoint = torch.load(model_path, map_location="cpu")
    state = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    conformer_time_reduction = 1
    if model_type == "conformer":
        conformer_time_reduction = infer_conformer_time_reduction_factor(state)
    model = _build_model(
        model_type,
        vocab_size,
        conformer_time_reduction=conformer_time_reduction,
    )
    model.load_state_dict(state)
    return model.to(device)


def _find_latest_checkpoint(model_type: str) -> str:
    candidates = sorted(Path("experiments/checkpoints").glob(f"{model_type}/*/best_model.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pth found for model_type={model_type}")
    return str(candidates[-1])


def resolve_model_path_for_inference(model_path: str, model_type: str) -> str:
    path_text = str(model_path).strip()
    if path_text:
        candidate = Path(path_text).expanduser().resolve()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Model checkpoint not found: {candidate}")
    return _find_latest_checkpoint(model_type)


def infer(audio_path, model_path, model_type="conformer", speed=1.0, device_name="auto"):
    device = resolve_torch_device(device_name)
    print(f"Using device: {device}")
    
    # 1. Load Model
    vocab_size = len(CHARS)
    resolved_model_path = resolve_model_path_for_inference(model_path, model_type)
    model = _load_model_for_inference(resolved_model_path, model_type, vocab_size, device)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_type} model from {resolved_model_path}")
    
    # 2. Process Audio
    processor = AudioProcessor(sample_rate=16000, n_mels=64)
    if not os.path.exists(audio_path):
        print(f"Error: File not found {audio_path}")
        return
        
    audio = processor.load_audio(audio_path)
    
    # Apply Time Stretch (Pitch Preserved)
    if speed != 1.0:
        print(f"Applying time stretch: x{speed}")
        try:
            audio = librosa.effects.time_stretch(audio, rate=speed)
        except Exception as e:
            print(f"Time stretch failed: {e}")
        
    cleaned = processor.clean_audio(audio)
    log_mel = processor.compute_log_mel(cleaned)
    features = processor.apply_cmvn(log_mel)
    
    # Match training preprocessing: Clamp features
    features = np.clip(features, -10.0, 10.0)
    
    # To Tensor: (B, 1, F, T)
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # (1, 1, T, F)
    features = features.permute(0, 1, 3, 2)  # (1, 1, F, T)
    features = features.to(device)
    
    # 3. Predict — unified forward(x, input_lengths)
    with torch.no_grad():
        input_lengths = torch.tensor([features.shape[3]], dtype=torch.long).to(device)
        logits = model(features, input_lengths)  # (B, T', C)
        output = F.log_softmax(logits, dim=2)
        output = output.permute(1, 0, 2)  # (T, B, C)
    synchronize_device(device)
        
    # 4. Decode
    text_transform = TextTransform()
    arg_maxes = torch.argmax(output, dim=2)  # (T, 1)
    decode = []
    prev_idx = 0
    for idx in arg_maxes[:, 0]:
        idx = idx.item()
        if idx != 0 and idx != prev_idx:
            decode.append(idx)
        prev_idx = idx
        
    result = text_transform.int_to_text(decode)
    return result


def run_realtime_route_on_audio(
    audio: np.ndarray,
    sample_rate: int,
    predictor: Any,
    route: str,
    buffer_duration: float = 3.0,
    step_duration: float = 0.75,
    chunk_duration: float = 0.25,
    silence_threshold: float = 0.01,
    reset_silence_steps: int = 2,
    confirm_steps: int = 2,
    min_segment_duration: float = 1.2,
    max_segment_duration: float = 24.0,
    show_full_predictions: bool = False,
    show_meta: bool = False,
    audio_path: str = "",
) -> RoutedInferenceResult:
    import demo_realtime_mic as realtime_demo

    decode_mode = "segment" if route == "realtime-segment" else "sliding"
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return RoutedInferenceResult(
            route=route,
            method=str(getattr(predictor, "method_name", "")),
            audio_path=audio_path,
            sample_rate=int(sample_rate),
            final_text="",
            raw_final_text="",
            emissions=[],
        )
    warmup_samples = max(1, int(round(float(step_duration) * float(sample_rate) * 4.0)))
    routed_audio = np.concatenate([np.zeros(warmup_samples, dtype=np.float32), audio]).astype(np.float32, copy=False)
    warmup_offset_s = float(warmup_samples) / float(sample_rate)

    class RoutedFileDemo(realtime_demo.RealtimeMicDemo):
        def __init__(self) -> None:
            super().__init__(
                predictor=predictor,
                sample_rate=sample_rate,
                buffer_duration=buffer_duration,
                step_duration=step_duration,
                chunk_duration=chunk_duration,
                silence_threshold=silence_threshold,
                reset_silence_steps=reset_silence_steps,
                confirm_steps=confirm_steps,
                decode_mode=decode_mode,
                min_segment_duration=min_segment_duration,
                max_segment_duration=max_segment_duration,
                debug_recorder=None,
                input_device=None,
                show_full_predictions=show_full_predictions,
                show_meta=show_meta,
            )
            self.route_events: list[RouteEmission] = []
            self.route_step_end_s = 0.0

        def _route_time_bounds(self, start_s: float, end_s: float) -> tuple[float, float]:
            adjusted_start = max(0.0, float(start_s) - warmup_offset_s)
            adjusted_end = max(0.0, float(end_s) - warmup_offset_s)
            return adjusted_start, adjusted_end

        def _handle_segment_prediction_output(
            self,
            text: str,
            latency_ms: float,
            meta: dict[str, Any],
            activity: Any,
            audio_chunk: np.ndarray,
            segment_path: str | None,
        ) -> None:
            del segment_path
            self._print_meta(text, latency_ms, meta, activity)
            duration_s = float(np.asarray(audio_chunk).size) / float(self.sample_rate)
            end_s = float(self.route_step_end_s)
            start_s = max(0.0, end_s - duration_s)
            start_s, end_s = self._route_time_bounds(start_s, end_s)
            self.route_events.append(
                RouteEmission(
                    kind="segment",
                    stable_text=text,
                    emitted_text=text,
                    start_s=start_s,
                    end_s=end_s,
                    latency_ms=float(latency_ms),
                    meta=dict(meta),
                )
            )
            self.last_prediction = text
            self._reset_pending_prediction()

        def _handle_sliding_prediction_output(
            self,
            stable_text: str,
            latency_ms: float,
            meta: dict[str, Any],
            activity: Any,
        ) -> None:
            self._print_meta(stable_text, latency_ms, meta, activity)

            if self.show_full_predictions:
                if stable_text == self.last_prediction:
                    return
                emitted_text = stable_text
                self.last_prediction = stable_text
            else:
                emitted_text = realtime_demo.extract_incremental_text(self.last_prediction, stable_text)
                if not emitted_text:
                    return
                self.last_prediction = stable_text

            end_s = float(self.route_step_end_s)
            start_s = max(0.0, end_s - float(self.buffer_duration))
            start_s, end_s = self._route_time_bounds(start_s, end_s)
            self.route_events.append(
                RouteEmission(
                    kind="sliding",
                    stable_text=stable_text,
                    emitted_text=emitted_text,
                    start_s=start_s,
                    end_s=end_s,
                    latency_ms=float(latency_ms),
                    meta=dict(meta),
                )
            )

    demo = RoutedFileDemo()
    chunk_samples = demo.block_size
    step_samples = max(1, int(round(float(step_duration) * sample_rate)))
    processed_samples = 0
    samples_since_last_step = 0

    for start in range(0, routed_audio.size, chunk_samples):
        chunk = routed_audio[start : start + chunk_samples]
        demo._append_audio(chunk)
        processed_samples += int(chunk.size)
        samples_since_last_step += int(chunk.size)

        while samples_since_last_step >= step_samples:
            overhang = samples_since_last_step - step_samples
            step_end_samples = processed_samples - overhang
            demo.route_step_end_s = float(step_end_samples) / float(sample_rate)
            demo._run_inference_step()
            samples_since_last_step -= step_samples

    if demo.step_chunks:
        demo.route_step_end_s = float(routed_audio.size) / float(sample_rate)
        demo._run_inference_step()
    demo.route_step_end_s = float(routed_audio.size) / float(sample_rate)
    demo._finalize_pending_segment_before_shutdown()

    final_text = demo.last_prediction.strip()
    if decode_mode == "segment" and demo.route_events:
        final_text = "\n".join(event.stable_text for event in demo.route_events)

    return RoutedInferenceResult(
        route=route,
        method=str(getattr(predictor, "method_name", "")),
        audio_path=audio_path,
        sample_rate=int(sample_rate),
        final_text=final_text,
        raw_final_text=final_text,
        emissions=demo.route_events,
    )


def apply_gap_formatting_to_routed_result(
    result: RoutedInferenceResult,
    audio: np.ndarray,
    sample_rate: int,
    gap_format: str = "relative",
) -> RoutedInferenceResult:
    format_mode = str(gap_format).strip().lower()
    if format_mode != "relative" or result.route != "realtime-segment" or not result.emissions:
        return result

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    new_emissions: list[RouteEmission] = []
    any_changes = False

    for event in result.emissions:
        if event.kind != "segment":
            new_emissions.append(event)
            continue

        start_sample = max(0, int(round(float(event.start_s) * float(sample_rate))))
        end_sample = min(audio.size, max(start_sample + 1, int(round(float(event.end_s) * float(sample_rate)))))
        clip = audio[start_sample:end_sample]
        gap_result = infer_gap_formatted_text(clip, sample_rate=sample_rate, text=event.stable_text)
        formatted_text = str(gap_result.formatted_text or event.stable_text).strip()
        if not formatted_text:
            new_emissions.append(event)
            continue

        meta = dict(event.meta)
        meta["raw_stable_text"] = event.stable_text
        meta["raw_emitted_text"] = event.emitted_text
        meta["gap_format"] = format_mode
        meta["gap_boundary_units"] = gap_result.boundary_units
        meta["gap_word_boundary_flags"] = gap_result.word_boundary_flags
        meta["gap_threshold_units"] = gap_result.threshold_units
        meta["gap_guided_dot_unit_s"] = gap_result.guided_dot_unit_s
        any_changes = any_changes or (formatted_text != str(event.stable_text).strip())

        new_emissions.append(
            RouteEmission(
                kind=event.kind,
                stable_text=formatted_text,
                emitted_text=formatted_text,
                start_s=event.start_s,
                end_s=event.end_s,
                latency_ms=event.latency_ms,
                meta=meta,
            )
        )

    if not any_changes:
        return result

    formatted_final_text = "\n".join(event.stable_text for event in new_emissions)
    return RoutedInferenceResult(
        route=result.route,
        method=result.method,
        audio_path=result.audio_path,
        sample_rate=result.sample_rate,
        final_text=formatted_final_text,
        raw_final_text=result.final_text,
        emissions=new_emissions,
    )


def infer_with_route(
    audio_path: str,
    model_path: str,
    model_type: str = "conformer",
    speed: float = 1.0,
    device_name: str = "auto",
    route: str = "direct",
    route_method: str = "",
    sample_rate: int = 16000,
    buffer_duration: float = 3.0,
    step_duration: float = 0.75,
    chunk_duration: float = 0.25,
    silence_threshold: float = 0.01,
    reset_silence_steps: int = 2,
    confirm_steps: int = 2,
    min_segment_duration: float = 1.2,
    max_segment_duration: float = 24.0,
    tuning_json: str = "",
    tone_frequency_hz: float = 0.0,
    show_full_predictions: bool = False,
    show_meta: bool = False,
    gap_format: str = "relative",
) -> RoutedInferenceResult:
    route = str(route).strip().lower()
    if route not in ROUTE_CHOICES:
        raise ValueError(f"Unsupported route: {route}")

    resolved_path = Path(audio_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved_path}")

    if route == "direct":
        prediction = infer(
            str(resolved_path),
            model_path=model_path,
            model_type=model_type,
            speed=speed,
            device_name=device_name,
        )
        audio, routed_sample_rate = librosa.load(str(resolved_path), sr=sample_rate, mono=True)
        emission = RouteEmission(
            kind="direct",
            stable_text=prediction,
            emitted_text=prediction,
            start_s=0.0,
            end_s=float(len(audio)) / float(routed_sample_rate) if len(audio) else 0.0,
            latency_ms=0.0,
            meta={"model_type": model_type, "speed": speed},
        )
        return RoutedInferenceResult(
            route=route,
            method=model_type,
            audio_path=str(resolved_path),
            sample_rate=int(routed_sample_rate),
            final_text=prediction,
            raw_final_text=prediction,
            emissions=[emission],
        )

    import demo_realtime_mic as realtime_demo

    method = str(route_method).strip().lower() or str(model_type).strip().lower()
    tuning_profile = None
    if tuning_json:
        tuning_profile = realtime_demo.load_tuning_profile(tuning_json, preferred_method=method)
    resolved_model_path = model_path
    if method in {"crnn", "conformer"}:
        resolved_model_path = resolve_model_path_for_inference(model_path, method)

    predictor = realtime_demo.build_predictor(
        method=method,
        sample_rate=sample_rate,
        model_path=resolved_model_path,
        device_name=device_name,
        tone_frequency_hz=tone_frequency_hz,
        tuning_profile=tuning_profile,
    )
    audio, routed_sample_rate = librosa.load(str(resolved_path), sr=sample_rate, mono=True)
    if speed != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=speed)
    result = run_realtime_route_on_audio(
        audio=audio,
        sample_rate=routed_sample_rate,
        predictor=predictor,
        route=route,
        buffer_duration=buffer_duration,
        step_duration=step_duration,
        chunk_duration=chunk_duration,
        silence_threshold=silence_threshold,
        reset_silence_steps=reset_silence_steps,
        confirm_steps=confirm_steps,
        min_segment_duration=min_segment_duration,
        max_segment_duration=max_segment_duration,
        show_full_predictions=show_full_predictions,
        show_meta=show_meta,
        audio_path=str(resolved_path),
    )
    return apply_gap_formatting_to_routed_result(
        result=result,
        audio=audio,
        sample_rate=int(routed_sample_rate),
        gap_format=gap_format,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to wav file")
    parser.add_argument("--model", default="", help="Checkpoint path for deep models. Empty means latest checkpoint.")
    parser.add_argument("--model_type", default="conformer", choices=["crnn", "conformer"])
    parser.add_argument("--speed", type=float, default=1.0, help="Speed up factor (e.g. 1.5, 2.0)")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--route", default="direct", choices=ROUTE_CHOICES)
    parser.add_argument("--route_method", default="", help="Optional realtime route method: energy_threshold, goertzel, crnn, conformer")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--buffer", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=0.75)
    parser.add_argument("--chunk", type=float, default=0.25)
    parser.add_argument("--silence_threshold", type=float, default=0.01)
    parser.add_argument("--reset_silence_steps", type=int, default=2)
    parser.add_argument("--confirm_steps", type=int, default=2)
    parser.add_argument("--min_segment", type=float, default=1.2)
    parser.add_argument("--max_segment", type=float, default=24.0)
    parser.add_argument("--tuning_json", default="", help="Optional tuning_result.json when routing through realtime deep pipeline")
    parser.add_argument("--tone_frequency", type=float, default=0.0, help="Fixed tone frequency for baseline realtime route")
    parser.add_argument("--gap_format", default="relative", choices=["none", "relative"], help="Reinsert word gaps from audio timing for realtime routes")
    parser.add_argument("--show_full_predictions", action="store_true")
    parser.add_argument("--show_meta", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print routed output as JSON")
    args = parser.parse_args()

    result = infer_with_route(
        audio_path=args.audio_path,
        model_path=args.model,
        model_type=args.model_type,
        speed=args.speed,
        device_name=args.device,
        route=args.route,
        route_method=args.route_method,
        sample_rate=args.sample_rate,
        buffer_duration=args.buffer,
        step_duration=args.step,
        chunk_duration=args.chunk,
        silence_threshold=args.silence_threshold,
        reset_silence_steps=args.reset_silence_steps,
        confirm_steps=args.confirm_steps,
        min_segment_duration=args.min_segment,
        max_segment_duration=args.max_segment,
        tuning_json=args.tuning_json,
        tone_frequency_hz=args.tone_frequency,
        gap_format=args.gap_format,
        show_full_predictions=args.show_full_predictions,
        show_meta=args.show_meta,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    elif args.route == "direct":
        print(f"\nPREDICTION (speed x{args.speed}): {result.final_text}")
    else:
        print(f"\nROUTE {result.route} ({result.method})")
        for event in result.emissions:
            print(
                f"[{event.kind}] {event.emitted_text} | stable={event.stable_text} | "
                f"{event.start_s:.2f}s->{event.end_s:.2f}s"
            )
        print(f"\nFINAL: {result.final_text}")
