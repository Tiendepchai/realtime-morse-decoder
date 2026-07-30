from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F
from editdistance import eval as edit_distance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.audio_processor import AudioProcessor
from src.inference import _load_model_for_inference, infer
from src.utils.device import resolve_torch_device, synchronize_device
from src.utils.text import CHARS, TextTransform


def normalize_prediction_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def infer_reference_target(reference_wav: str, device_name: str) -> str:
    candidates = [
        ("conformer", "experiments/checkpoints/conformer/20260318_104843/best_model.pth"),
        ("crnn", "experiments/checkpoints/crnn/20260318_103856/best_model.pth"),
    ]
    predictions = []
    for model_type, checkpoint in candidates:
        prediction = infer(reference_wav, checkpoint, model_type=model_type, device_name=device_name)
        predictions.append(normalize_prediction_text(prediction))

    predictions = [text for text in predictions if text]
    if not predictions:
        raise RuntimeError(f"Could not infer a target text from reference WAV: {reference_wav}")
    if len(set(predictions)) == 1:
        return predictions[0]
    return max(predictions, key=len)


def load_mono_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return audio, int(sample_rate)


def auto_crop_active_region(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int, int]:
    nyquist = sample_rate / 2.0
    b, a = scipy.signal.butter(4, [700.0 / nyquist, 1200.0 / nyquist], btype="band")
    filtered = scipy.signal.filtfilt(b, a, audio)

    frame_size = int(round(0.05 * sample_rate))
    hop_size = int(round(0.02 * sample_rate))
    energies = []
    starts = []
    for start in range(0, max(1, len(filtered) - frame_size + 1), hop_size):
        chunk = filtered[start : start + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        energies.append(float(np.sqrt(np.mean(chunk**2))))
        starts.append(start)

    energies_array = np.asarray(energies, dtype=np.float32)
    threshold = max(float(np.percentile(energies_array, 70)) * 1.2, float(np.mean(energies_array)) * 1.5)
    active_indices = np.where(energies_array >= threshold)[0]
    if active_indices.size == 0:
        return audio.copy(), 0, len(audio)

    start_index = starts[max(0, int(active_indices[0]) - 5)]
    end_index = min(len(starts) - 1, int(active_indices[-1]) + 5)
    stop = min(len(audio), starts[end_index] + frame_size)
    return audio[start_index:stop].copy(), int(start_index), int(stop)


def build_models(device_name: str, selected_models: list[str] | None = None) -> tuple[torch.device, TextTransform, dict[str, torch.nn.Module]]:
    device = resolve_torch_device(device_name)
    text_transform = TextTransform()
    vocab_size = len(CHARS)
    checkpoints = {
        "conformer": "experiments/checkpoints/conformer/20260318_104843/best_model.pth",
        "crnn": "experiments/checkpoints/crnn/20260318_103856/best_model.pth",
    }
    if selected_models:
        checkpoints = {model_type: checkpoints[model_type] for model_type in selected_models}
    models: dict[str, torch.nn.Module] = {}
    for model_type, checkpoint in checkpoints.items():
        model = _load_model_for_inference(checkpoint, model_type, vocab_size, device)
        model.eval()
        models[model_type] = model
    return device, text_transform, models


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    speed: float,
    band_low: float,
    band_high: float,
    clip_threshold: float,
    companding_exponent: float,
) -> np.ndarray:
    if speed == 1.0:
        processed = audio.astype(np.float32, copy=True)
    else:
        new_length = max(1, int(round(len(audio) / speed)))
        processed = scipy.signal.resample(audio, new_length).astype(np.float32)

    nyquist = sample_rate / 2.0
    b, a = scipy.signal.butter(4, [band_low / nyquist, band_high / nyquist], btype="band")
    processed = scipy.signal.filtfilt(b, a, processed).astype(np.float32)

    processed = np.tanh(processed / clip_threshold) * clip_threshold

    if companding_exponent != 1.0:
        processed = np.sign(processed) * np.power(np.abs(processed), companding_exponent)

    peak = float(np.max(np.abs(processed)))
    if peak > 0.0:
        processed = processed / peak
    return processed.astype(np.float32)


def decode_with_model(
    model: torch.nn.Module,
    device: torch.device,
    text_transform: TextTransform,
    audio: np.ndarray,
    sample_rate: int,
    band_low: float,
    band_high: float,
) -> str:
    processor = AudioProcessor(sample_rate=sample_rate, n_mels=64, low_cut=band_low, high_cut=band_high)
    cleaned = processor.clean_audio(audio.astype(np.float32))
    log_mel = processor.compute_log_mel(cleaned)
    features = processor.apply_cmvn(log_mel)
    features = np.clip(features, -10.0, 10.0)

    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    features_tensor = features_tensor.permute(0, 1, 3, 2).to(device)
    with torch.no_grad():
        input_lengths = torch.tensor([features_tensor.shape[3]], dtype=torch.long, device=device)
        logits = model(features_tensor, input_lengths)
        output = F.log_softmax(logits, dim=2).permute(1, 0, 2)
    synchronize_device(device)

    arg_maxes = torch.argmax(output, dim=2)[:, 0]
    decode = []
    prev = 0
    for idx in arg_maxes:
        token = int(idx.item())
        if token != 0 and token != prev:
            decode.append(token)
        prev = token
    return normalize_prediction_text(text_transform.int_to_text(decode))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search preprocessing parameters that make a recording match a reference target transcript.")
    parser.add_argument("recording", help="Recorded microphone WAV to tune against")
    parser.add_argument("--reference-wav", default="alovu.wav", help="Reference WAV whose transcript is treated as the target")
    parser.add_argument("--target-text", default="", help="Optional explicit target text. If empty, inferred from --reference-wav")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--out-json", default="", help="Optional path to write the best configuration/result as JSON")
    parser.add_argument("--models", nargs="+", default=["conformer", "crnn"], choices=("conformer", "crnn"))
    args = parser.parse_args()

    target_text = normalize_prediction_text(args.target_text) if args.target_text else infer_reference_target(args.reference_wav, args.device)
    recording_audio, sample_rate = load_mono_audio(args.recording)
    cropped_audio, crop_start, crop_end = auto_crop_active_region(recording_audio, sample_rate)

    device, text_transform, models = build_models(args.device, selected_models=args.models)

    print(f"Target text: {target_text}")
    print(f"Auto crop: start={crop_start / sample_rate:.2f}s end={crop_end / sample_rate:.2f}s duration={len(cropped_audio) / sample_rate:.2f}s")

    search_space = {
        "pre_pad_s": [0.0, 0.25],
        "post_pad_s": [0.0, 0.25],
        "speed": [0.9, 0.95, 1.0, 1.05],
        "band": [(400.0, 1200.0), (700.0, 1200.0), (850.0, 1150.0), (900.0, 1100.0)],
        "clip_threshold": [0.75, 0.85, 0.95],
        "companding_exponent": [1.0, 0.8],
    }

    results: list[dict[str, object]] = []
    exact_match: dict[str, object] | None = None
    total_trials = (
        len(models)
        * len(search_space["pre_pad_s"])
        * len(search_space["post_pad_s"])
        * len(search_space["speed"])
        * len(search_space["band"])
        * len(search_space["clip_threshold"])
        * len(search_space["companding_exponent"])
    )
    completed_trials = 0
    for model_type, model in models.items():
        for pre_pad_s in search_space["pre_pad_s"]:
            for post_pad_s in search_space["post_pad_s"]:
                start = max(0, crop_start - int(round(pre_pad_s * sample_rate)))
                stop = min(len(recording_audio), crop_end + int(round(post_pad_s * sample_rate)))
                base_audio = recording_audio[start:stop].copy()

                for speed in search_space["speed"]:
                    for band_low, band_high in search_space["band"]:
                        for clip_threshold in search_space["clip_threshold"]:
                            for companding_exponent in search_space["companding_exponent"]:
                                tuned_audio = preprocess_audio(
                                    audio=base_audio,
                                    sample_rate=sample_rate,
                                    speed=speed,
                                    band_low=band_low,
                                    band_high=band_high,
                                    clip_threshold=clip_threshold,
                                    companding_exponent=companding_exponent,
                                )
                                prediction = decode_with_model(
                                    model=model,
                                    device=device,
                                    text_transform=text_transform,
                                    audio=tuned_audio,
                                    sample_rate=16000,
                                    band_low=band_low,
                                    band_high=band_high,
                                )
                                distance = int(edit_distance(prediction, target_text))
                                results.append(
                                    {
                                        "distance": distance,
                                        "model_type": model_type,
                                        "pre_pad_s": pre_pad_s,
                                        "post_pad_s": post_pad_s,
                                        "speed": speed,
                                        "band_low_hz": band_low,
                                        "band_high_hz": band_high,
                                        "clip_threshold": clip_threshold,
                                        "companding_exponent": companding_exponent,
                                        "prediction": prediction,
                                    }
                                )
                                completed_trials += 1
                                if completed_trials % 100 == 0:
                                    current_best = min(results, key=lambda row: int(row["distance"]))
                                    print(
                                        f"Progress {completed_trials}/{total_trials} | "
                                        f"best_distance={current_best['distance']} | "
                                        f"best_prediction={current_best['prediction']}"
                                    )
                                if distance == 0:
                                    exact_match = results[-1]
                                    break
                            if exact_match is not None:
                                break
                        if exact_match is not None:
                            break
                    if exact_match is not None:
                        break
                if exact_match is not None:
                    break
            if exact_match is not None:
                break
        if exact_match is not None:
            break

    results.sort(key=lambda row: (int(row["distance"]), str(row["model_type"]), len(str(row["prediction"]))))
    best = results[0]
    best_by_model = {
        model_type: min(
            (row for row in results if str(row["model_type"]) == model_type),
            key=lambda row: (int(row["distance"]), len(str(row["prediction"]))),
        )
        for model_type in models
    }
    exact_match_found = bool(int(best["distance"]) == 0)
    resolution_strategy = "model_search" if exact_match_found else "known_reference_label"
    resolved_text = str(best["prediction"]) if exact_match_found else target_text
    print("\nBest result:")
    print(json.dumps(best, indent=2))
    print("\nBest by model:")
    print(json.dumps(best_by_model, indent=2))
    print(f"\nResolved text ({resolution_strategy}): {resolved_text}")
    print("\nTop 20:")
    for row in results[:20]:
        print(json.dumps(row, ensure_ascii=True))

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_text": target_text,
            "recording": str(Path(args.recording).expanduser().resolve()),
            "reference_wav": str(Path(args.reference_wav).expanduser().resolve()),
            "crop_start_s": crop_start / sample_rate,
            "crop_end_s": crop_end / sample_rate,
            "exact_match_found": exact_match_found,
            "resolution_strategy": resolution_strategy,
            "resolved_text": resolved_text,
            "best": best,
            "best_by_model": best_by_model,
            "top20": results[:20],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote tuning report: {out_path}")


if __name__ == "__main__":
    main()
