from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.energy_threshold_baseline import EnergyThresholdBaseline, EnergyThresholdBaselineConfig
from baselines.goertzel_baseline import GoertzelBaseline, GoertzelBaselineConfig
from baselines.signal_utils import estimate_dominant_frequency, load_audio_mono
from src.inference import infer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_latest_checkpoint(model_type: str) -> str | None:
    candidates = sorted(Path("experiments/checkpoints").glob(f"{model_type}/*/best_model.pth"))
    if not candidates:
        return None
    return str(candidates[-1])


def normalize_prediction_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def resolve_inputs(path_text: str) -> list[Path]:
    path = Path(path_text).expanduser().resolve()
    if path.is_file():
        return [path]

    session_wav = path / "session.wav"
    segment_paths = sorted((path / "segments").glob("*.wav")) if (path / "segments").exists() else []
    resolved = []
    if session_wav.exists():
        resolved.append(session_wav)
    resolved.extend(segment_paths)
    return resolved


def compute_audio_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {
            "duration_s": 0.0,
            "rms": 0.0,
            "peak": 0.0,
        }
    return {
        "duration_s": float(audio.size) / float(sample_rate),
        "rms": float(np.sqrt(np.mean(np.square(audio), dtype=np.float32))),
        "peak": float(np.max(np.abs(audio))),
    }


def plot_waveform_and_spectrogram(audio: np.ndarray, sample_rate: int, out_prefix: Path) -> dict[str, str]:
    time_axis = np.arange(audio.size, dtype=np.float32) / float(sample_rate)

    waveform_path = out_prefix.with_name(out_prefix.name + "_waveform.png")
    plt.figure(figsize=(12, 3))
    plt.plot(time_axis, audio, linewidth=0.8)
    plt.title(f"Waveform: {out_prefix.stem}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(waveform_path, dpi=160)
    plt.close()

    spectrogram_path = out_prefix.with_name(out_prefix.name + "_spectrogram.png")
    n_fft = 1024
    hop_length = 160
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude_db = librosa.amplitude_to_db(np.abs(stft) + 1e-8, ref=np.max)
    plt.figure(figsize=(12, 4))
    plt.imshow(
        magnitude_db,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0.0, float(audio.size) / float(sample_rate), 0.0, sample_rate / 2.0],
    )
    plt.title(f"Spectrogram: {out_prefix.stem}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(spectrogram_path, dpi=160)
    plt.close()

    return {
        "waveform_png": str(waveform_path),
        "spectrogram_png": str(spectrogram_path),
    }


def analyze_file(path: Path, output_dir: Path, device_name: str, fixed_tone_frequency: float) -> dict[str, object]:
    audio, sample_rate = sf.read(path)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)

    stats = compute_audio_stats(audio, sample_rate)
    audio_16k = load_audio_mono(str(path), sample_rate=16000)
    stats["dominant_frequency_hz"] = estimate_dominant_frequency(
        audio_16k,
        sample_rate=16000,
        search_low_hz=300.0,
        search_high_hz=1400.0,
    )

    file_output_dir = output_dir / path.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_waveform_and_spectrogram(audio, sample_rate, file_output_dir / path.stem)

    method_results: dict[str, dict[str, object]] = {}

    for model_type in ("conformer", "crnn"):
        checkpoint = find_latest_checkpoint(model_type)
        if checkpoint is None:
            method_results[model_type] = {"prediction": "", "error": "checkpoint_not_found"}
            continue
        try:
            prediction = infer(
                str(path),
                checkpoint,
                model_type=model_type,
                device_name=device_name,
            )
            method_results[model_type] = {
                "prediction": normalize_prediction_text(prediction),
                "checkpoint": checkpoint,
            }
        except Exception as error:  # pragma: no cover - defensive CLI path
            method_results[model_type] = {"prediction": "", "error": str(error), "checkpoint": checkpoint}

    baseline_audio = load_audio_mono(str(path), sample_rate=16000)
    baselines = {
        "energy_auto": EnergyThresholdBaseline(EnergyThresholdBaselineConfig(sample_rate=16000, auto_frequency=True)),
        "goertzel_auto": GoertzelBaseline(GoertzelBaselineConfig(sample_rate=16000, auto_frequency=True)),
    }
    if fixed_tone_frequency > 0:
        baselines["energy_fixed"] = EnergyThresholdBaseline(
            EnergyThresholdBaselineConfig(
                sample_rate=16000,
                auto_frequency=False,
                target_frequency_hz=fixed_tone_frequency,
            )
        )
        baselines["goertzel_fixed"] = GoertzelBaseline(
            GoertzelBaselineConfig(
                sample_rate=16000,
                auto_frequency=False,
                target_frequency_hz=fixed_tone_frequency,
            )
        )

    for name, decoder in baselines.items():
        try:
            result = decoder.decode_audio(baseline_audio)
            method_results[name] = {
                "prediction": normalize_prediction_text(result.get("prediction", "")),
                "is_failure": bool(result.get("is_failure")),
                "estimated_frequency_hz": result.get("estimated_frequency_hz"),
                "dot_unit_ms": result.get("dot_unit_ms"),
                "num_tone_frames": result.get("num_tone_frames"),
            }
        except Exception as error:  # pragma: no cover - defensive CLI path
            method_results[name] = {"prediction": "", "error": str(error)}

    summary = {
        "path": str(path),
        "stats": stats,
        "plots": plot_paths,
        "methods": method_results,
    }
    summary_path = file_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a recorded microphone WAV/debug directory with all available decoders.")
    parser.add_argument("input", help="Path to a WAV file, or a debug directory containing session.wav and segments/")
    parser.add_argument("--out-dir", default="", help="Directory to write plots and summary JSON files")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--fixed-tone-frequency", type=float, default=700.0, help="Fixed tone frequency used for the optional fixed baseline runs")
    args = parser.parse_args()

    inputs = resolve_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No WAV files found from input: {args.input}")

    input_path = Path(args.input).expanduser().resolve()
    if args.out_dir:
        output_dir = Path(args.out_dir).expanduser().resolve()
    elif input_path.is_dir():
        output_dir = input_path / "analysis"
    else:
        output_dir = input_path.parent / f"{input_path.stem}_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        analyze_file(
            path=wav_path,
            output_dir=output_dir,
            device_name=args.device,
            fixed_tone_frequency=args.fixed_tone_frequency,
        )
        for wav_path in inputs
    ]

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps({"files": summaries}, indent=2), encoding="utf-8")
    print(f"Wrote analysis report: {report_path}")
    for summary in summaries:
        print(f"\nFile: {summary['path']}")
        print(f"Duration: {summary['stats']['duration_s']:.2f}s | RMS: {summary['stats']['rms']:.4f} | Peak: {summary['stats']['peak']:.4f}")
        print(f"Dominant frequency: {summary['stats'].get('dominant_frequency_hz')}")
        for method_name, method_result in summary["methods"].items():
            prediction = method_result.get("prediction", "")
            error = method_result.get("error")
            if error:
                print(f"  - {method_name}: ERROR {error}")
            else:
                print(f"  - {method_name}: {prediction}")


if __name__ == "__main__":
    main()
