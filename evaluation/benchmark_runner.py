from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional as F
import torchaudio

from baselines.energy_threshold_baseline import EnergyThresholdBaseline, EnergyThresholdBaselineConfig
from baselines.goertzel_baseline import GoertzelBaseline, GoertzelBaselineConfig
from baselines.morse_rules import FAILURE_TIMING_PARSE, FAILURE_TONE_DETECTION, DurationRules
from baselines.signal_utils import bandpass_filter, load_audio_mono
from baselines.timing_estimation import DotEstimationConfig
from evaluation.confusion import (
    build_confusion_counter,
    confusion_counter_to_dataframe,
    is_digit_related,
    is_space_related,
    top_confusion_pairs,
)
from evaluation.metrics_extended import aggregate_metrics, enrich_prediction_records, summarize_by_field
from src.models.conformer import Conformer, infer_conformer_time_reduction_factor
from src.models.crnn import CRNN
from src.utils.device import resolve_torch_device, synchronize_device
from src.utils.text import CHARS, TextTransform


def default_dataset_csv() -> str:
    candidates = [
        Path("data/dataset_large/valid.csv"),
        Path("data/dataset_large_2/valid.csv"),
        Path("data/dataset/valid.csv"),
        Path("data/tiny_valid.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "data/dataset_large/valid.csv"


def default_output_dir() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(Path("reports") / "benchmark_suite" / timestamp)


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(token.strip()) for token in str(seed_text).split(",") if token.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_audio_path(raw_path: str, audio_dir: str = "") -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path

    if audio_dir:
        candidate = Path(audio_dir) / path.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"audio file not found: {raw_path}")


def resolve_sample_id(row: pd.Series, index: int) -> str:
    for field_name in ("sample_id", "id", "utterance_id"):
        value = row.get(field_name)
        if value is not None and str(value) != "":
            return str(value)

    path_value = row.get("path")
    if path_value is not None and str(path_value) != "":
        return Path(str(path_value)).stem
    return f"sample_{index:05d}"


def find_latest_checkpoint(model_type: str) -> str:
    candidates = sorted(Path("experiments/checkpoints").glob(f"{model_type}/*/best_model.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pth found for model_type={model_type}")
    return str(candidates[-1])


@dataclass(frozen=True)
class BenchmarkDecoderConfig:
    sample_rate: int = 16000
    use_manifest_frequency: bool = False
    frequency_search_low_hz: float = 400.0
    frequency_search_high_hz: float = 1200.0
    energy_lowcut: float = 400.0
    energy_highcut: float = 1200.0
    energy_filter_order: int = 4
    energy_frame_length_ms: float = 20.0
    energy_hop_length_ms: float = 10.0
    energy_smoothing_window: int = 5
    energy_threshold_mode: str = "adaptive_percentile"
    energy_hysteresis_high: float = 0.55
    energy_hysteresis_low: float = 0.35
    energy_min_tone_duration_ms: float = 30.0
    energy_fill_gap_duration_ms: float = 20.0
    energy_band_margin_hz: float = 140.0
    goertzel_frequency_tolerance_hz: float = 80.0
    goertzel_neighboring_bins: int = 1
    goertzel_frame_length_ms: float = 20.0
    goertzel_hop_length_ms: float = 10.0
    goertzel_smoothing_window: int = 3
    goertzel_threshold_mode: str = "adaptive_percentile"
    goertzel_hysteresis_high: float = 0.55
    goertzel_hysteresis_low: float = 0.35
    goertzel_min_tone_duration_ms: float = 30.0
    goertzel_fill_gap_duration_ms: float = 20.0
    dot_dash_split_units: float = 2.0
    letter_gap_split_units: float = 2.5
    word_gap_split_units: float = 6.0
    max_tone_units: float = 8.0
    max_gap_units: float = 32.0
    dot_lower_quantile: float = 0.35
    dot_min_unit_ms: float = 20.0
    dot_max_unit_ms: float = 400.0
    dot_search_ratio_low: float = 0.5
    dot_search_ratio_high: float = 1.75
    dot_search_steps: int = 41
    dot_tone_weight: float = 1.0
    dot_gap_weight: float = 0.75


def build_duration_rules(config: BenchmarkDecoderConfig) -> DurationRules:
    return DurationRules(
        dot_dash_split_units=config.dot_dash_split_units,
        letter_gap_split_units=config.letter_gap_split_units,
        word_gap_split_units=config.word_gap_split_units,
        max_tone_units=config.max_tone_units,
        max_gap_units=config.max_gap_units,
    )


def build_dot_estimation(config: BenchmarkDecoderConfig) -> DotEstimationConfig:
    return DotEstimationConfig(
        lower_quantile=config.dot_lower_quantile,
        min_unit_ms=config.dot_min_unit_ms,
        max_unit_ms=config.dot_max_unit_ms,
        search_ratio_low=config.dot_search_ratio_low,
        search_ratio_high=config.dot_search_ratio_high,
        search_steps=config.dot_search_steps,
        tone_weight=config.dot_tone_weight,
        gap_weight=config.dot_gap_weight,
    )


@dataclass(frozen=True)
class DeepFeatureExtractorConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 64
    low_cut: int = 400
    high_cut: int = 1200
    filter_order: int = 4
    clamp_min: float = -10.0
    clamp_max: float = 10.0


class DeepFeatureExtractor:
    def __init__(self, config: DeepFeatureExtractorConfig):
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )

    @staticmethod
    def soft_clip(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        return np.tanh(audio / threshold) * threshold

    def clean_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio

        audio = self.soft_clip(audio)
        audio = bandpass_filter(
            audio.astype(np.float32),
            sample_rate=self.config.sample_rate,
            lowcut=self.config.low_cut,
            highcut=self.config.high_cut,
            order=self.config.filter_order,
        )
        peak = float(np.max(np.abs(audio)))
        if peak > 0.0:
            audio = audio / peak
        return audio.astype(np.float32, copy=False)

    def load_and_extract(self, audio_path: str) -> tuple[torch.Tensor, float]:
        audio = load_audio_mono(audio_path, sample_rate=self.config.sample_rate)
        duration_s = float(audio.shape[0]) / float(self.config.sample_rate) if audio.size else 0.0
        cleaned = self.clean_audio(audio)
        if cleaned.size == 0:
            features = torch.zeros((1, self.config.n_mels), dtype=torch.float32)
            return features, duration_s

        waveform = torch.tensor(cleaned.copy(), dtype=torch.float32)
        mel = self.mel_transform(waveform).transpose(0, 1)
        mel = torch.clamp(mel, min=1e-10)
        log_mel = 10.0 * torch.log10(mel)
        log_mel = log_mel - torch.max(log_mel)
        mean = log_mel.mean(dim=0)
        std = log_mel.std(dim=0, unbiased=False)
        features = (log_mel - mean) / (std + 1e-9)
        features = torch.clamp(features, self.config.clamp_min, self.config.clamp_max)
        return features, duration_s


@dataclass(frozen=True)
class DeepModelDecoderConfig:
    model_type: str
    checkpoint_path: str
    device: str = "auto"
    feature_config: DeepFeatureExtractorConfig = field(default_factory=DeepFeatureExtractorConfig)


class DeepModelDecoder:
    def __init__(self, config: DeepModelDecoderConfig):
        self.config = config
        self.method_name = config.model_type
        self.device = self._resolve_device(config.device)
        self.feature_extractor = DeepFeatureExtractor(config.feature_config)
        self.text_transform = TextTransform()
        self.model = self._load_model()
        self.model.eval()

    def _resolve_device(self, requested_device: str) -> torch.device:
        return resolve_torch_device(requested_device)

    def _build_model(self) -> torch.nn.Module:
        num_classes = len(CHARS) + 1
        if self.config.model_type == "crnn":
            return CRNN(num_classes=num_classes)
        if self.config.model_type == "conformer":
            return Conformer(
                num_classes=num_classes,
                input_dim=64,
                d_model=256,
                num_layers=8,
                time_reduction_factor=1,
            )
        raise ValueError(f"Unsupported deep model type: {self.config.model_type}")

    def _load_model(self) -> torch.nn.Module:
        checkpoint = torch.load(self.config.checkpoint_path, map_location="cpu")
        state_dict = {key.removeprefix('module.'): value for key, value in checkpoint.items()}
        if self.config.model_type == "conformer":
            model = Conformer(
                num_classes=len(CHARS) + 1,
                input_dim=64,
                d_model=256,
                num_layers=8,
                time_reduction_factor=infer_conformer_time_reduction_factor(state_dict),
            )
        else:
            model = self._build_model()
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def synchronize(self) -> None:
        synchronize_device(self.device)

    def warmup(self, audio_path: str) -> None:
        self.decode_file(audio_path)

    def decode_file(self, audio_path: str, sample_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        del sample_metadata
        features, audio_duration_s = self.feature_extractor.load_and_extract(audio_path)
        if features.shape[0] == 0:
            return {
                "prediction": "",
                "is_failure": True,
                "failure_type": FAILURE_TONE_DETECTION,
                "failure_details": ["empty_audio_after_feature_extraction"],
                "audio_duration_s": audio_duration_s,
            }

        features_tensor = features.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).to(self.device)
        input_lengths = torch.tensor([features_tensor.shape[3]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(features_tensor, input_lengths)
            output = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        self.synchronize()

        argmax = torch.argmax(output, dim=2)[:, 0].tolist()
        decoded_indices: list[int] = []
        previous = 0
        for index in argmax:
            if index != 0 and index != previous:
                decoded_indices.append(index)
            previous = index

        prediction = self.text_transform.int_to_text(decoded_indices)
        return {
            "prediction": prediction,
            "is_failure": False,
            "failure_type": "",
            "failure_details": [],
            "audio_duration_s": audio_duration_s,
            "num_frames": int(features.shape[0]),
        }


def build_all_decoders(
    config: BenchmarkDecoderConfig,
    device: str,
    crnn_checkpoint: str | None,
    conformer_checkpoint: str | None,
) -> dict[str, Any]:
    duration_rules = build_duration_rules(config)
    dot_estimation = build_dot_estimation(config)

    decoders: dict[str, Any] = {
        "energy_threshold": EnergyThresholdBaseline(
            EnergyThresholdBaselineConfig(
                sample_rate=config.sample_rate,
                lowcut=config.energy_lowcut,
                highcut=config.energy_highcut,
                filter_order=config.energy_filter_order,
                frame_length_ms=config.energy_frame_length_ms,
                hop_length_ms=config.energy_hop_length_ms,
                smoothing_window=config.energy_smoothing_window,
                threshold_mode=config.energy_threshold_mode,
                hysteresis_high=config.energy_hysteresis_high,
                hysteresis_low=config.energy_hysteresis_low,
                min_tone_duration_ms=config.energy_min_tone_duration_ms,
                fill_gap_duration_ms=config.energy_fill_gap_duration_ms,
                auto_frequency=not config.use_manifest_frequency,
                frequency_search_low_hz=config.frequency_search_low_hz,
                frequency_search_high_hz=config.frequency_search_high_hz,
                band_margin_hz=config.energy_band_margin_hz,
                target_frequency_hz=None,
                use_manifest_frequency=config.use_manifest_frequency,
                duration_rules=duration_rules,
                dot_estimation=dot_estimation,
            )
        ),
        "goertzel": GoertzelBaseline(
            GoertzelBaselineConfig(
                sample_rate=config.sample_rate,
                target_frequency_hz=None,
                use_manifest_frequency=config.use_manifest_frequency,
                auto_frequency=not config.use_manifest_frequency,
                frequency_search_low_hz=config.frequency_search_low_hz,
                frequency_search_high_hz=config.frequency_search_high_hz,
                frequency_tolerance_hz=config.goertzel_frequency_tolerance_hz,
                neighboring_bins=config.goertzel_neighboring_bins,
                frame_length_ms=config.goertzel_frame_length_ms,
                hop_length_ms=config.goertzel_hop_length_ms,
                smoothing_window=config.goertzel_smoothing_window,
                threshold_mode=config.goertzel_threshold_mode,
                hysteresis_high=config.goertzel_hysteresis_high,
                hysteresis_low=config.goertzel_hysteresis_low,
                min_tone_duration_ms=config.goertzel_min_tone_duration_ms,
                fill_gap_duration_ms=config.goertzel_fill_gap_duration_ms,
                duration_rules=duration_rules,
                dot_estimation=dot_estimation,
            )
        ),
        "crnn": DeepModelDecoder(
            DeepModelDecoderConfig(
                model_type="crnn",
                checkpoint_path=crnn_checkpoint or find_latest_checkpoint("crnn"),
                device=device,
            )
        ),
        "conformer": DeepModelDecoder(
            DeepModelDecoderConfig(
                model_type="conformer",
                checkpoint_path=conformer_checkpoint or find_latest_checkpoint("conformer"),
                device=device,
            )
        ),
    }
    return decoders


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    reserved_keys = {
        "method",
        "num_samples",
        "num_failures",
        "cer",
        "wer",
        "exact_match_rate",
        "decode_failure_rate",
        "failure_breakdown",
    }
    row = {
        **{key: value for key, value in summary.items() if key not in reserved_keys},
        "method": summary["method"],
        "num_samples": summary["num_samples"],
        "num_failures": summary["num_failures"],
        "cer": summary["cer"],
        "wer": summary["wer"],
        "exact_match_rate": summary["exact_match_rate"],
        "decode_failure_rate": summary["decode_failure_rate"],
        "cer_percent": summary["cer"] * 100.0,
        "wer_percent": summary["wer"] * 100.0,
        "exact_match_percent": summary["exact_match_rate"] * 100.0,
        "decode_failure_percent": summary["decode_failure_rate"] * 100.0,
    }
    for failure_type, stats in summary["failure_breakdown"].items():
        row[f"{failure_type}_count"] = stats["count"]
        row[f"{failure_type}_rate"] = stats["rate"]
    return row


def export_confusions(output_dir: Path, method: str, records: list[dict[str, Any]], seed: int) -> None:
    counter = build_confusion_counter(records)
    confusion_counter_to_dataframe(counter).to_csv(output_dir / f"{method}_confusion_matrix_seed{seed}.csv")
    top_confusion_pairs(counter, top_n=20).to_csv(output_dir / f"{method}_top_confusions_seed{seed}.csv", index=False)
    top_confusion_pairs(counter, top_n=20, predicate=is_space_related).to_csv(
        output_dir / f"{method}_space_confusions_seed{seed}.csv",
        index=False,
    )
    top_confusion_pairs(counter, top_n=20, predicate=is_digit_related).to_csv(
        output_dir / f"{method}_digit_confusions_seed{seed}.csv",
        index=False,
    )


def export_group_breakdowns(output_dir: Path, method: str, records: list[dict[str, Any]], seed: int) -> None:
    for field_name in ("snr", "noise_type", "wpm"):
        if not records or field_name not in records[0]:
            continue
        summaries = summarize_by_field(records, field_name)
        if not summaries:
            continue
        dataframe = pd.DataFrame([flatten_summary(summary) for summary in summaries])
        dataframe.to_csv(output_dir / f"{method}_breakdown_by_{field_name}_seed{seed}.csv", index=False)


def select_manifest_for_run(
    manifest: pd.DataFrame,
    seed: int,
    limit: int | None,
    shuffle_each_run: bool,
) -> pd.DataFrame:
    subset = manifest.head(limit).copy() if limit is not None else manifest.copy()
    if shuffle_each_run:
        subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return subset


def _reference_from_row(row: pd.Series) -> str:
    for field_name in ("text", "reference", "label"):
        value = row.get(field_name)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def _checkpoint_for_decoder(decoder: Any) -> str:
    config = getattr(decoder, "config", None)
    checkpoint_path = getattr(config, "checkpoint_path", "")
    return str(checkpoint_path or "")


def _device_for_decoder(decoder: Any) -> str:
    device = getattr(decoder, "device", None)
    return str(device) if device is not None else "cpu"


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "latency_mean_ms": 0.0,
            "latency_std_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p90_ms": 0.0,
        }

    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "latency_mean_ms": float(np.mean(values)),
        "latency_std_ms": float(np.std(values)),
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p90_ms": float(np.percentile(values, 90)),
    }


def run_single_method(
    decoder: Any,
    manifest: pd.DataFrame,
    audio_dir: str,
    seed: int,
    run_index: int,
    limit: int | None,
    shuffle_each_run: bool,
    warmup_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subset = select_manifest_for_run(manifest, seed=seed, limit=limit, shuffle_each_run=shuffle_each_run)
    records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    total_processing_time_sec = 0.0
    total_audio_duration_sec = 0.0
    method_name = str(getattr(decoder, "method_name", type(decoder).__name__.lower()))

    if warmup_samples > 0 and hasattr(decoder, "warmup") and len(subset) > 0:
        for _, row in subset.head(warmup_samples).iterrows():
            try:
                decoder.warmup(str(resolve_audio_path(str(row["path"]), audio_dir=audio_dir)))
            except Exception:
                break

    for row_index, row in subset.iterrows():
        sample_metadata = row.to_dict()
        sample_id = resolve_sample_id(row, row_index)
        reference = _reference_from_row(row)

        try:
            audio_path = resolve_audio_path(str(row["path"]), audio_dir=audio_dir)
            start = time.perf_counter()
            result = decoder.decode_file(str(audio_path), sample_metadata=sample_metadata)
            if hasattr(decoder, "synchronize"):
                decoder.synchronize()
            elapsed = time.perf_counter() - start
        except FileNotFoundError as error:
            audio_path = Path(str(row.get("path", "")))
            elapsed = 0.0
            result = {
                "prediction": "",
                "is_failure": True,
                "failure_type": FAILURE_TONE_DETECTION,
                "failure_details": [str(error)],
                "audio_duration_s": float(row.get("duration", 0.0) or 0.0),
            }
        except Exception as error:
            audio_path = resolve_audio_path(str(row["path"]), audio_dir=audio_dir)
            elapsed = 0.0
            result = {
                "prediction": "",
                "is_failure": True,
                "failure_type": FAILURE_TIMING_PARSE,
                "failure_details": [repr(error)],
                "audio_duration_s": float(row.get("duration", 0.0) or 0.0),
            }

        audio_duration_s = float(result.get("audio_duration_s", row.get("duration", 0.0) or 0.0))
        total_processing_time_sec += elapsed
        total_audio_duration_sec += audio_duration_s
        latency_ms = elapsed * 1000.0
        latencies_ms.append(latency_ms)

        record: dict[str, Any] = {
            "method": method_name,
            "sample_id": sample_id,
            "reference": reference,
            "prediction": str(result.get("prediction", "")),
            "is_failure": bool(result.get("is_failure", False)),
            "failure_type": str(result.get("failure_type") or ""),
            "failure_details": json.dumps(result.get("failure_details", []), ensure_ascii=False),
            "audio_path": str(audio_path),
            "audio_duration_s": audio_duration_s,
            "latency_ms": latency_ms,
            "seed": seed,
            "run_index": run_index,
        }
        for key, value in sample_metadata.items():
            if key not in record:
                record[key] = value
        for key, value in result.items():
            if key not in record:
                record[key] = value
        records.append(record)

    enriched_records = enrich_prediction_records(records)
    summary = flatten_summary(aggregate_metrics(enriched_records, method=method_name))
    summary.update(
        {
            "run_index": run_index,
            "seed": seed,
            "checkpoint_path": _checkpoint_for_decoder(decoder),
            "device": _device_for_decoder(decoder),
            "total_processing_time_sec": float(total_processing_time_sec),
            "total_audio_duration_sec": float(total_audio_duration_sec),
            "rtf_cpu": float(total_processing_time_sec / total_audio_duration_sec) if total_audio_duration_sec > 0 else 0.0,
        }
    )
    summary.update(_latency_summary(latencies_ms))
    return enriched_records, summary


def aggregate_repeated_runs(run_summaries: Iterable[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(list(run_summaries))
    if dataframe.empty:
        return dataframe

    numeric_columns = [
        column
        for column in dataframe.columns
        if column not in {"method", "run_index", "seed", "checkpoint_path", "device"}
        and pd.api.types.is_numeric_dtype(dataframe[column])
    ]

    rows: list[dict[str, Any]] = []
    for method, group in dataframe.groupby("method", sort=True):
        row: dict[str, Any] = {
            "method": method,
            "num_runs": int(len(group)),
            "seed_list": ",".join(str(int(seed_value)) for seed_value in group["seed"].tolist()),
            "checkpoint_path": str(group["checkpoint_path"].iloc[0]) if "checkpoint_path" in group else "",
            "device": str(group["device"].iloc[0]) if "device" in group else "",
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        rows.append(row)

    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No data_"
    try:
        return dataframe.to_markdown(index=False)
    except Exception:
        return "```\n" + dataframe.to_string(index=False) + "\n```"


def write_benchmark_report(
    output_dir: str | Path,
    aggregate_df: pd.DataFrame,
    run_df: pd.DataFrame,
    seeds: list[int],
    dataset_csv: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "benchmark_report.md"

    lines = [
        "# Benchmark Report",
        "",
        f"- Dataset: `{dataset_csv}`",
        f"- Seeds: `{','.join(str(seed) for seed in seeds)}`",
        f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Aggregate Summary",
        "",
        _dataframe_to_markdown(aggregate_df),
        "",
        "## Run Level Summary",
        "",
        _dataframe_to_markdown(run_df),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_config(
    output_dir: str | Path,
    dataset_csv: str,
    audio_dir: str,
    config: BenchmarkDecoderConfig,
    seeds: list[int],
    methods: list[str],
    crnn_checkpoint: str,
    conformer_checkpoint: str,
    device: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    payload = {
        "dataset_csv": dataset_csv,
        "audio_dir": audio_dir,
        "config": asdict(config),
        "seeds": seeds,
        "methods": methods,
        "crnn_checkpoint": crnn_checkpoint,
        "conformer_checkpoint": conformer_checkpoint,
        "device": device,
    }
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path
