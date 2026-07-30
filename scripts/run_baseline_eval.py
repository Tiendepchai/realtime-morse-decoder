from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.morse_rules import (
    FAILURE_TIMING_PARSE,
    FAILURE_TONE_DETECTION,
    DurationRules,
)
from baselines.energy_threshold_baseline import EnergyThresholdBaseline, EnergyThresholdBaselineConfig
from baselines.goertzel_baseline import GoertzelBaseline, GoertzelBaselineConfig
from baselines.timing_estimation import DotEstimationConfig
from evaluation.confusion import (
    build_confusion_counter,
    confusion_counter_to_dataframe,
    is_digit_related,
    is_space_related,
    top_confusion_pairs,
)
from evaluation.metrics_extended import aggregate_metrics, enrich_prediction_records, summarize_by_field

LOGGER = logging.getLogger("baseline_eval")


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("reports") / "baseline_eval" / timestamp)


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
    for field in ("sample_id", "id", "utterance_id"):
        value = row.get(field)
        if value is not None and str(value) != "":
            return str(value)

    path_value = row.get("path")
    if path_value is not None and str(path_value) != "":
        return Path(str(path_value)).stem
    return f"sample_{index:05d}"


def build_duration_rules(args: argparse.Namespace) -> DurationRules:
    return DurationRules(
        dot_dash_split_units=args.dot_dash_split_units,
        letter_gap_split_units=args.letter_gap_split_units,
        word_gap_split_units=args.word_gap_split_units,
        max_tone_units=args.max_tone_units,
        max_gap_units=args.max_gap_units,
    )


def build_dot_estimation(args: argparse.Namespace) -> DotEstimationConfig:
    return DotEstimationConfig(
        lower_quantile=args.dot_lower_quantile,
        min_unit_ms=args.dot_min_unit_ms,
        max_unit_ms=args.dot_max_unit_ms,
        search_ratio_low=args.dot_search_ratio_low,
        search_ratio_high=args.dot_search_ratio_high,
        search_steps=args.dot_search_steps,
        tone_weight=args.dot_tone_weight,
        gap_weight=args.dot_gap_weight,
    )


def build_decoders(args: argparse.Namespace) -> dict[str, Any]:
    duration_rules = build_duration_rules(args)
    dot_estimation = build_dot_estimation(args)

    energy = EnergyThresholdBaseline(
        EnergyThresholdBaselineConfig(
            sample_rate=args.sample_rate,
            lowcut=args.energy_lowcut,
            highcut=args.energy_highcut,
            filter_order=args.energy_filter_order,
            frame_length_ms=args.energy_frame_length_ms,
            hop_length_ms=args.energy_hop_length_ms,
            smoothing_window=args.energy_smoothing_window,
            threshold_mode=args.energy_threshold_mode,
            hysteresis_high=args.energy_hysteresis_high,
            hysteresis_low=args.energy_hysteresis_low,
            min_tone_duration_ms=args.energy_min_tone_duration_ms,
            fill_gap_duration_ms=args.energy_fill_gap_duration_ms,
            auto_frequency=not args.disable_auto_frequency,
            frequency_search_low_hz=args.frequency_search_low_hz,
            frequency_search_high_hz=args.frequency_search_high_hz,
            band_margin_hz=args.energy_band_margin_hz,
            target_frequency_hz=args.energy_target_frequency_hz,
            use_manifest_frequency=args.use_manifest_frequency,
            duration_rules=duration_rules,
            dot_estimation=dot_estimation,
        )
    )
    goertzel = GoertzelBaseline(
        GoertzelBaselineConfig(
            sample_rate=args.sample_rate,
            target_frequency_hz=args.goertzel_target_frequency_hz,
            use_manifest_frequency=args.use_manifest_frequency,
            auto_frequency=not args.disable_auto_frequency,
            frequency_search_low_hz=args.frequency_search_low_hz,
            frequency_search_high_hz=args.frequency_search_high_hz,
            frequency_tolerance_hz=args.goertzel_frequency_tolerance_hz,
            neighboring_bins=args.goertzel_neighboring_bins,
            frame_length_ms=args.goertzel_frame_length_ms,
            hop_length_ms=args.goertzel_hop_length_ms,
            smoothing_window=args.goertzel_smoothing_window,
            threshold_mode=args.goertzel_threshold_mode,
            hysteresis_high=args.goertzel_hysteresis_high,
            hysteresis_low=args.goertzel_hysteresis_low,
            min_tone_duration_ms=args.goertzel_min_tone_duration_ms,
            fill_gap_duration_ms=args.goertzel_fill_gap_duration_ms,
            duration_rules=duration_rules,
            dot_estimation=dot_estimation,
        )
    )

    return {
        energy.method_name: energy,
        goertzel.method_name: goertzel,
    }


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


def export_confusions(output_dir: Path, method: str, records: list[dict[str, Any]]) -> None:
    counter = build_confusion_counter(records)
    confusion_df = confusion_counter_to_dataframe(counter)
    confusion_df.to_csv(output_dir / f"{method}_confusion_matrix.csv")

    top_confusion_pairs(counter, top_n=20).to_csv(
        output_dir / f"{method}_top_confusions.csv",
        index=False,
    )
    top_confusion_pairs(counter, top_n=20, predicate=is_space_related).to_csv(
        output_dir / f"{method}_space_confusions.csv",
        index=False,
    )
    top_confusion_pairs(counter, top_n=20, predicate=is_digit_related).to_csv(
        output_dir / f"{method}_digit_confusions.csv",
        index=False,
    )


def export_group_breakdowns(output_dir: Path, method: str, records: list[dict[str, Any]]) -> None:
    for field in ("snr", "noise_type", "wpm"):
        if not records or field not in records[0]:
            continue
        summaries = summarize_by_field(records, field)
        if not summaries:
            continue
        dataframe = pd.DataFrame([flatten_summary(summary) for summary in summaries])
        dataframe.to_csv(output_dir / f"{method}_breakdown_by_{field}.csv", index=False)


def run_decoder(
    decoder: Any,
    manifest: pd.DataFrame,
    audio_dir: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    subset = manifest if limit is None else manifest.head(limit)

    for index, row in subset.iterrows():
        sample_metadata = row.to_dict()
        sample_id = resolve_sample_id(row, index)
        reference = str(row.get("text", ""))

        try:
            audio_path = resolve_audio_path(str(row["path"]), audio_dir=audio_dir)
            result = decoder.decode_file(str(audio_path), sample_metadata=sample_metadata)
        except FileNotFoundError as error:
            LOGGER.error("Missing audio for sample %s: %s", sample_id, error)
            audio_path = Path(str(row.get("path", "")))
            result = {
                "prediction": "",
                "is_failure": True,
                "failure_type": FAILURE_TONE_DETECTION,
                "failure_details": [str(error)],
                "estimated_frequency_hz": None,
                "dot_unit_ms": None,
            }
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("Decoder crashed on sample %s", sample_id)
            audio_path = Path(str(row.get("path", "")))
            result = {
                "prediction": "",
                "is_failure": True,
                "failure_type": FAILURE_TIMING_PARSE,
                "failure_details": [str(error)],
                "estimated_frequency_hz": None,
                "dot_unit_ms": None,
            }

        record = {
            "sample_id": sample_id,
            "audio_path": str(audio_path),
            "reference": reference,
            "prediction": str(result.get("prediction", "")),
            "method": decoder.method_name,
            "is_failure": bool(result.get("is_failure", False)),
            "failure_type": str(result.get("failure_type") or ""),
            "failure_details": " | ".join(result.get("failure_details", [])),
            "estimated_frequency_hz": result.get("estimated_frequency_hz"),
            "dot_unit_ms": result.get("dot_unit_ms"),
            "high_threshold": result.get("high_threshold"),
            "low_threshold": result.get("low_threshold"),
            "num_frames": result.get("num_frames"),
            "num_tone_frames": result.get("num_tone_frames"),
        }

        for column, value in sample_metadata.items():
            if column == "text":
                continue
            record.setdefault(column, value)

        records.append(record)

        if (len(records) % 50) == 0:
            LOGGER.info("%s processed %d samples", decoder.method_name, len(records))

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run traditional Morse decoding baselines on a CSV manifest.")
    parser.add_argument("--dataset_csv", default=default_dataset_csv())
    parser.add_argument("--audio_dir", default="")
    parser.add_argument("--method", choices=["energy_threshold", "goertzel", "all"], default="all")
    parser.add_argument("--output_dir", default=default_output_dir())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--use_manifest_frequency", action="store_true")
    parser.add_argument("--disable_auto_frequency", action="store_true")
    parser.add_argument("--frequency_search_low_hz", type=float, default=400.0)
    parser.add_argument("--frequency_search_high_hz", type=float, default=1200.0)
    parser.add_argument("--log_level", default="INFO")

    parser.add_argument("--dot_dash_split_units", type=float, default=2.0)
    parser.add_argument("--letter_gap_split_units", type=float, default=2.5)
    parser.add_argument("--word_gap_split_units", type=float, default=6.0)
    parser.add_argument("--max_tone_units", type=float, default=8.0)
    parser.add_argument("--max_gap_units", type=float, default=32.0)

    parser.add_argument("--dot_lower_quantile", type=float, default=0.35)
    parser.add_argument("--dot_min_unit_ms", type=float, default=20.0)
    parser.add_argument("--dot_max_unit_ms", type=float, default=400.0)
    parser.add_argument("--dot_search_ratio_low", type=float, default=0.5)
    parser.add_argument("--dot_search_ratio_high", type=float, default=1.75)
    parser.add_argument("--dot_search_steps", type=int, default=41)
    parser.add_argument("--dot_tone_weight", type=float, default=1.0)
    parser.add_argument("--dot_gap_weight", type=float, default=0.75)

    parser.add_argument("--energy_lowcut", type=float, default=400.0)
    parser.add_argument("--energy_highcut", type=float, default=1200.0)
    parser.add_argument("--energy_filter_order", type=int, default=4)
    parser.add_argument("--energy_frame_length_ms", type=float, default=20.0)
    parser.add_argument("--energy_hop_length_ms", type=float, default=10.0)
    parser.add_argument("--energy_smoothing_window", type=int, default=5)
    parser.add_argument("--energy_threshold_mode", default="adaptive_percentile")
    parser.add_argument("--energy_hysteresis_high", type=float, default=0.55)
    parser.add_argument("--energy_hysteresis_low", type=float, default=0.35)
    parser.add_argument("--energy_min_tone_duration_ms", type=float, default=30.0)
    parser.add_argument("--energy_fill_gap_duration_ms", type=float, default=20.0)
    parser.add_argument("--energy_band_margin_hz", type=float, default=140.0)
    parser.add_argument("--energy_target_frequency_hz", type=float, default=None)

    parser.add_argument("--goertzel_target_frequency_hz", type=float, default=None)
    parser.add_argument("--goertzel_frequency_tolerance_hz", type=float, default=80.0)
    parser.add_argument("--goertzel_neighboring_bins", type=int, default=1)
    parser.add_argument("--goertzel_frame_length_ms", type=float, default=20.0)
    parser.add_argument("--goertzel_hop_length_ms", type=float, default=10.0)
    parser.add_argument("--goertzel_smoothing_window", type=int, default=3)
    parser.add_argument("--goertzel_threshold_mode", default="adaptive_percentile")
    parser.add_argument("--goertzel_hysteresis_high", type=float, default=0.55)
    parser.add_argument("--goertzel_hysteresis_low", type=float, default=0.35)
    parser.add_argument("--goertzel_min_tone_duration_ms", type=float, default=30.0)
    parser.add_argument("--goertzel_fill_gap_duration_ms", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    manifest_path = Path(args.dataset_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset csv not found: {manifest_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    decoders = build_decoders(args)
    methods = list(decoders.keys()) if args.method == "all" else [args.method]

    all_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for method in methods:
        LOGGER.info("Running %s on %s", method, manifest_path)
        decoder = decoders[method]
        raw_records = run_decoder(decoder, manifest, audio_dir=args.audio_dir, limit=args.limit)
        enriched_records = enrich_prediction_records(raw_records)
        summary = aggregate_metrics(enriched_records, method=method)
        summary_rows.append(flatten_summary(summary))
        all_records.extend(enriched_records)

        export_confusions(output_dir, method=method, records=enriched_records)
        export_group_breakdowns(output_dir, method=method, records=enriched_records)

    summary_df = pd.DataFrame(summary_rows)
    prediction_df = pd.DataFrame(all_records)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as handle:
        try:
            handle.write(summary_df.to_markdown(index=False))
        except ImportError:
            handle.write(summary_df.to_string(index=False))
    prediction_df.to_csv(output_dir / "prediction_results.csv", index=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_csv": str(manifest_path),
                "audio_dir": args.audio_dir,
                "methods": methods,
                "energy_config": asdict(decoders["energy_threshold"].config),
                "goertzel_config": asdict(decoders["goertzel"].config),
            },
            handle,
            indent=2,
        )

    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    LOGGER.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
