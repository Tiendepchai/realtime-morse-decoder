from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import soundfile as sf

from evaluation.confusion import (
    build_confusion_counter,
    is_digit_related,
    is_space_related,
    top_confusion_pairs,
)
from evaluation.metrics_extended import aggregate_metrics
from src.utils.text import CHARS, normalize_text

REAL_TEMPLATE_COLUMNS = [
    "sample_id",
    "audio_path",
    "reference",
    "duration_sec",
    "split",
    "source_type",
    "device_type",
    "device_model",
    "session_id",
    "operator_id",
    "environment_tag",
    "noise_type",
    "delta_f_hz",
    "wpm_est",
    "wpm_bin",
    "farnsworth",
    "qrm_present",
    "label_vocab_ok",
]

WPM_BINS = (
    ("5-10", 5.0, 10.0),
    ("11-20", 11.0, 20.0),
    ("21-30", 21.0, 30.0),
    ("31-35", 31.0, 35.0),
)


@dataclass(frozen=True)
class ChecklistItem:
    section: str
    item: str
    status: str
    details: str


def wpm_to_bin(wpm_value: float | int | None) -> str:
    if wpm_value is None or pd.isna(wpm_value):
        return ""
    value = float(wpm_value)
    for label, low, high in WPM_BINS:
        if low <= value <= high:
            return label
    return ""


def estimate_audio_metadata(audio_path: str | Path) -> dict[str, Any]:
    info = sf.info(str(audio_path))
    return {
        "actual_sample_rate": int(info.samplerate),
        "actual_channels": int(info.channels),
        "actual_duration_sec": float(info.duration),
        "audio_format_ok": bool(info.channels == 1 and info.samplerate == 16000),
    }


def validate_label_vocab(text: str) -> bool:
    return normalize_text(text) == str(text).upper().strip().replace("\t", " ")


def load_real_manifest(manifest_csv: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(manifest_csv)
    missing_columns = [column for column in REAL_TEMPLATE_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Manifest missing required columns: {missing_columns}")

    standardized = dataframe.copy()
    standardized["sample_id"] = standardized["sample_id"].astype(str)
    standardized["audio_path"] = standardized["audio_path"].astype(str)
    standardized["reference"] = standardized["reference"].fillna("").astype(str)
    standardized["text"] = standardized["reference"]
    standardized["path"] = standardized["audio_path"]

    standardized["reference_normalized"] = standardized["reference"].map(normalize_text)
    standardized["label_vocab_ok"] = standardized["reference"].map(validate_label_vocab)

    if "wpm_bin" in standardized.columns:
        standardized["wpm_bin"] = standardized["wpm_bin"].fillna("").astype(str)
    if "wpm_est" in standardized.columns:
        standardized["wpm_bin"] = standardized.apply(
            lambda row: row["wpm_bin"] if str(row["wpm_bin"]).strip() else wpm_to_bin(row.get("wpm_est")),
            axis=1,
        )

    standardized["qrm_present"] = standardized["qrm_present"].fillna("").astype(str).str.lower()
    standardized["farnsworth"] = standardized["farnsworth"].fillna("").astype(str).str.lower()
    standardized["noise_type"] = standardized["noise_type"].fillna("").astype(str)
    standardized["environment_tag"] = standardized["environment_tag"].fillna("").astype(str)
    standardized["duration_sec"] = pd.to_numeric(standardized["duration_sec"], errors="coerce")
    standardized["delta_f_hz"] = pd.to_numeric(standardized["delta_f_hz"], errors="coerce")
    standardized["wpm_est"] = pd.to_numeric(standardized["wpm_est"], errors="coerce")

    duration_values: list[float] = []
    audio_format_ok_values: list[bool] = []
    actual_sr_values: list[int] = []
    actual_ch_values: list[int] = []

    for _, row in standardized.iterrows():
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.exists():
            duration_values.append(float("nan"))
            audio_format_ok_values.append(False)
            actual_sr_values.append(-1)
            actual_ch_values.append(-1)
            continue
        metadata = estimate_audio_metadata(audio_path)
        duration_values.append(
            float(row["duration_sec"]) if pd.notna(row["duration_sec"]) and float(row["duration_sec"]) > 0 else metadata["actual_duration_sec"]
        )
        audio_format_ok_values.append(metadata["audio_format_ok"])
        actual_sr_values.append(metadata["actual_sample_rate"])
        actual_ch_values.append(metadata["actual_channels"])

    standardized["duration_sec"] = duration_values
    standardized["actual_sample_rate"] = actual_sr_values
    standardized["actual_channels"] = actual_ch_values
    standardized["audio_format_ok"] = audio_format_ok_values
    standardized["abs_delta_f_hz"] = standardized["delta_f_hz"].abs()
    standardized["delta_f_20_80"] = standardized["abs_delta_f_hz"].between(20.0, 80.0, inclusive="both")
    standardized["duration"] = standardized["duration_sec"]
    return standardized


def coverage_counts(manifest: pd.DataFrame) -> dict[str, int]:
    return {
        "num_samples": int(len(manifest)),
        "wpm_5_10": int((manifest["wpm_bin"] == "5-10").sum()),
        "wpm_11_20": int((manifest["wpm_bin"] == "11-20").sum()),
        "wpm_21_30": int((manifest["wpm_bin"] == "21-30").sum()),
        "wpm_31_35": int((manifest["wpm_bin"] == "31-35").sum()),
        "continuous_background": int((manifest["environment_tag"] == "continuous_background").sum()),
        "qrm": int(((manifest["qrm_present"] == "yes") | (manifest["noise_type"].str.lower() == "qrm")).sum()),
        "delta_f_20_80": int(manifest["delta_f_20_80"].sum()),
        "consumer_mic": int((manifest["device_type"] == "consumer_mic").sum()),
        "radio_line_in": int((manifest["source_type"] == "line_in").sum()),
    }


def build_real_data_checklist(
    manifest: pd.DataFrame,
    methods: Iterable[str] | None = None,
    device_note: str = "",
) -> list[ChecklistItem]:
    methods = list(methods or [])
    coverage = coverage_counts(manifest)
    checklist: list[ChecklistItem] = []

    def add(section: str, item: str, passed: bool, details: str) -> None:
        checklist.append(ChecklistItem(section=section, item=item, status="PASS" if passed else "FAIL", details=details))

    add("Dataset Readiness", "Manifest contains required template columns", True, ", ".join(REAL_TEMPLATE_COLUMNS))
    add("Dataset Readiness", "Unique sample_id for every row", manifest["sample_id"].is_unique, "sample_id uniqueness check")
    add("Dataset Readiness", "Every sample has a valid audio path", bool(manifest["audio_path"].map(lambda p: Path(str(p)).exists()).all()), "audio_path existence")
    add("Dataset Readiness", "Every sample stays inside thesis label set", bool(manifest["label_vocab_ok"].all()), "label_vocab_ok derived from normalize_text")
    add("Dataset Readiness", "Audio already stored as mono 16kHz", bool(manifest["audio_format_ok"].all()), "validated via soundfile headers")
    add("Dataset Readiness", "Each sample has duration_sec", bool(manifest["duration_sec"].notna().all()), "duration_sec filled from manifest or audio headers")

    for column in ("source_type", "device_type", "device_model", "session_id", "operator_id", "environment_tag", "noise_type", "wpm_bin", "farnsworth"):
        add("Metadata Completeness", f"{column} filled", bool(manifest[column].astype(str).str.strip().ne("").all()), f"non-empty check on column {column}")
    for column in ("delta_f_hz", "wpm_est"):
        add("Metadata Completeness", f"{column} filled", bool(manifest[column].notna().all()), f"not-na check on column {column}")

    add("Coverage", "At least 240 real samples", coverage["num_samples"] >= 240, f"count={coverage['num_samples']}")
    add("Coverage", "40-60 samples in WPM 5-10", coverage["wpm_5_10"] >= 40, f"count={coverage['wpm_5_10']}")
    add("Coverage", "40-60 samples in WPM 11-20", coverage["wpm_11_20"] >= 40, f"count={coverage['wpm_11_20']}")
    add("Coverage", "40-60 samples in WPM 21-30", coverage["wpm_21_30"] >= 40, f"count={coverage['wpm_21_30']}")
    add("Coverage", "40-60 samples in WPM 31-35", coverage["wpm_31_35"] >= 40, f"count={coverage['wpm_31_35']}")
    add("Coverage", "40-60 samples in continuous_background", coverage["continuous_background"] >= 40, f"count={coverage['continuous_background']}")
    add("Coverage", "40-60 samples in qrm", coverage["qrm"] >= 40, f"count={coverage['qrm']}")
    add("Coverage", "40-60 samples in delta_f 20-80", coverage["delta_f_20_80"] >= 40, f"count={coverage['delta_f_20_80']}")
    add(
        "Coverage",
        "Two capture chains represented",
        coverage["consumer_mic"] > 0 and coverage["radio_line_in"] > 0,
        f"consumer_mic={coverage['consumer_mic']} radio_line_in={coverage['radio_line_in']}",
    )

    if methods:
        required = {"energy_threshold", "goertzel", "crnn", "conformer"}
        add("Protocol", "All four methods selected", set(methods) == required, f"methods={methods}")
    else:
        checklist.append(ChecklistItem("Protocol", "All four methods selected", "MANUAL", "Requires benchmark config context"))
    checklist.append(ChecklistItem("Sanity", "No transcript leakage across splits", "MANUAL", "Requires split policy review"))
    checklist.append(ChecklistItem("Sanity", "Manual listening spot-check completed", "MANUAL", "Human verification required"))
    checklist.append(ChecklistItem("Timing", "Device note recorded", "PASS" if bool(device_note) else "FAIL", device_note or "device note missing"))
    return checklist


def dominant_error_note(records: Iterable[Mapping[str, Any]]) -> str:
    records = list(records)
    if not records:
        return "no_data"
    failure_counter = Counter(str(record.get("failure_type") or "") for record in records if record.get("is_failure"))
    if failure_counter.get("tone_detection_failure", 0) > 0:
        return "tone loss under interference"

    counter = build_confusion_counter(records)
    if counter:
        space_pairs = top_confusion_pairs(counter, top_n=20, predicate=is_space_related)
        digit_pairs = top_confusion_pairs(counter, top_n=20, predicate=is_digit_related)
        space_count = int(space_pairs["count"].sum()) if "count" in space_pairs else 0
        digit_count = int(digit_pairs["count"].sum()) if "count" in digit_pairs else 0
    else:
        space_count = 0
        digit_count = 0
    deletion_count = sum(
        count
        for (reference_token, prediction_token), count in counter.items()
        if prediction_token == "<DEL>" and reference_token != "<INS>"
    )
    if deletion_count >= max(space_count, digit_count, 1):
        return "deletion-heavy outputs"
    if space_count >= max(digit_count, 1):
        return "space errors"
    if digit_count > 0:
        return "digit confusions"
    return "mixed substitutions"


def build_condition_subsets(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "WPM 5-10": manifest[manifest["wpm_bin"] == "5-10"].copy(),
        "WPM 11-20": manifest[manifest["wpm_bin"] == "11-20"].copy(),
        "WPM 21-30": manifest[manifest["wpm_bin"] == "21-30"].copy(),
        "WPM 31-35": manifest[manifest["wpm_bin"] == "31-35"].copy(),
        "continuous_background": manifest[manifest["environment_tag"] == "continuous_background"].copy(),
        "qrm": manifest[(manifest["qrm_present"] == "yes") | (manifest["noise_type"].str.lower() == "qrm")].copy(),
        "delta_f_20_80": manifest[manifest["delta_f_20_80"]].copy(),
    }


def export_real_checklist(output_dir: str | Path, checklist: Iterable[ChecklistItem]) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [item.__dict__ for item in checklist]
    dataframe = pd.DataFrame(rows)
    csv_path = output_dir / "real_data_checklist.csv"
    md_path = output_dir / "real_data_checklist.md"
    dataframe.to_csv(csv_path, index=False)

    lines = ["# Real-Data Checklist", ""]
    for section, group in dataframe.groupby("section", sort=False):
        lines.append(f"## {section}")
        lines.append("")
        for _, row in group.iterrows():
            lines.append(f"- [{row['status']}] {row['item']}: {row['details']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def export_real_condition_tables(output_dir: str | Path, records: Iterable[Mapping[str, Any]], seed: int) -> Path:
    output_dir = Path(output_dir)
    dataframe = pd.DataFrame(list(records))
    if dataframe.empty:
        path = output_dir / f"real_condition_table_seed{seed}.csv"
        pd.DataFrame().to_csv(path, index=False)
        return path

    rows: list[dict[str, Any]] = []
    subsets = build_condition_subsets(dataframe)
    for condition_name, subset in subsets.items():
        if subset.empty:
            continue
        for method, method_df in subset.groupby("method", sort=True):
            summary = aggregate_metrics(method_df.to_dict("records"), method=method)
            rows.append(
                {
                    "condition": condition_name,
                    "method": method,
                    "num_samples": int(summary["num_samples"]),
                    "cer": summary["cer"],
                    "wer": summary["wer"],
                    "exact_match_rate": summary["exact_match_rate"],
                    "decode_failure_rate": summary["decode_failure_rate"],
                    "best_wer_percent": summary["wer"] * 100.0,
                    "dominant_error_type": dominant_error_note(method_df.to_dict("records")),
                }
            )

    result = pd.DataFrame(rows).sort_values(["condition", "wer", "cer"], ascending=[True, True, True])
    output_path = output_dir / f"real_condition_table_seed{seed}.csv"
    result.to_csv(output_path, index=False)
    return output_path


def export_real_manifest_summary(output_dir: str | Path, manifest: pd.DataFrame) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    summary_path = output_dir / "real_manifest_summary.json"
    coverage_path = output_dir / "real_manifest_coverage.csv"
    summary_payload = {
        "num_samples": int(len(manifest)),
        "coverage": coverage_counts(manifest),
        "template_columns": REAL_TEMPLATE_COLUMNS,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    coverage_df = pd.DataFrame([coverage_counts(manifest)])
    coverage_df.to_csv(coverage_path, index=False)
    return summary_path, coverage_path
