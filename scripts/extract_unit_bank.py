from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.real_style import (
    AlignmentConfig,
    DetectionConfig,
    extract_unit_bank_entries,
    load_audio_for_detection,
    load_manifest_rows,
    save_unit_bank,
)


def _is_exact_row(row: dict[str, str]) -> bool:
    return str(row.get("exact_match_found", "")).strip().lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a reusable real-style Morse unit bank from trusted microphone captures.")
    parser.add_argument("--manifest", default="data/real_seed/manifests/all_labeled.csv")
    parser.add_argument("--out-dir", default="data/real_unit_bank")
    parser.add_argument("--allow-weak-labels", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--target-frequency", type=float, default=None)
    parser.add_argument("--search-low", type=float, default=400.0)
    parser.add_argument("--search-high", type=float, default=1200.0)
    parser.add_argument("--tolerance", type=float, default=80.0)
    parser.add_argument("--bandpass-half-width", type=float, default=150.0)
    parser.add_argument("--frame-ms", type=float, default=12.0)
    parser.add_argument("--hop-ms", type=float, default=4.0)
    parser.add_argument("--smoothing-window", type=int, default=1)
    parser.add_argument("--min-tone-ms", type=float, default=10.0)
    parser.add_argument("--fill-gap-ms", type=float, default=5.0)
    parser.add_argument("--hysteresis-high", type=float, default=0.45)
    parser.add_argument("--hysteresis-low", type=float, default=0.25)
    parser.add_argument("--extract-pad-ms", type=float, default=5.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = load_manifest_rows(manifest_path)
    if not args.allow_weak_labels:
        rows = [row for row in rows if _is_exact_row(row)]

    if not rows:
        raise SystemExit("No rows available after filtering. Pass --allow-weak-labels or provide exact-match records.")

    detection_config = DetectionConfig(
        sample_rate=int(args.sample_rate),
        target_frequency_hz=args.target_frequency,
        auto_frequency=args.target_frequency is None,
        frequency_search_low_hz=float(args.search_low),
        frequency_search_high_hz=float(args.search_high),
        frequency_tolerance_hz=float(args.tolerance),
        bandpass_half_width_hz=float(args.bandpass_half_width),
        frame_length_ms=float(args.frame_ms),
        hop_length_ms=float(args.hop_ms),
        smoothing_window=int(args.smoothing_window),
        min_tone_duration_ms=float(args.min_tone_ms),
        fill_gap_duration_ms=float(args.fill_gap_ms),
        hysteresis_high=float(args.hysteresis_high),
        hysteresis_low=float(args.hysteresis_low),
    )
    alignment_config = AlignmentConfig()

    all_entries: list[dict[str, object]] = []
    record_summaries: list[dict[str, object]] = []
    for row in rows:
        path = str(row.get("path", "")).strip()
        text = str(row.get("text", "")).strip()
        if not path or not text:
            continue

        source_id = str(row.get("take", "")).strip() or Path(path).stem
        audio = load_audio_for_detection(path, sample_rate=detection_config.sample_rate)
        entries, summary = extract_unit_bank_entries(
            audio=audio,
            text=text,
            source_id=source_id,
            out_dir=out_dir,
            detection_config=detection_config,
            alignment_config=alignment_config,
            extract_pad_ms=float(args.extract_pad_ms),
        )
        all_entries.extend(entries)
        record_summaries.append(
            {
                **summary,
                "path": path,
                "exact_match_found": _is_exact_row(row),
                "label_confidence": str(row.get("label_confidence", "")),
            }
        )

    payload = save_unit_bank(
        out_dir=out_dir,
        sample_rate=detection_config.sample_rate,
        entries=all_entries,
        records=record_summaries,
        meta={
            "manifest": str(manifest_path),
            "allow_weak_labels": bool(args.allow_weak_labels),
            "extract_pad_ms": float(args.extract_pad_ms),
        },
    )
    print(f"Saved unit bank: {out_dir / 'bank.json'}")
    print(f"Entries: {payload['entry_count']}")
    print(f"Counts by kind: {payload['counts_by_kind']}")


if __name__ == "__main__":
    main()
