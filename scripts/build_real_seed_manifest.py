from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


@dataclass
class RealSeedRecord:
    take: str
    path: str
    text: str
    split: str
    label_source: str
    label_confidence: str
    exact_match_found: bool
    resolution_strategy: str
    source_audio: str
    reference_wav: str
    duration_sec: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return audio.reshape(-1)


def export_audio_clip(
    source_path: Path,
    out_path: Path,
    start_s: float | None = None,
    end_s: float | None = None,
) -> float:
    audio, sample_rate = sf.read(source_path)
    audio = ensure_mono_float32(audio)

    start_idx = 0 if start_s is None else max(0, int(round(float(start_s) * sample_rate)))
    end_idx = len(audio) if end_s is None else min(len(audio), int(round(float(end_s) * sample_rate)))
    if end_idx <= start_idx:
        raise ValueError(f"Invalid clip window for {source_path}: start={start_s} end={end_s}")

    clip = audio[start_idx:end_idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, clip, sample_rate)
    return float(len(clip)) / float(sample_rate)


def resolve_label(
    take_dir: Path,
    tuning_payload: dict[str, Any] | None,
    allow_known_reference_label: bool,
    default_text: str,
) -> tuple[str | None, str, str, bool, str]:
    if tuning_payload is not None:
        resolved_text = normalize_text(tuning_payload.get("resolved_text", ""))
        exact_match_found = bool(tuning_payload.get("exact_match_found"))
        resolution_strategy = str(tuning_payload.get("resolution_strategy", "")).strip()
        if exact_match_found and resolved_text:
            return resolved_text, "tuning_result", "exact", True, resolution_strategy
        if allow_known_reference_label and resolution_strategy == "known_reference_label" and resolved_text:
            return resolved_text, "tuning_result", "known_reference", False, resolution_strategy

    normalized_default = normalize_text(default_text)
    if normalized_default:
        return normalized_default, "default_text", "manual_override", False, "default_text"

    return None, "", "", False, ""


def resolve_clip_source(
    take_dir: Path,
    session_payload: dict[str, Any] | None,
    tuning_payload: dict[str, Any] | None,
    clips_dir: Path,
) -> tuple[Path, Path, float]:
    take_name = take_dir.name
    best = tuning_payload.get("best", {}) if tuning_payload is not None else {}

    derived_clip_text = str(best.get("derived_clip_wav", "")).strip() if best else ""
    if derived_clip_text:
        derived_clip_wav = Path(derived_clip_text).expanduser()
    else:
        derived_clip_wav = None
    if derived_clip_wav is not None and derived_clip_wav.exists():
        out_path = clips_dir / f"{take_name}.wav"
        duration = export_audio_clip(derived_clip_wav, out_path)
        return out_path, derived_clip_wav, duration

    source_clip_start_s = best.get("source_clip_start_s")
    source_clip_end_s = best.get("source_clip_end_s")
    if source_clip_start_s is not None and source_clip_end_s is not None:
        source_audio = Path(str(take_dir / "session.wav")).resolve()
        if source_audio.exists():
            out_path = clips_dir / f"{take_name}.wav"
            duration = export_audio_clip(
                source_audio,
                out_path,
                start_s=float(source_clip_start_s),
                end_s=float(source_clip_end_s),
            )
            return out_path, source_audio, duration

    if tuning_payload is not None:
        recording_path = Path(str(tuning_payload.get("recording", "")).strip()).expanduser()
        crop_start_s = tuning_payload.get("crop_start_s")
        crop_end_s = tuning_payload.get("crop_end_s")
        if recording_path.exists() and crop_start_s is not None and crop_end_s is not None:
            start_s = max(0.0, float(crop_start_s) - float(best.get("pre_pad_s", 0.0) or 0.0))
            end_s = float(crop_end_s) + float(best.get("post_pad_s", 0.0) or 0.0)
            out_path = clips_dir / f"{take_name}.wav"
            duration = export_audio_clip(recording_path, out_path, start_s=start_s, end_s=end_s)
            return out_path, recording_path, duration

    if session_payload is not None:
        segments = session_payload.get("segments", [])
        if segments:
            segment_path = Path(str(segments[0].get("path", "")).strip()).expanduser()
            if segment_path.exists():
                out_path = clips_dir / f"{take_name}.wav"
                duration = export_audio_clip(segment_path, out_path)
                return out_path, segment_path, duration

        session_wav = Path(str(session_payload.get("session_wav", "")).strip()).expanduser()
        if session_wav.exists():
            out_path = clips_dir / f"{take_name}.wav"
            duration = export_audio_clip(session_wav, out_path)
            return out_path, session_wav, duration

    raise FileNotFoundError(f"Could not resolve an audio clip source for {take_dir}")


def write_manifest(path: Path, records: list[RealSeedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(RealSeedRecord.__annotations__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative real-domain seed manifest from debug_runs microphone captures.")
    parser.add_argument("--debug-runs-root", default="debug_runs")
    parser.add_argument("--out-dir", default="data/real_seed")
    parser.add_argument("--allow-known-reference-label", action="store_true", help="Include takes labeled via tuning_result resolution_strategy=known_reference_label")
    parser.add_argument("--default-text", default="", help="Optional manual fallback label for unlabeled takes")
    args = parser.parse_args()

    debug_root = Path(args.debug_runs_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    clips_dir = out_dir / "clips"
    manifests_dir = out_dir / "manifests"

    labeled_records: list[RealSeedRecord] = []
    exact_records: list[RealSeedRecord] = []
    weak_records: list[RealSeedRecord] = []
    skipped: list[dict[str, str]] = []

    for take_dir in sorted(path for path in debug_root.glob("phone_take_*") if path.is_dir()):
        session_path = take_dir / "session.json"
        tuning_path = take_dir / "tuning_result.json"
        session_payload = read_json(session_path) if session_path.exists() else None
        tuning_payload = read_json(tuning_path) if tuning_path.exists() else None

        label_text, label_source, label_confidence, exact_match_found, resolution_strategy = resolve_label(
            take_dir=take_dir,
            tuning_payload=tuning_payload,
            allow_known_reference_label=bool(args.allow_known_reference_label),
            default_text=args.default_text,
        )
        if not label_text:
            skipped.append({"take": take_dir.name, "reason": "no_trusted_label"})
            continue

        try:
            clip_path, source_audio, duration_sec = resolve_clip_source(
                take_dir=take_dir,
                session_payload=session_payload,
                tuning_payload=tuning_payload,
                clips_dir=clips_dir,
            )
        except Exception as error:
            skipped.append({"take": take_dir.name, "reason": f"clip_resolution_failed: {error}"})
            continue

        reference_wav = ""
        if tuning_payload is not None:
            reference_wav = str(tuning_payload.get("reference_wav", "")).strip()

        record = RealSeedRecord(
            take=take_dir.name,
            path=str(clip_path),
            text=label_text,
            split="",
            label_source=label_source,
            label_confidence=label_confidence,
            exact_match_found=exact_match_found,
            resolution_strategy=resolution_strategy,
            source_audio=str(source_audio),
            reference_wav=reference_wav,
            duration_sec=duration_sec,
        )
        labeled_records.append(record)
        if exact_match_found:
            exact_records.append(record)
        else:
            weak_records.append(record)

    exact_records.sort(key=lambda row: row.take)
    weak_records.sort(key=lambda row: row.take)

    valid_records: list[RealSeedRecord] = []
    train_records: list[RealSeedRecord] = []
    if exact_records:
        valid_records = [RealSeedRecord(**{**asdict(exact_records[-1]), "split": "valid"})]
        train_source = weak_records + exact_records[:-1]
    else:
        train_source = weak_records

    train_records = [RealSeedRecord(**{**asdict(record), "split": "train"}) for record in train_source]
    all_labeled_records = [RealSeedRecord(**{**asdict(record), "split": "train"}) for record in weak_records + exact_records]

    overlap_takes = sorted(set(record.take for record in train_records) & set(record.take for record in valid_records))
    summary = {
        "debug_runs_root": str(debug_root),
        "out_dir": str(out_dir),
        "labeled_count": len(labeled_records),
        "exact_count": len(exact_records),
        "weak_count": len(weak_records),
        "train_count": len(train_records),
        "valid_count": len(valid_records),
        "all_labeled_count": len(all_labeled_records),
        "allow_known_reference_label": bool(args.allow_known_reference_label),
        "default_text": normalize_text(args.default_text),
        "skipped": skipped,
        "overlap_takes": overlap_takes,
    }

    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifests_dir / "all_labeled.csv", all_labeled_records)
    write_manifest(manifests_dir / "train.csv", train_records)
    write_manifest(manifests_dir / "valid.csv", valid_records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote manifests to: {manifests_dir}")


if __name__ == "__main__":
    main()
