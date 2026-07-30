from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.real_style import (
    SynthesisConfig,
    build_manifest_rows,
    format_label_text,
    load_unit_bank,
    synthesize_from_unit_bank,
    write_wav,
)


def _random_texts(
    rng: np.random.Generator,
    count: int,
    min_length: int,
    max_length: int,
    charset: str,
) -> list[str]:
    outputs: list[str] = []
    for _ in range(max(0, count)):
        length = int(rng.integers(max(1, min_length), max(min_length, max_length) + 1))
        outputs.append("".join(rng.choice(list(charset), size=length)))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a real-style Morse dataset by concatenating extracted dot/dash/gap units.")
    parser.add_argument("--bank-dir", default="data/real_unit_bank")
    parser.add_argument("--out-dir", default="data/real_style_dataset")
    parser.add_argument("--alphabet-repeats", type=int, default=8)
    parser.add_argument("--digit-repeats", type=int, default=8)
    parser.add_argument("--random-count", type=int, default=200)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--label-format", choices=["char_spaced", "compact"], default="char_spaced")
    parser.add_argument("--jitter-ratio", type=float, default=0.05)
    parser.add_argument("--crossfade-ms", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    bank = load_unit_bank(args.bank_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    wav_dir = out_dir / "wav"

    rng = np.random.default_rng(int(args.seed))
    texts: list[str] = []
    texts.extend(list(string.ascii_uppercase) * max(0, int(args.alphabet_repeats)))
    texts.extend(list(string.digits) * max(0, int(args.digit_repeats)))
    texts.extend(
        _random_texts(
            rng=rng,
            count=int(args.random_count),
            min_length=int(args.min_length),
            max_length=int(args.max_length),
            charset=string.ascii_uppercase + string.digits,
        )
    )

    synth_config = SynthesisConfig(
        jitter_ratio=float(args.jitter_ratio),
        crossfade_ms=float(args.crossfade_ms),
    )

    generated_records: list[dict[str, object]] = []
    for index, text in enumerate(texts):
        audio, sample_rate = synthesize_from_unit_bank(
            text=text,
            bank=bank,
            config=synth_config,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        split = "valid" if rng.random() < float(args.valid_fraction) else "train"
        rel_path = Path("wav") / split / f"real_style_{index:06d}.wav"
        abs_path = out_dir / rel_path
        write_wav(abs_path, audio, sample_rate=sample_rate)
        generated_records.append(
            {
                "path": str(abs_path),
                "text": format_label_text(text, label_format=args.label_format),
                "canonical_text": text,
                "split": split,
                "sample_rate": sample_rate,
                "source": "real_style_bank",
                "bank_dir": str(Path(args.bank_dir).expanduser().resolve()),
            }
        )

    train_records = [record for record in generated_records if record["split"] == "train"]
    valid_records = [record for record in generated_records if record["split"] == "valid"]
    build_manifest_rows(out_dir, generated_records, "all.csv")
    build_manifest_rows(out_dir, train_records, "train.csv")
    build_manifest_rows(out_dir, valid_records, "valid.csv")
    print(f"Saved dataset under: {out_dir}")
    print(f"Train samples: {len(train_records)}")
    print(f"Valid samples: {len(valid_records)}")


if __name__ == "__main__":
    main()
