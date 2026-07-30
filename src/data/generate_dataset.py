import os
import random
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data.morse_generator import MorseGenerator


VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
COMMON_WORDS = [
    "THE", "BE", "TO", "OF", "AND", "A", "IN", "THAT", "HAVE", "I",
    "IT", "FOR", "NOT", "ON", "WITH", "HE", "AS", "YOU", "DO", "AT",
    "CQ", "SOS", "TEST", "DE", "K", "R", "AR", "SK", "QTH", "QSL"
]


def random_text(min_len=5, max_len=20):
    mode = random.choice(['chars', 'words', 'callsigns'])

    if mode == 'chars':
        length = random.randint(min_len, max_len)
        s = ''.join(random.choices(VOCAB + " ", k=length)).strip()
        # avoid all-spaces
        return s if s else "HI"

    if mode == 'words':
        count = random.randint(1, 5)
        return ' '.join(random.choices(COMMON_WORDS, k=count))

    # callsigns
    prefix = random.choice(['K', 'W', 'N', 'A', 'JA', 'G'])
    digit = str(random.randint(0, 9))
    suffix = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1, 3)))
    return f"{prefix}{digit}{suffix}"


def sample_params(stage: int):
    """
    Curriculum stages:
      0: easy (high SNR, no drift/jitter, white noise only)
      1: add small jitter/offset
      2: add drift + wider freq + lower SNR + light room coloration
      3: mic-realistic hard mode with room/QRM/channel distortion
    """
    if stage == 0:
        wpm = random.randint(18, 26)
        farn = random.randint(max(12, wpm - 3), wpm)
        freq = random.randint(600, 850)
        snr = round(random.uniform(16.0, 28.0), 2)
        jitter = 0.0
        drift = 0.0
        offset = 0.0
        noise_type = "white"
        return {
            "wpm": wpm,
            "farnsworth_wpm": farn,
            "frequency": freq,
            "snr_db": snr,
            "timing_jitter": jitter,
            "freq_drift": drift,
            "freq_offset": offset,
            "noise_type": noise_type,
        }

    if stage == 1:
        wpm = random.randint(16, 30)
        farn = random.randint(max(10, wpm - 5), wpm)
        freq = random.randint(500, 950)
        snr = round(random.uniform(10.0, 24.0), 2)
        jitter = random.uniform(0.0, 0.09) if random.random() < 0.6 else 0.0
        drift = 0.0
        offset = random.uniform(-35, 35) if random.random() < 0.35 else 0.0
        noise_type = random.choice(["white", "white", "pink"])
        return {
            "wpm": wpm,
            "farnsworth_wpm": farn,
            "frequency": freq,
            "snr_db": snr,
            "timing_jitter": jitter,
            "freq_drift": drift,
            "freq_offset": offset,
            "noise_type": noise_type,
        }

    if stage == 2:
        wpm = random.randint(15, 30)
        farn = random.randint(max(9, wpm - 7), wpm)
        freq = random.randint(450, 1050)
        snr = round(random.uniform(4.0, 22.0), 2)
        jitter = random.uniform(0.0, 0.13) if random.random() < 0.65 else 0.0
        drift = random.uniform(0.0, 14.0) if random.random() < 0.4 else 0.0
        offset = random.uniform(-70.0, 70.0) if random.random() < 0.4 else 0.0
        noise_type = random.choice(["white", "pink", "hum", "room", "interference"])
        return {
            "wpm": wpm,
            "farnsworth_wpm": farn,
            "frequency": freq,
            "snr_db": snr,
            "timing_jitter": jitter,
            "freq_drift": drift,
            "freq_offset": offset,
            "noise_type": noise_type,
        }

    # stage 3 (hard / microphone-realistic)
    wpm = random.randint(12, 32)
    farn = random.randint(max(8, wpm - 9), wpm)
    freq = random.randint(500, 1150)
    snr = round(random.uniform(-8.0, 18.0), 2)
    jitter = random.uniform(0.02, 0.18) if random.random() < 0.8 else 0.0
    drift = random.uniform(0.0, 28.0) if random.random() < 0.55 else 0.0
    offset = random.uniform(-120.0, 120.0) if random.random() < 0.55 else 0.0
    noise_type = random.choice(["room", "mic", "qrm", "realistic", "realistic_mic", "pink", "interference"])
    return {
        "wpm": wpm,
        "farnsworth_wpm": farn,
        "frequency": freq,
        "snr_db": snr,
        "timing_jitter": jitter,
        "freq_drift": drift,
        "freq_offset": offset,
        "noise_type": noise_type,
    }


def generate_dataset(output_dir, num_samples, set_name="train", curriculum_stage=2, seed=1337):
    random.seed(seed)
    np.random.seed(seed)

    save_dir = os.path.join(output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    metadata = []
    print(f"Generating {num_samples} samples for {set_name} (stage={curriculum_stage}) ...")

    for i in range(num_samples):
        # optional: mix stages within one set
        if set_name == "train":
            # 60% at chosen stage, 40% easier (stage-1) for stability
            stage = curriculum_stage if random.random() < 0.6 else max(0, curriculum_stage - 1)
        else:
            # validation: fixed stage for consistent metric
            stage = curriculum_stage

        params = sample_params(stage)
        wpm = int(params["wpm"])
        farn = int(params["farnsworth_wpm"])
        freq = float(params["frequency"])
        snr = float(params["snr_db"])
        jitter = float(params["timing_jitter"])
        drift = float(params["freq_drift"])
        offset = float(params["freq_offset"])
        noise_type = str(params["noise_type"])

        text = random_text()
        if not text:
            text = "HI"

        gen = MorseGenerator(
            wpm=wpm,
            farnsworth_wpm=farn,
            frequency=freq,
            timing_jitter=jitter,
            freq_drift=drift,
            freq_offset=offset
        )

        audio = gen.generate_audio(text, snr_db=snr, noise_type=noise_type, normalize_peak=True)
        render_meta = dict(getattr(gen, "last_render_metadata", {}) or {})

        filename = f"{set_name}_{i:05d}.wav"
        file_path = os.path.join(save_dir, filename)
        gen.save(audio, file_path)

        duration = len(audio) / float(gen.sample_rate)
        row = {
            "path": os.path.join(set_name, filename),
            "abs_path": os.path.abspath(file_path),
            "text": text,
            "duration": duration,
            "wpm": wpm,
            "farnsworth": farn,
            "freq": freq,
            "snr": snr,
            "jitter": jitter,
            "drift": drift,
            "offset": offset,
            "noise_type": noise_type,
            "stage": stage
        }
        row.update(
            {
                "noise_components": render_meta.get("noise_components", ""),
                "channel_profile": render_meta.get("channel_profile", "clean"),
                "pre_silence_s": render_meta.get("pre_silence_s", 0.0),
                "post_silence_s": render_meta.get("post_silence_s", 0.0),
                "channel_low_hz": render_meta.get("channel_low_hz", 0.0),
                "channel_high_hz": render_meta.get("channel_high_hz", 0.0),
                "reverb_mix": render_meta.get("reverb_mix", 0.0),
                "codec_rate": render_meta.get("codec_rate", 0),
                "qsb_depth": render_meta.get("qsb_depth", 0.0),
                "dropout_count": render_meta.get("dropout_count", 0),
                "click_count": render_meta.get("click_count", 0),
                "companding_drive": render_meta.get("companding_drive", 0.0),
            }
        )
        metadata.append(row)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples}")

    df = pd.DataFrame(metadata)
    csv_path = os.path.join(output_dir, f"{set_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/dataset")
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--val_size", type=int, default=200)
    parser.add_argument("--stage", type=int, default=2, choices=[0, 1, 2, 3], help="Curriculum difficulty stage")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    generate_dataset(args.out, args.train_size, "train", curriculum_stage=args.stage, seed=args.seed)
    generate_dataset(args.out, args.val_size, "valid", curriculum_stage=args.stage, seed=args.seed)
