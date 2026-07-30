# Morse Decoder Realtime

A research prototype for decoding Morse code from synthetic audio, WAV files, and live microphone input.

The project compares classical DSP decoders with CTC-based deep-learning models. It is intended for experimentation and benchmarking, not production deployment.

## What is included

- Synthetic Morse audio and dataset generation with configurable WPM, Farnsworth spacing, frequency variation, timing jitter, and noise.
- Two DSP baselines:
  - `energy_threshold`
  - `goertzel`
- Two CTC models:
  - `CRNN`
  - `Conformer`
- WAV-file inference for trained deep models.
- Realtime microphone decoding for all four methods.
- Repeated benchmark runs, confusion analysis, and real-data manifest validation.
- Unit tests for models, DSP utilities, metrics, text handling, and data helpers.

## Status and limitations

- Research/prototype quality. No accuracy or latency guarantee.
- No pretrained checkpoints are included. Train a model or provide a compatible checkpoint before using deep-model inference.
- No generated dataset, checkpoint, experiment output, debug capture, or local recording is included in this source release.
- The deep-model vocabulary is currently uppercase `A-Z`, digits `0-9`, and spaces. The synthetic generator supports additional Morse symbols.
- `sounddevice` is optional and is not listed in `requirements.txt`; the microphone demo also needs a working local audio stack.
- `pytest` is a development dependency and must be installed separately.
- `evaluate.py` generates mock thesis-style metrics by default. Use the benchmark scripts for real dataset/model evaluation.

## Project layout

```text
.
├── baselines/                 DSP decoders and timing rules
├── evaluation/                Metrics, reports, and benchmark helpers
├── scripts/                   Benchmark, debugging, and analysis CLIs
├── src/
│   ├── app/                   Realtime application logic
│   ├── data/                  Morse and synthetic dataset generators
│   ├── features/              Audio preprocessing and log-mel features
│   ├── models/                CRNN, Conformer, and focal CTC loss
│   ├── inference.py           Offline WAV inference
│   └── train.py               Training entry point
├── tests/                     Pytest test suite
├── demo_realtime_mic.py       Standalone microphone demo
├── evaluate.py                Mock thesis-style report generator
├── main.py                    Small sample-audio generator
└── requirements.txt           Runtime dependencies
```

## Installation

Create an isolated environment, then install the runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

For microphone input, install the optional dependency:

```bash
python3 -m pip install sounddevice
```

Some systems also require PortAudio to be installed at the OS level.

For tests:

```bash
python3 -m pip install pytest
```

## Quick start

Generate the default sample WAV:

```bash
python3 main.py
```

Generate a custom sample:

```bash
python3 src/data/morse_generator.py \
  --text "SOS CQ TEST" \
  --out data/samples/sample_start.wav \
  --snr 15 \
  --noise_type white
```

List available microphone devices:

```bash
python3 demo_realtime_mic.py --list-devices
```

## Generate a synthetic dataset

[`src/data/generate_dataset.py`](src/data/generate_dataset.py) writes train/validation WAV files and CSV manifests.

```bash
python3 src/data/generate_dataset.py \
  --out data/dataset \
  --train_size 1000 \
  --val_size 200 \
  --stage 2 \
  --seed 1337
```

Curriculum stages range from `0` (easiest) to `3` (hardest, microphone-oriented).

## Train a model

[`src/train.py`](src/train.py) supports `crnn` and `conformer` models.

Example: Conformer training:

```bash
python3 src/train.py \
  --train_csv data/dataset/train.csv \
  --valid_csv data/dataset/valid.csv \
  --model_type conformer \
  --device auto \
  --epochs 30 \
  --batch_size 16 \
  --lr 2e-4 \
  --save_dir experiments/checkpoints
```

Example: CRNN training:

```bash
python3 src/train.py \
  --train_csv data/dataset/train.csv \
  --valid_csv data/dataset/valid.csv \
  --model_type crnn \
  --device auto \
  --epochs 30 \
  --batch_size 16 \
  --lr 2e-4 \
  --save_dir experiments/checkpoints
```

Supported device names include `auto`, `cpu`, `cuda`, and `mps` where supported by the local PyTorch installation.

Training creates timestamped artifacts under:

```text
experiments/checkpoints/<model_type>/<timestamp>/
```

These artifacts remain local and are ignored by Git.

## Offline inference

[`src/inference.py`](src/inference.py) requires a trained checkpoint:

```bash
python3 src/inference.py data/samples/sample_start.wav \
  --model experiments/checkpoints/conformer/<run_timestamp>/best_model.pth \
  --model_type conformer \
  --device auto
```

The `<run_timestamp>` path is a placeholder. Replace it with a checkpoint generated by your own training run.

## Realtime microphone demo

Use a DSP baseline without a model checkpoint:

```bash
python3 demo_realtime_mic.py \
  --method energy_threshold \
  --tone-frequency 700 \
  --show-meta
```

```bash
python3 demo_realtime_mic.py \
  --method goertzel \
  --tone-frequency 700 \
  --show-meta
```

Use a trained deep model:

```bash
python3 demo_realtime_mic.py \
  --method conformer \
  --device auto \
  --show-meta
```

The demo can discover the latest local checkpoint under `experiments/checkpoints`, or use the checkpoint options exposed by the script.

## Evaluation and benchmarks

Evaluate the two DSP baselines with [`scripts/run_baseline_eval.py`](scripts/run_baseline_eval.py):

```bash
python3 scripts/run_baseline_eval.py \
  --dataset_csv data/dataset/valid.csv \
  --method all
```

Run repeated benchmarks across the available decoders with [`scripts/run_benchmark_suite.py`](scripts/run_benchmark_suite.py):

```bash
python3 scripts/run_benchmark_suite.py \
  --dataset_csv data/dataset/valid.csv \
  --methods all \
  --device auto
```

For a real-recording manifest, use [`scripts/run_real_benchmark.py`](scripts/run_real_benchmark.py):

```bash
python3 scripts/run_real_benchmark.py \
  --manifest_csv path/to/real_manifest.csv \
  --output_dir real_benchmark_output \
  --methods all \
  --device auto
```

The real-data manifest must provide the columns expected by `evaluation/real_data.py`, including audio path, reference text, split, recording metadata, and vocabulary validation fields.

## Mock report generation

[`evaluate.py`](evaluate.py) creates thesis-style tables, figures, and discussion text:

```bash
python3 evaluate.py --model all
```

This entry point uses mock metrics by default. Treat its output as presentation scaffolding, not measured model performance.

## Tests

Install `pytest`, then run:

```bash
python3 -m pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Repository hygiene

Large or machine-specific artifacts stay outside the public source commit:

- `data/dataset_mic_hard/`
- `experiments/`
- `debug_runs/`
- `*.pth`
- root-level temporary WAV recordings
- Python caches, virtual environments, and `.env` files

Keep datasets and checkpoints in local storage, then pass their paths through the documented CLI options.
