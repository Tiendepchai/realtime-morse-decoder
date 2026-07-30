from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark_runner import (
    BenchmarkDecoderConfig,
    aggregate_repeated_runs,
    build_all_decoders,
    default_dataset_csv,
    default_output_dir,
    export_confusions,
    export_group_breakdowns,
    find_latest_checkpoint,
    parse_seed_list,
    run_single_method,
    set_global_seed,
    write_benchmark_report,
    write_config,
)
from evaluation.benchmark_visuals import plot_benchmark_dashboard, plot_run_scatter
from src.utils.device import DEVICE_CHOICES

LOGGER = logging.getLogger("benchmark_suite")


def parse_methods(method_text: str) -> list[str]:
    if method_text == "all":
        return ["energy_threshold", "goertzel", "crnn", "conformer"]
    methods = [token.strip() for token in method_text.split(",") if token.strip()]
    valid = {"energy_threshold", "goertzel", "crnn", "conformer"}
    invalid = [method for method in methods if method not in valid]
    if invalid:
        raise ValueError(f"Unsupported methods requested: {invalid}")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated-run benchmark suite for Morse decoders.")
    parser.add_argument("--dataset_csv", default=default_dataset_csv())
    parser.add_argument("--audio_dir", default="")
    parser.add_argument("--output_dir", default=default_output_dir())
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seeds", default="13,42,123")
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle_each_run", action="store_true", default=True)
    parser.add_argument("--no_shuffle_each_run", action="store_false", dest="shuffle_each_run")
    parser.add_argument("--warmup_samples", type=int, default=1)
    parser.add_argument("--use_manifest_frequency", action="store_true")
    parser.add_argument("--crnn_checkpoint", default="")
    parser.add_argument("--conformer_checkpoint", default="")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dataset_path = Path(args.dataset_csv)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_methods(args.methods)
    seeds = parse_seed_list(args.seeds)
    config = BenchmarkDecoderConfig(use_manifest_frequency=args.use_manifest_frequency)
    crnn_checkpoint = args.crnn_checkpoint or find_latest_checkpoint("crnn")
    conformer_checkpoint = args.conformer_checkpoint or find_latest_checkpoint("conformer")
    manifest = pd.read_csv(dataset_path)

    all_decoders = build_all_decoders(
        config=config,
        device=args.device,
        crnn_checkpoint=crnn_checkpoint,
        conformer_checkpoint=conformer_checkpoint,
    )
    decoders = {method: all_decoders[method] for method in methods}

    run_summaries: list[dict[str, object]] = []
    all_records: list[dict[str, object]] = []
    first_seed_records: dict[str, list[dict[str, object]]] = {}

    for run_index, seed in enumerate(seeds):
        set_global_seed(seed)
        LOGGER.info("Starting run %d with seed=%d", run_index, seed)
        for method in methods:
            decoder = decoders[method]
            LOGGER.info("Benchmarking %s on run %d (seed=%d)", method, run_index, seed)
            records, summary = run_single_method(
                decoder=decoder,
                manifest=manifest,
                audio_dir=args.audio_dir,
                seed=seed,
                run_index=run_index,
                limit=args.limit,
                shuffle_each_run=args.shuffle_each_run,
                warmup_samples=args.warmup_samples,
            )
            run_summaries.append(summary)
            all_records.extend(records)
            if seed == seeds[0]:
                first_seed_records[method] = records

    run_df = pd.DataFrame(run_summaries).sort_values(["method", "run_index"]).reset_index(drop=True)
    aggregate_df = aggregate_repeated_runs(run_summaries).sort_values("cer_percent_mean").reset_index(drop=True)
    prediction_df = pd.DataFrame(all_records).sort_values(["method", "run_index", "sample_id"]).reset_index(drop=True)

    run_df.to_csv(output_dir / "run_level_summary.csv", index=False)
    aggregate_df.to_csv(output_dir / "aggregate_summary.csv", index=False)
    prediction_df.to_csv(output_dir / "prediction_results.csv", index=False)

    write_benchmark_report(output_dir, aggregate_df, run_df, seeds=seeds, dataset_csv=str(dataset_path))
    write_config(
        output_dir,
        dataset_csv=str(dataset_path),
        audio_dir=args.audio_dir,
        config=config,
        seeds=seeds,
        methods=methods,
        crnn_checkpoint=crnn_checkpoint,
        conformer_checkpoint=conformer_checkpoint,
        device=args.device,
    )

    for method, records in first_seed_records.items():
        export_confusions(output_dir, method=method, records=records, seed=seeds[0])
        export_group_breakdowns(output_dir, method=method, records=records, seed=seeds[0])

    plot_benchmark_dashboard(aggregate_df, output_dir)
    plot_run_scatter(run_df, output_dir)

    print("\n=== Aggregate Summary (Mean ± Std over runs) ===")
    print(aggregate_df.to_string(index=False))
    LOGGER.info("Benchmark artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
