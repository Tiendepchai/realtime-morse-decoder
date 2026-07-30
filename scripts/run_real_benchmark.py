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
    export_confusions,
    find_latest_checkpoint,
    parse_seed_list,
    run_single_method,
    set_global_seed,
    write_config,
    write_benchmark_report,
)
from evaluation.benchmark_visuals import plot_benchmark_dashboard, plot_run_scatter
from evaluation.real_data import (
    REAL_TEMPLATE_COLUMNS,
    build_real_data_checklist,
    export_real_checklist,
    export_real_condition_tables,
    export_real_manifest_summary,
    load_real_manifest,
)
from src.utils.device import DEVICE_CHOICES

LOGGER = logging.getLogger("real_benchmark")


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
    parser = argparse.ArgumentParser(description="Benchmark all four Morse decoders on a real-data manifest.")
    parser.add_argument("--manifest_csv", required=True, help="Path to real-data manifest following real_data_metadata_template.csv")
    parser.add_argument("--audio_dir", default="")
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--timing_device_note", default="")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    manifest_path = Path(args.manifest_csv)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Real-data manifest not found: {manifest_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_methods(args.methods)
    seeds = parse_seed_list(args.seeds)
    config = BenchmarkDecoderConfig(use_manifest_frequency=args.use_manifest_frequency)
    crnn_checkpoint = args.crnn_checkpoint or find_latest_checkpoint("crnn")
    conformer_checkpoint = args.conformer_checkpoint or find_latest_checkpoint("conformer")
    manifest = load_real_manifest(manifest_path)

    export_real_manifest_summary(output_dir, manifest)
    checklist = build_real_data_checklist(
        manifest=manifest,
        methods=methods,
        device_note=args.timing_device_note or args.device,
    )
    export_real_checklist(output_dir, checklist)

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
        LOGGER.info("Starting real-data run %d with seed=%d", run_index, seed)
        for method in methods:
            LOGGER.info("Benchmarking %s on real data (run=%d seed=%d)", method, run_index, seed)
            records, summary = run_single_method(
                decoder=decoders[method],
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
    write_benchmark_report(output_dir, aggregate_df, run_df, seeds=seeds, dataset_csv=str(manifest_path))

    write_config(
        output_dir=output_dir,
        dataset_csv=str(manifest_path),
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

    export_real_condition_tables(output_dir, records=all_records if len(seeds) == 1 else [record for record in all_records if int(record["seed"]) == seeds[0]], seed=seeds[0])
    plot_benchmark_dashboard(aggregate_df, output_dir)
    plot_run_scatter(run_df, output_dir)

    template_path = output_dir / "template_columns.txt"
    template_path.write_text("\n".join(REAL_TEMPLATE_COLUMNS), encoding="utf-8")
    LOGGER.info("Real-data benchmark artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
