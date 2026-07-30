import unittest

import pandas as pd

from evaluation.benchmark_runner import aggregate_repeated_runs, parse_seed_list


class TestBenchmarkRunner(unittest.TestCase):
    def test_parse_seed_list(self):
        self.assertEqual(parse_seed_list("13, 42,123"), [13, 42, 123])

    def test_aggregate_repeated_runs(self):
        rows = [
            {
                "method": "crnn",
                "run_index": 0,
                "seed": 13,
                "num_samples": 10,
                "num_failures": 1,
                "cer": 0.1,
                "wer": 0.2,
                "exact_match_rate": 0.8,
                "decode_failure_rate": 0.1,
                "cer_percent": 10.0,
                "wer_percent": 20.0,
                "exact_match_percent": 80.0,
                "decode_failure_percent": 10.0,
                "tone_detection_failure_count": 0,
                "tone_detection_failure_rate": 0.0,
                "timing_parse_failure_count": 0,
                "timing_parse_failure_rate": 0.0,
                "symbol_mapping_failure_count": 1,
                "symbol_mapping_failure_rate": 0.1,
                "rtf_cpu": 0.5,
                "latency_mean_ms": 15.0,
                "latency_std_ms": 2.0,
                "latency_p50_ms": 14.0,
                "latency_p90_ms": 18.0,
                "total_processing_time_sec": 1.0,
                "total_audio_duration_sec": 2.0,
                "checkpoint_path": "a",
                "device": "cpu",
            },
            {
                "method": "crnn",
                "run_index": 1,
                "seed": 42,
                "num_samples": 10,
                "num_failures": 1,
                "cer": 0.3,
                "wer": 0.4,
                "exact_match_rate": 0.6,
                "decode_failure_rate": 0.1,
                "cer_percent": 30.0,
                "wer_percent": 40.0,
                "exact_match_percent": 60.0,
                "decode_failure_percent": 10.0,
                "tone_detection_failure_count": 0,
                "tone_detection_failure_rate": 0.0,
                "timing_parse_failure_count": 0,
                "timing_parse_failure_rate": 0.0,
                "symbol_mapping_failure_count": 1,
                "symbol_mapping_failure_rate": 0.1,
                "rtf_cpu": 0.7,
                "latency_mean_ms": 17.0,
                "latency_std_ms": 3.0,
                "latency_p50_ms": 16.0,
                "latency_p90_ms": 20.0,
                "total_processing_time_sec": 1.4,
                "total_audio_duration_sec": 2.0,
                "checkpoint_path": "a",
                "device": "cpu",
            },
        ]
        dataframe = aggregate_repeated_runs(rows)
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(dataframe.iloc[0]["method"], "crnn")
        self.assertAlmostEqual(dataframe.iloc[0]["cer_percent_mean"], 20.0)
        self.assertAlmostEqual(dataframe.iloc[0]["cer_percent_std"], 10.0)


if __name__ == "__main__":
    unittest.main()
