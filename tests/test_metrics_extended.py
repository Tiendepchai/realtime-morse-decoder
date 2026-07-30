import unittest

from evaluation.confusion import INS_TOKEN, align_token_sequences, build_confusion_counter
from evaluation.metrics_extended import aggregate_metrics, calculate_cer, calculate_wer, enrich_prediction_records
from src.utils.text import normalize_text


class TestMetricsExtended(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  cq\tde  k1abc!! "), "CQ DE K1ABC")

    def test_alignment_marks_insertions(self):
        alignment = align_token_sequences(list("AB"), list("ACB"))
        self.assertEqual(alignment, [("A", "A"), (INS_TOKEN, "C"), ("B", "B")])

    def test_metric_aggregation(self):
        enriched = enrich_prediction_records(
            [
                {
                    "sample_id": "one",
                    "reference": "SOS",
                    "prediction": "SOS",
                    "method": "energy_threshold",
                    "is_failure": False,
                    "failure_type": "",
                },
                {
                    "sample_id": "two",
                    "reference": "CQ TEST",
                    "prediction": "CQ TST",
                    "method": "energy_threshold",
                    "is_failure": False,
                    "failure_type": "",
                },
            ]
        )
        summary = aggregate_metrics(enriched, method="energy_threshold")
        self.assertEqual(summary["num_samples"], 2)
        self.assertAlmostEqual(summary["exact_match_rate"], 0.5)
        self.assertGreater(summary["cer"], 0.0)
        self.assertGreater(summary["wer"], 0.0)

    def test_confusion_counter_uses_normalized_fields(self):
        records = enrich_prediction_records(
            [
                {
                    "reference": "SOS",
                    "prediction": "SO5",
                    "method": "goertzel",
                    "is_failure": False,
                    "failure_type": "",
                }
            ]
        )
        counter = build_confusion_counter(records)
        self.assertEqual(counter[("S", "5")], 1)

    def test_cer_and_wer_normalize_before_scoring(self):
        self.assertEqual(calculate_cer("cq", "CQ"), 0.0)
        self.assertEqual(calculate_wer(" CQ   TEST ", "cq test"), 0.0)


if __name__ == "__main__":
    unittest.main()
