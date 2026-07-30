import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.morse_generator import MorseGenerator
from src.real_style import (
    AlignmentConfig,
    DetectedRun,
    build_expected_units,
    canonicalize_label_text,
    format_label_text,
    infer_gap_formatted_text,
    load_unit_bank,
    save_unit_bank,
    synthesize_from_unit_bank,
    write_wav,
    align_expected_to_observed,
)


class TestRealStyleUtilities(unittest.TestCase):
    def test_canonicalize_char_spaced_text(self):
        self.assertEqual(canonicalize_label_text("A L O V U"), "ALOVU")
        self.assertEqual(canonicalize_label_text("CQ 73"), "CQ 73")
        self.assertEqual(format_label_text("ALOVU", "char_spaced"), "A L O V U")
        self.assertEqual(format_label_text("CQ 73", "compact"), "CQ 73")

    def test_build_expected_units_for_a(self):
        units = build_expected_units("A")
        self.assertEqual([unit.kind for unit in units], ["dot", "intra_gap", "dash"])
        self.assertEqual([unit.nominal_units for unit in units], [1.0, 1.0, 3.0])

    def test_alignment_matches_simple_sequence(self):
        expected = build_expected_units("A")
        observed = [
            DetectedRun(index=0, is_tone=True, start_frame=0, end_frame=2, start_sample=0, end_sample=10, duration_s=0.1, units=1.0),
            DetectedRun(index=1, is_tone=False, start_frame=2, end_frame=4, start_sample=10, end_sample=20, duration_s=0.1, units=1.0),
            DetectedRun(index=2, is_tone=True, start_frame=4, end_frame=10, start_sample=20, end_sample=50, duration_s=0.3, units=3.0),
        ]
        result = align_expected_to_observed(expected, observed, AlignmentConfig())
        self.assertEqual(result.matched_count, 3)
        self.assertAlmostEqual(result.coverage, 1.0)

    def test_synthesize_from_bank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            bank_dir = tmpdir_path / "bank"

            sample_rate = 16000
            dot_path = bank_dir / "units" / "dot" / "dot.wav"
            dash_path = bank_dir / "units" / "dash" / "dash.wav"
            intra_gap_path = bank_dir / "units" / "intra_gap" / "intra.wav"
            write_wav(dot_path, np.ones(800, dtype=np.float32) * 0.25, sample_rate)
            write_wav(dash_path, np.ones(2400, dtype=np.float32) * 0.25, sample_rate)
            write_wav(intra_gap_path, np.zeros(800, dtype=np.float32), sample_rate)

            payload = save_unit_bank(
                out_dir=bank_dir,
                sample_rate=sample_rate,
                entries=[
                    {"kind": "dot", "path": str(dot_path), "source_id": "test"},
                    {"kind": "dash", "path": str(dash_path), "source_id": "test"},
                    {"kind": "intra_gap", "path": str(intra_gap_path), "source_id": "test"},
                ],
                records=[],
                meta={"test": True},
            )
            self.assertTrue((bank_dir / "bank.json").exists())
            self.assertEqual(payload["counts_by_kind"]["dot"], 1)

            bank = load_unit_bank(bank_dir)
            audio, loaded_sample_rate = synthesize_from_unit_bank("A", bank, seed=7)
            self.assertEqual(loaded_sample_rate, sample_rate)
            self.assertGreater(len(audio), 3000)
            self.assertGreater(float(np.max(np.abs(audio))), 0.0)

    def test_infer_gap_formatted_text_groups_relative_long_gaps(self):
        np.random.seed(0)
        generator = MorseGenerator(sample_rate=16000, frequency=1000, wpm=10, farnsworth_wpm=10)
        audio = generator.generate_audio("A L O   V U   A   V U", snr_db=80.0, noise_type="white")
        result = infer_gap_formatted_text(audio, sample_rate=16000, text="ALOVUAVU")
        self.assertEqual(result.formatted_text, "ALO VU A VU")
        self.assertIsNotNone(result.threshold_units)

    def test_infer_gap_formatted_text_keeps_compact_text_without_gap_clusters(self):
        np.random.seed(0)
        generator = MorseGenerator(sample_rate=16000, frequency=1000, wpm=14, farnsworth_wpm=14)
        audio = generator.generate_audio("ALOVUAVU", snr_db=80.0, noise_type="white")
        result = infer_gap_formatted_text(audio, sample_rate=16000, text="ALOVUAVU")
        self.assertEqual(result.formatted_text, "ALOVUAVU")
        self.assertIsNone(result.threshold_units)


if __name__ == "__main__":
    unittest.main()
