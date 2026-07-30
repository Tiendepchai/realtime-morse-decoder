import tempfile
import unittest
from pathlib import Path

import pandas as pd
import soundfile as sf

from evaluation.real_data import (
    build_real_data_checklist,
    export_real_condition_tables,
    load_real_manifest,
    wpm_to_bin,
)


class TestRealDataUtilities(unittest.TestCase):
    def test_wpm_to_bin(self):
        self.assertEqual(wpm_to_bin(7), "5-10")
        self.assertEqual(wpm_to_bin(18), "11-20")
        self.assertEqual(wpm_to_bin(25), "21-30")
        self.assertEqual(wpm_to_bin(33), "31-35")
        self.assertEqual(wpm_to_bin(40), "")

    def test_load_manifest_and_export_condition_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = tmpdir_path / "sample.wav"
            sf.write(audio_path, [0.0] * 16000, 16000)

            manifest_path = tmpdir_path / "real_manifest.csv"
            pd.DataFrame(
                [
                    {
                        "sample_id": "real_0001",
                        "audio_path": str(audio_path),
                        "reference": "CQ 73",
                        "duration_sec": 0.0,
                        "split": "test",
                        "source_type": "microphone",
                        "device_type": "consumer_mic",
                        "device_model": "usb_mic",
                        "session_id": "s1",
                        "operator_id": "op1",
                        "environment_tag": "continuous_background",
                        "noise_type": "qrm",
                        "delta_f_hz": 32.0,
                        "wpm_est": 18.0,
                        "wpm_bin": "",
                        "farnsworth": "no",
                        "qrm_present": "yes",
                        "label_vocab_ok": "",
                    }
                ]
            ).to_csv(manifest_path, index=False)

            manifest = load_real_manifest(manifest_path)
            self.assertEqual(manifest.loc[0, "wpm_bin"], "11-20")
            self.assertTrue(bool(manifest.loc[0, "audio_format_ok"]))
            self.assertTrue(bool(manifest.loc[0, "label_vocab_ok"]))

            checklist = build_real_data_checklist(manifest, methods=["energy_threshold", "goertzel", "crnn", "conformer"], device_note="cpu")
            self.assertTrue(any(item.section == "Dataset Readiness" for item in checklist))

            records = [
                {
                    "method": "crnn",
                    "reference": "CQ 73",
                    "prediction": "CQ73",
                    "reference_normalized": "CQ 73",
                    "prediction_normalized": "CQ73",
                    "is_failure": False,
                    "failure_type": "",
                    "wpm_bin": "11-20",
                    "environment_tag": "continuous_background",
                    "noise_type": "qrm",
                    "qrm_present": "yes",
                    "delta_f_20_80": True,
                }
            ]
            output_path = export_real_condition_tables(tmpdir_path, records=records, seed=13)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
