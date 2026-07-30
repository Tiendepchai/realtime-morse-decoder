import unittest

import numpy as np

from baselines.energy_threshold_baseline import EnergyThresholdBaseline, EnergyThresholdBaselineConfig
from baselines.goertzel_baseline import GoertzelBaseline, GoertzelBaselineConfig
from baselines.morse_rules import CHAR_TO_MORSE, FAILURE_SYMBOL_MAPPING, DurationRules, MorseRun, decode_runs_to_text
from baselines.timing_estimation import DotEstimationConfig, estimate_dot_unit


def synthesize_morse_audio(
    text: str,
    sample_rate: int = 16000,
    frequency_hz: float = 700.0,
    dot_duration_s: float = 0.06,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    words = text.split()

    for word_index, word in enumerate(words):
        for char_index, char in enumerate(word):
            code = CHAR_TO_MORSE[char]
            for symbol_index, symbol in enumerate(code):
                tone_duration_s = dot_duration_s if symbol == "." else 3.0 * dot_duration_s
                num_samples = max(1, int(round(tone_duration_s * sample_rate)))
                time_axis = np.arange(num_samples, dtype=np.float32) / sample_rate
                tone = 0.8 * np.sin(2.0 * np.pi * frequency_hz * time_axis)
                pieces.append(tone.astype(np.float32))

                if symbol_index < len(code) - 1:
                    intra_gap = np.zeros(int(round(dot_duration_s * sample_rate)), dtype=np.float32)
                    pieces.append(intra_gap)

            if char_index < len(word) - 1:
                inter_letter_gap = np.zeros(int(round(3.0 * dot_duration_s * sample_rate)), dtype=np.float32)
                pieces.append(inter_letter_gap)

        if word_index < len(words) - 1:
            inter_word_gap = np.zeros(int(round(7.0 * dot_duration_s * sample_rate)), dtype=np.float32)
            pieces.append(inter_word_gap)

    return np.concatenate(pieces).astype(np.float32)


class TestTraditionalBaselines(unittest.TestCase):
    def test_timing_estimation_and_rule_decoder(self):
        unit = 0.06
        runs = [
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
            MorseRun(False, 3, 3 * unit),
            MorseRun(True, 3, 3 * unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 3, 3 * unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 3, 3 * unit),
            MorseRun(False, 3, 3 * unit),
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
        ]
        dot_unit_s, _ = estimate_dot_unit(runs, DotEstimationConfig())
        outcome = decode_runs_to_text(runs, dot_unit_s, DurationRules())
        self.assertFalse(outcome.is_failure)
        self.assertEqual(outcome.prediction, "SOS")

    def test_invalid_symbol_sets_mapping_failure(self):
        unit = 0.05
        runs = [
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
            MorseRun(False, 1, unit),
            MorseRun(True, 1, unit),
        ]
        outcome = decode_runs_to_text(runs, unit, DurationRules())
        self.assertTrue(outcome.is_failure)
        self.assertEqual(outcome.failure_type, FAILURE_SYMBOL_MAPPING)

    def test_energy_baseline_decodes_synthetic_sos(self):
        audio = synthesize_morse_audio("SOS")
        decoder = EnergyThresholdBaseline(
            EnergyThresholdBaselineConfig(
                sample_rate=16000,
                target_frequency_hz=700.0,
                auto_frequency=False,
                frame_length_ms=20.0,
                hop_length_ms=10.0,
            )
        )
        result = decoder.decode_audio(audio)
        self.assertEqual(result["prediction"], "SOS")
        self.assertFalse(result["is_failure"])

    def test_goertzel_baseline_decodes_synthetic_sos(self):
        audio = synthesize_morse_audio("SOS")
        decoder = GoertzelBaseline(
            GoertzelBaselineConfig(
                sample_rate=16000,
                target_frequency_hz=700.0,
                auto_frequency=False,
                frame_length_ms=20.0,
                hop_length_ms=10.0,
            )
        )
        result = decoder.decode_audio(audio)
        self.assertEqual(result["prediction"], "SOS")
        self.assertFalse(result["is_failure"])


if __name__ == "__main__":
    unittest.main()
