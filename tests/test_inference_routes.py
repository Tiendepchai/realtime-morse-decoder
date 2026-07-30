import unittest

import numpy as np

from demo_realtime_mic import PredictionOutput
from src.data.morse_generator import MorseGenerator
from src.inference import (
    RouteEmission,
    RoutedInferenceResult,
    apply_gap_formatting_to_routed_result,
    run_realtime_route_on_audio,
)


class FakePredictor:
    method_name = "conformer"
    segment_pre_padding_samples = 0
    segment_post_padding_samples = 0

    def __init__(self, text: str):
        self.text = text

    def predict(self, audio_chunk: np.ndarray) -> PredictionOutput:
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if audio_chunk.size == 0 or float(np.sqrt(np.mean(audio_chunk ** 2))) < 0.01:
            return PredictionOutput(text="", meta={"reason": "silence"})
        return PredictionOutput(text=self.text, meta={"num_frames": 10})


class TestInferenceRoutes(unittest.TestCase):
    def test_segment_route_emits_finalized_segment(self):
        predictor = FakePredictor("SOS")
        tone = np.ones(int(1.5 * 16000), dtype=np.float32) * 0.2
        silence = np.zeros(int(0.75 * 16000), dtype=np.float32)
        audio = np.concatenate([tone, silence]).astype(np.float32)

        result = run_realtime_route_on_audio(
            audio=audio,
            sample_rate=16000,
            predictor=predictor,
            route="realtime-segment",
            step_duration=0.25,
            chunk_duration=0.25,
            reset_silence_steps=2,
            min_segment_duration=0.2,
            max_segment_duration=10.0,
        )

        self.assertEqual(result.final_text, "SOS")
        self.assertEqual(len(result.emissions), 1)
        self.assertEqual(result.emissions[0].stable_text, "SOS")
        self.assertEqual(result.emissions[0].kind, "segment")

    def test_sliding_route_emits_stable_prediction(self):
        predictor = FakePredictor("SOS")
        tone = np.ones(int(2.5 * 16000), dtype=np.float32) * 0.2

        result = run_realtime_route_on_audio(
            audio=tone,
            sample_rate=16000,
            predictor=predictor,
            route="realtime-sliding",
            buffer_duration=1.0,
            step_duration=0.25,
            chunk_duration=0.25,
            confirm_steps=1,
        )

        self.assertEqual(result.final_text, "SOS")
        self.assertGreaterEqual(len(result.emissions), 1)
        self.assertEqual(result.emissions[0].emitted_text, "SOS")
        self.assertEqual(result.emissions[0].kind, "sliding")

    def test_apply_gap_formatting_to_routed_result_reinserts_relative_word_gaps(self):
        np.random.seed(0)
        generator = MorseGenerator(sample_rate=16000, frequency=1000, wpm=10, farnsworth_wpm=10)
        audio = generator.generate_audio("A L O   V U   A   V U", snr_db=80.0, noise_type="white")
        raw_result = RoutedInferenceResult(
            route="realtime-segment",
            method="conformer",
            audio_path="synthetic.wav",
            sample_rate=16000,
            final_text="ALOVUAVU",
            raw_final_text="ALOVUAVU",
            emissions=[
                RouteEmission(
                    kind="segment",
                    stable_text="ALOVUAVU",
                    emitted_text="ALOVUAVU",
                    start_s=0.0,
                    end_s=float(len(audio)) / 16000.0,
                    latency_ms=0.0,
                    meta={},
                )
            ],
        )

        formatted_result = apply_gap_formatting_to_routed_result(
            result=raw_result,
            audio=audio,
            sample_rate=16000,
            gap_format="relative",
        )

        self.assertEqual(formatted_result.final_text, "ALO VU A VU")
        self.assertEqual(formatted_result.raw_final_text, "ALOVUAVU")
        self.assertEqual(formatted_result.emissions[0].stable_text, "ALO VU A VU")
        self.assertEqual(formatted_result.emissions[0].meta["gap_format"], "relative")


if __name__ == "__main__":
    unittest.main()
