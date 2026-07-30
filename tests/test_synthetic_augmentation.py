import random
import unittest

import numpy as np

from src.data.generate_dataset import sample_params
from src.data.morse_generator import MorseGenerator
from src.features.audio_processor import AudioProcessor


class TestSyntheticAugmentation(unittest.TestCase):
    def test_sample_params_stage3_targets_realistic_profiles(self):
        random.seed(0)
        params = sample_params(3)
        self.assertIn(
            params["noise_type"],
            {"room", "mic", "qrm", "realistic", "realistic_mic", "pink", "interference"},
        )
        self.assertLessEqual(float(params["snr_db"]), 18.0)
        self.assertGreaterEqual(float(params["frequency"]), 500.0)

    def test_morse_generator_realistic_mic_produces_context_and_metadata(self):
        np.random.seed(0)
        gen = MorseGenerator(sample_rate=16000, wpm=18, farnsworth_wpm=15, frequency=780)
        clean = gen.generate_audio("SOS", snr_db=None, noise_type="white")

        np.random.seed(0)
        noisy = gen.generate_audio("SOS", snr_db=4.0, noise_type="realistic_mic")

        self.assertEqual(noisy.dtype, np.float32)
        self.assertTrue(np.isfinite(noisy).all())
        self.assertGreater(len(noisy), len(clean))
        self.assertLessEqual(float(np.max(np.abs(noisy))), 0.90001)
        self.assertEqual(gen.last_render_metadata["channel_profile"], "speaker_mic_chain")
        self.assertTrue(gen.last_render_metadata["noise_components"])
        self.assertGreater(float(gen.last_render_metadata["pre_silence_s"]), 0.0)

    def test_audio_processor_heavy_augment_stays_valid_for_features(self):
        np.random.seed(1)
        processor = AudioProcessor(
            sample_rate=16000,
            n_mels=64,
            augment_noise_prob=1.0,
            augment_snr_min_db=6.0,
            augment_snr_max_db=6.0,
        )
        t = np.linspace(0.0, 1.2, int(1.2 * processor.sample_rate), endpoint=False, dtype=np.float32)
        audio = (0.6 * np.sin(2 * np.pi * 750.0 * t)).astype(np.float32)

        augmented = processor.augment_audio(audio)

        self.assertEqual(augmented.dtype, np.float32)
        self.assertEqual(augmented.ndim, 1)
        self.assertGreater(len(augmented), 0)
        self.assertTrue(np.isfinite(augmented).all())
        self.assertLessEqual(float(np.max(np.abs(augmented))), 1.00001)

        log_mel = processor.compute_log_mel(augmented)
        self.assertEqual(log_mel.shape[1], 64)


if __name__ == "__main__":
    unittest.main()
