import numpy as np
import librosa
import scipy.signal


class AudioProcessor:
    """
    Handles audio loading, preprocessing, and feature extraction.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=80,
        low_cut=400,
        high_cut=1200,
        filter_order=4,
        augment_noise_prob=0.65,
        augment_snr_min_db=6.0,
        augment_snr_max_db=24.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Bandpass defaults (wider, more robust)
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.filter_order = filter_order
        self.augment_noise_prob = float(augment_noise_prob)
        self.augment_snr_min_db = float(min(augment_snr_min_db, augment_snr_max_db))
        self.augment_snr_max_db = float(max(augment_snr_min_db, augment_snr_max_db))

    @staticmethod
    def _scale_to_target_rms(noise, target_rms):
        noise = np.asarray(noise, dtype=np.float32)
        rms = float(np.sqrt(np.mean(noise ** 2) + 1e-12))
        if rms <= 0.0 or target_rms <= 0.0:
            return np.zeros_like(noise, dtype=np.float32)
        return noise * (float(target_rms) / rms)

    def _colored_noise(self, n, alpha):
        if n <= 0:
            return np.array([], dtype=np.float32)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        scale = np.ones_like(freqs, dtype=np.float64)
        valid = freqs > 0
        scale[valid] = 1.0 / np.maximum(freqs[valid], 1.0) ** (alpha / 2.0)
        spectrum = (
            np.random.normal(0.0, 1.0, len(freqs)) +
            1j * np.random.normal(0.0, 1.0, len(freqs))
        ) * scale
        spectrum[0] = 0.0
        noise = np.fft.irfft(spectrum, n=n).astype(np.float32)
        return self._scale_to_target_rms(noise, 1.0)

    def _build_noise_component(self, n, noise_type):
        if n <= 0:
            return np.array([], dtype=np.float32)

        t = np.linspace(0, n / self.sample_rate, n, endpoint=False, dtype=np.float32)
        noise_type = str(noise_type).lower()

        if noise_type == "pink":
            noise = self._colored_noise(n, alpha=1.0)
        elif noise_type == "brown":
            noise = self._colored_noise(n, alpha=2.0)
        elif noise_type == "hum":
            mains = float(np.random.choice([50.0, 60.0]))
            noise = (
                np.sin(2 * np.pi * mains * t) +
                0.45 * np.sin(2 * np.pi * mains * 2.0 * t + 0.2) +
                0.15 * np.sin(2 * np.pi * mains * 3.0 * t + 0.4)
            ).astype(np.float32)
        elif noise_type == "interference":
            freq = np.random.uniform(550.0, 1050.0) + np.random.choice([-220.0, -120.0, 120.0, 220.0])
            noise = np.sin(2 * np.pi * freq * t + np.random.uniform(0.0, np.pi)).astype(np.float32)
            duty = float(np.random.uniform(0.2, 0.7))
            gate = 0.5 * (1.0 + scipy.signal.square(2 * np.pi * np.random.uniform(2.0, 8.0) * t, duty=duty))
            noise *= gate.astype(np.float32)
        elif noise_type == "bursts":
            noise = np.zeros(n, dtype=np.float32)
            for _ in range(int(np.random.randint(2, 6))):
                start = int(np.random.randint(0, max(1, n - 1)))
                width = int(np.random.randint(max(4, self.sample_rate // 320), max(8, self.sample_rate // 25)))
                end = min(n, start + width)
                window = np.hanning(max(2, end - start)).astype(np.float32)
                noise[start:end] += np.random.normal(0.0, 1.0, end - start).astype(np.float32) * window[: end - start]
        elif noise_type == "room":
            noise = (
                0.45 * self._colored_noise(n, alpha=1.0) +
                0.3 * np.random.normal(0.0, 1.0, n).astype(np.float32) +
                0.15 * self._build_noise_component(n, "hum") +
                0.1 * self._build_noise_component(n, "bursts")
            )
        elif noise_type == "qrm":
            noise = (
                0.35 * self._build_noise_component(n, "hum") +
                0.35 * self._build_noise_component(n, "interference") +
                0.2 * self._build_noise_component(n, "bursts") +
                0.1 * np.random.normal(0.0, 1.0, n).astype(np.float32)
            )
        else:
            noise = np.random.normal(0.0, 1.0, n).astype(np.float32)

        return np.asarray(noise, dtype=np.float32)

    def _inject_noise_by_type(self, audio, noise_type, target_noise_rms):
        if len(audio) == 0 or target_noise_rms <= 0.0:
            return audio.astype(np.float32)
        noise = self._build_noise_component(len(audio), noise_type)
        return (audio + self._scale_to_target_rms(noise, target_noise_rms)).astype(np.float32)

    def _apply_qsb(self, audio):
        if len(audio) == 0:
            return audio.astype(np.float32)
        t = np.arange(len(audio), dtype=np.float32) / float(self.sample_rate)
        depth = float(np.random.uniform(0.12, 0.45))
        slow = 0.5 * (1.0 + np.sin(2 * np.pi * np.random.uniform(0.08, 0.8) * t + np.random.uniform(0.0, np.pi)))
        fast = 0.5 * (1.0 + np.sin(2 * np.pi * np.random.uniform(0.8, 3.0) * t + np.random.uniform(0.0, np.pi)))
        envelope = 1.0 - depth * (0.7 * slow + 0.3 * fast)
        return (audio * np.clip(envelope, 0.2, 1.25)).astype(np.float32)

    def _apply_room_reverb(self, audio):
        if len(audio) == 0:
            return audio.astype(np.float32)
        tap_count = int(np.random.randint(2, 5))
        delays_ms = np.random.uniform(10.0, 70.0, tap_count)
        gains = np.random.uniform(0.08, 0.3, tap_count)
        max_delay = int(round(np.max(delays_ms) * self.sample_rate / 1000.0))
        ir = np.zeros(max_delay + 1, dtype=np.float32)
        ir[0] = 1.0
        for delay_ms, gain in zip(delays_ms, gains):
            ir[int(round(delay_ms * self.sample_rate / 1000.0))] += float(gain)
        wet = scipy.signal.fftconvolve(audio, ir, mode="full")[: len(audio)].astype(np.float32)
        wet_peak = float(np.max(np.abs(wet))) if wet.size else 0.0
        if wet_peak > 1e-9:
            wet = wet / wet_peak
        mix = float(np.random.uniform(0.06, 0.22))
        return ((1.0 - mix) * audio + mix * wet).astype(np.float32)

    def _apply_channel_response(self, audio):
        if len(audio) == 0:
            return audio.astype(np.float32)
        nyquist = 0.5 * self.sample_rate
        low = float(np.random.uniform(250.0, 520.0)) / nyquist
        high = float(np.random.uniform(1300.0, min(3200.0, nyquist * 0.95))) / nyquist
        if not (0.0 < low < high < 1.0):
            return audio.astype(np.float32)
        sos = scipy.signal.butter(4, [low, high], btype='band', output='sos')
        return scipy.signal.sosfiltfilt(sos, audio).astype(np.float32)

    def _apply_codec_artifacts(self, audio):
        if len(audio) == 0:
            return audio.astype(np.float32)
        codec_rate = int(np.random.choice([8000, 11025, 12000]))
        degraded = scipy.signal.resample_poly(audio, codec_rate, self.sample_rate).astype(np.float32)
        degraded = scipy.signal.resample_poly(degraded, self.sample_rate, codec_rate).astype(np.float32)
        if len(degraded) < len(audio):
            degraded = np.pad(degraded, (0, len(audio) - len(degraded)))
        degraded = degraded[: len(audio)]
        return (np.round(np.clip(degraded, -1.0, 1.0) * 2048.0) / 2048.0).astype(np.float32)

    def _apply_dropouts(self, audio):
        if len(audio) == 0:
            return audio.astype(np.float32)
        result = audio.astype(np.float32).copy()
        for _ in range(int(np.random.randint(1, 4))):
            start = int(np.random.randint(0, max(1, len(audio) - 1)))
            width = int(np.random.randint(max(4, self.sample_rate // 350), max(8, self.sample_rate // 18)))
            end = min(len(audio), start + width)
            result[start:end] *= float(np.random.uniform(0.15, 0.7))
        return result

    def _apply_clicks(self, audio):
        if len(audio) == 0:
            return audio.astype(np.float32)
        clicks = np.zeros_like(audio, dtype=np.float32)
        for _ in range(int(np.random.randint(1, 4))):
            start = int(np.random.randint(0, max(1, len(audio) - 1)))
            width = int(np.random.randint(2, max(3, self.sample_rate // 260)))
            end = min(len(audio), start + width)
            win = np.hanning(max(2, end - start)).astype(np.float32)
            clicks[start:end] += float(np.random.uniform(0.08, 0.4) * np.random.choice([-1.0, 1.0])) * win[: end - start]
        return (audio + clicks).astype(np.float32)

    def load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.array([])

    def soft_clip(self, audio, threshold=0.95):
        scale = threshold
        clipped = np.tanh(audio / scale) * scale
        return clipped

    def clean_audio(self, audio):
        """
        Soft Clip -> Bandpass -> AGC
        """
        if len(audio) == 0:
            return audio

        audio = self.soft_clip(audio, threshold=0.95)

        # Bandpass filter (optional if cutoffs invalid)
        nyquist = 0.5 * self.sample_rate
        low = float(self.low_cut) / nyquist
        high = float(self.high_cut) / nyquist

        if 0.0 < low < high < 1.0:
            b, a = scipy.signal.butter(self.filter_order, [low, high], btype='band')
            filtered = scipy.signal.filtfilt(b, a, audio)
        else:
            filtered = audio

        peak = np.max(np.abs(filtered))
        if peak > 0:
            normalized = filtered / peak
        else:
            normalized = filtered

        return normalized

    def augment_audio(self, audio):
        if len(audio) == 0:
            return audio

        audio = np.asarray(audio, dtype=np.float32)

        # 1) Random crop (streaming simulation)
        if np.random.random() < 0.25:
            gap = int(0.05 * self.sample_rate)
            if len(audio) > 2 * gap:
                start = np.random.randint(0, gap)
                end = len(audio) - np.random.randint(0, gap)
                audio = audio[start:end]

        # 2) Optional pad to simulate onset uncertainty from microphone captures
        if np.random.random() < 0.35:
            pre = int(np.random.uniform(0.0, 0.45) * self.sample_rate)
            post = int(np.random.uniform(0.0, 0.25) * self.sample_rate)
            if pre > 0 or post > 0:
                audio = np.pad(audio, (pre, post))

        # 3) Time stretching via resample (speed + pitch)
        if np.random.random() < 0.55:
            rate = np.random.uniform(0.78, 1.18)
            try:
                new_len = int(len(audio) / rate)
                audio = scipy.signal.resample(audio, new_len)
            except Exception:
                pass

        # 4) Channel coloration before additive noise
        if np.random.random() < 0.45:
            audio = self._apply_channel_response(audio)

        if np.random.random() < 0.25:
            audio = self._apply_room_reverb(audio)

        # 5) Noise injection
        if np.random.random() < self.augment_noise_prob:
            noise_type = np.random.choice(
                ['white', 'pink', 'hum', 'interference', 'room', 'qrm', 'bursts'],
                p=[0.15, 0.2, 0.15, 0.15, 0.15, 0.12, 0.08],
            )
            snr_db = np.random.uniform(self.augment_snr_min_db, self.augment_snr_max_db)

            active = np.abs(audio) > 1e-4
            signal_power = np.mean(audio[active] ** 2) if np.any(active) else np.mean(audio ** 2)
            if signal_power > 0:
                noise_power = signal_power / (10 ** (snr_db / 10.0))
                target_noise_rms = float(np.sqrt(noise_power))
                audio = self._inject_noise_by_type(audio, noise_type, target_noise_rms)

        # 6) QSB / gain flutter
        if np.random.random() < 0.35:
            audio = self._apply_qsb(audio)

        # 7) Dropouts, clicks and codec-like degradation
        if np.random.random() < 0.2:
            audio = self._apply_dropouts(audio)

        if np.random.random() < 0.18:
            audio = self._apply_clicks(audio)

        if np.random.random() < 0.18:
            audio = self._apply_codec_artifacts(audio)

        # 8) Dynamic range compression / clipping
        if np.random.random() < 0.35:
            threshold = float(np.random.uniform(0.55, 0.9))
            audio = self.soft_clip(audio, threshold=threshold)

        # Re-normalize lightly to avoid extreme scale drift post-augment
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        return audio.astype(np.float32)

    def compute_spectrogram(self, audio):
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(stft)
        return spectrogram

    def compute_log_mel(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel.T  # (T, n_mels)

    def apply_cmvn(self, features):
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        return (features - mean) / (std + 1e-9)
