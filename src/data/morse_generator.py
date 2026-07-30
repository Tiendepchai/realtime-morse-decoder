from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf


class MorseGenerator:
    """
    Generates synthetic Morse code audio from text.

    Key fixes vs your version:
    - generate_audio(..., noise_type=...) supported
    - SNR is computed on ACTIVE (non-silence) samples, not full waveform
    - Optional peak normalization to avoid clipping
    """

    CODE = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..',
        '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....',
        '6': '-....', '7': '--...', '8': '---..', '9': '----.', '0': '-----',
        ' ': ' ', '.': '.-.-.-', ',': '--..--', '?': '..--..', '/': '-..-.',
        '@': '.--.-.', '=': '-...-'
    }

    REALISTIC_NOISE_TYPES = {
        "pink",
        "brown",
        "room",
        "mic",
        "qrm",
        "realistic",
        "realistic_mic",
    }

    def __init__(
        self,
        sample_rate=16000,
        frequency=700,
        wpm=20,
        farnsworth_wpm=None,
        timing_jitter=0.0,   # fraction: 0.0..0.15
        freq_drift=0.0,      # Hz/s
        freq_offset=0.0,     # Hz
        harmonic_mix=0.08,
    ):
        self.sample_rate = int(sample_rate)
        self.frequency = float(frequency)
        self.wpm = float(wpm)
        self.farnsworth_wpm = float(farnsworth_wpm) if farnsworth_wpm else float(wpm)
        self.timing_jitter = float(timing_jitter)
        self.freq_drift = float(freq_drift)
        self.freq_offset = float(freq_offset)
        self.harmonic_mix = float(max(0.0, harmonic_mix))
        self.last_render_metadata: dict[str, float | int | str] = {}
        self.last_noise_components: list[str] = []

        # Basic element durations
        self.dot_len = 1.2 / self.wpm
        self.dash_len = 3.0 * self.dot_len

        # Farnsworth spacing (commonly dot_len computed from Farnsworth speed)
        spacing_dot_len = 1.2 / self.farnsworth_wpm
        self.intra_char_space = spacing_dot_len
        self.inter_char_space = 3.0 * spacing_dot_len
        self.word_space = 7.0 * spacing_dot_len

    def apply_jitter(self, duration: float) -> float:
        if self.timing_jitter > 0:
            noise = np.random.uniform(-self.timing_jitter, self.timing_jitter)
            return max(0.0, duration * (1.0 + noise))
        return max(0.0, duration)

    def generate_tone(self, duration: float, current_freq: float | None = None) -> np.ndarray:
        if current_freq is None:
            current_freq = self.frequency + self.freq_offset

        duration = self.apply_jitter(duration)
        n = int(self.sample_rate * duration)
        if n <= 0:
            return np.array([], dtype=np.float32)

        t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)

        # Envelope: soften edges to simulate keying
        envelope = np.ones_like(t, dtype=np.float32)
        ramp_time = 0.005  # 5 ms
        ramp_samples = int(ramp_time * self.sample_rate)
        if ramp_samples > 0 and n > 2 * ramp_samples:
            envelope[:ramp_samples] = np.linspace(0, 1, ramp_samples, dtype=np.float32)
            envelope[-ramp_samples:] = np.linspace(1, 0, ramp_samples, dtype=np.float32)

        # Frequency drift
        if self.freq_drift != 0.0:
            phase = 2 * np.pi * (current_freq * t + 0.5 * self.freq_drift * (t ** 2))
        else:
            phase = 2 * np.pi * current_freq * t

        sig = np.sin(phase).astype(np.float32)
        if self.harmonic_mix > 0.0:
            harmonic = (
                0.35 * np.sin(2.0 * phase + 0.15) +
                0.15 * np.sin(3.0 * phase + 0.3)
            ).astype(np.float32)
            sig = sig + self.harmonic_mix * harmonic

        sig = sig.astype(np.float32) * envelope
        peak = float(np.max(np.abs(sig))) if sig.size else 0.0
        if peak > 1.0:
            sig = sig / peak
        return sig

    def generate_silence(self, duration: float) -> np.ndarray:
        duration = self.apply_jitter(duration)
        n = int(self.sample_rate * duration)
        if n <= 0:
            return np.array([], dtype=np.float32)
        return np.zeros(n, dtype=np.float32)

    def text_to_morse_str(self, text: str):
        return [self.CODE.get(c.upper(), '') for c in text]

    @staticmethod
    def _normalize_std(signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        std = float(np.std(signal))
        if std <= 1e-9:
            return np.zeros_like(signal, dtype=np.float32)
        return signal / std

    def _colored_noise(self, n: int, alpha: float) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=np.float32)

        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        scale = np.ones_like(freqs, dtype=np.float64)
        valid = freqs > 0
        scale[valid] = 1.0 / np.maximum(freqs[valid], 1.0) ** (alpha / 2.0)
        real = np.random.normal(0.0, 1.0, len(freqs))
        imag = np.random.normal(0.0, 1.0, len(freqs))
        spectrum = (real + 1j * imag) * scale
        spectrum[0] = 0.0
        noise = np.fft.irfft(spectrum, n=n)
        return self._normalize_std(noise.astype(np.float32))

    def _build_noise_component(self, kind: str, n: int, t: np.ndarray) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=np.float32)

        if kind == "pink":
            return self._colored_noise(n, alpha=1.0)

        if kind == "brown":
            return self._colored_noise(n, alpha=2.0)

        if kind == "hum":
            mains = float(np.random.choice([50.0, 60.0]))
            hum = (
                np.sin(2 * np.pi * mains * t) +
                0.45 * np.sin(2 * np.pi * mains * 2.0 * t + 0.2) +
                0.15 * np.sin(2 * np.pi * mains * 3.0 * t + 0.4)
            )
            return self._normalize_std(hum)

        if kind == "interference":
            offset = float(np.random.choice([-1.0, 1.0]) * np.random.uniform(80.0, 280.0))
            interferer_freq = float(self.frequency + self.freq_offset + offset)
            interferer = np.sin(2 * np.pi * interferer_freq * t + np.random.uniform(0.0, np.pi))
            gate_hz = np.random.uniform(2.0, 9.0)
            gate = 0.5 * (1.0 + scipy.signal.square(2.0 * np.pi * gate_hz * t, duty=np.random.uniform(0.2, 0.7)))
            interferer = interferer * gate.astype(np.float32)
            return self._normalize_std(interferer)

        if kind == "bursts":
            bursts = np.zeros(n, dtype=np.float32)
            burst_count = int(np.random.randint(2, 7))
            for _ in range(burst_count):
                start = int(np.random.randint(0, max(1, n - 1)))
                length = int(np.random.randint(max(4, self.sample_rate // 300), max(8, self.sample_rate // 35)))
                end = min(n, start + length)
                width = end - start
                if width <= 1:
                    continue
                window = np.hanning(width).astype(np.float32)
                bursts[start:end] += np.random.normal(0.0, 1.0, width).astype(np.float32) * window
            return self._normalize_std(bursts)

        return self._normalize_std(np.random.normal(0.0, 1.0, n).astype(np.float32))

    def _resolve_noise_components(self, noise_type: str) -> list[str]:
        noise_key = str(noise_type or "white").lower()
        if noise_key == "hum":
            return ["hum"]
        if noise_key == "interference":
            return ["interference"]
        if noise_key == "pink":
            return ["pink"]
        if noise_key == "brown":
            return ["brown"]
        if noise_key == "room":
            return ["pink", "white", "hum", "bursts"]
        if noise_key in {"mic", "realistic_mic"}:
            return ["pink", "hum", "interference", "bursts"]
        if noise_key == "qrm":
            return ["hum", "interference", "bursts", "white"]
        if noise_key == "realistic":
            pool = ["white", "pink", "hum", "interference", "bursts"]
            count = int(np.random.randint(2, 5))
            chosen = list(np.random.choice(pool, size=count, replace=False))
            return [str(item) for item in chosen]
        return ["white"]

    def _add_context_padding(self, audio: np.ndarray, noise_type: str) -> tuple[np.ndarray, float, float]:
        noise_key = str(noise_type or "white").lower()
        if noise_key in {"mic", "realistic_mic", "qrm"}:
            pre_s = float(np.random.uniform(0.15, 1.2))
            post_s = float(np.random.uniform(0.05, 0.7))
        elif noise_key in {"room", "realistic", "pink", "brown"}:
            pre_s = float(np.random.uniform(0.05, 0.5))
            post_s = float(np.random.uniform(0.05, 0.35))
        else:
            return audio.astype(np.float32), 0.0, 0.0

        pre = np.zeros(int(round(pre_s * self.sample_rate)), dtype=np.float32)
        post = np.zeros(int(round(post_s * self.sample_rate)), dtype=np.float32)
        return np.concatenate([pre, audio.astype(np.float32), post]).astype(np.float32), pre_s, post_s

    def _apply_qsb(self, audio: np.ndarray, depth_range: tuple[float, float] = (0.18, 0.55)) -> tuple[np.ndarray, float]:
        if audio.size == 0:
            return audio.astype(np.float32), 0.0

        depth = float(np.random.uniform(*depth_range))
        t = np.arange(audio.size, dtype=np.float32) / float(self.sample_rate)
        slow = 0.5 * (1.0 + np.sin(2 * np.pi * np.random.uniform(0.08, 0.8) * t + np.random.uniform(0.0, np.pi)))
        fast = 0.5 * (1.0 + np.sin(2 * np.pi * np.random.uniform(0.8, 3.5) * t + np.random.uniform(0.0, np.pi)))
        envelope = 1.0 - depth * (0.65 * slow + 0.35 * fast)
        envelope = np.clip(envelope, 0.18, 1.25).astype(np.float32)
        return (audio * envelope).astype(np.float32), depth

    def _apply_channel_response(self, audio: np.ndarray, strong: bool = False) -> tuple[np.ndarray, float, float]:
        if audio.size == 0:
            return audio.astype(np.float32), 0.0, 0.0

        nyquist = 0.5 * float(self.sample_rate)
        low_hz = float(np.random.uniform(250.0, 520.0 if strong else 420.0))
        high_hz = float(np.random.uniform(1300.0 if strong else 1500.0, min(3200.0, nyquist * 0.95)))
        if not (0.0 < low_hz < high_hz < nyquist):
            return audio.astype(np.float32), 0.0, 0.0

        sos = scipy.signal.butter(4, [low_hz / nyquist, high_hz / nyquist], btype="band", output="sos")
        filtered = scipy.signal.sosfiltfilt(sos, audio).astype(np.float32)
        return filtered, low_hz, high_hz

    def _apply_short_reverb(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        if audio.size == 0:
            return audio.astype(np.float32), 0.0

        tap_count = int(np.random.randint(2, 5))
        delays_ms = np.random.uniform(10.0, 75.0, tap_count)
        gains = np.random.uniform(0.08, 0.35, tap_count)
        max_delay = int(round(np.max(delays_ms) * self.sample_rate / 1000.0))
        ir = np.zeros(max_delay + 1, dtype=np.float32)
        ir[0] = 1.0
        for delay_ms, gain in zip(delays_ms, gains):
            delay = int(round(delay_ms * self.sample_rate / 1000.0))
            ir[delay] += float(gain)

        wet = scipy.signal.fftconvolve(audio, ir, mode="full")[: audio.size].astype(np.float32)
        wet_peak = float(np.max(np.abs(wet))) if wet.size else 0.0
        if wet_peak > 1e-9:
            wet = wet / wet_peak
        mix = float(np.random.uniform(0.08, 0.24))
        blended = (1.0 - mix) * audio + mix * wet
        return blended.astype(np.float32), mix

    def _apply_codec_artifacts(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        if audio.size == 0:
            return audio.astype(np.float32), 0

        codec_rate = int(np.random.choice([8000, 11025, 12000]))
        degraded = scipy.signal.resample_poly(audio, codec_rate, self.sample_rate).astype(np.float32)
        degraded = scipy.signal.resample_poly(degraded, self.sample_rate, codec_rate).astype(np.float32)
        if degraded.size < audio.size:
            degraded = np.pad(degraded, (0, audio.size - degraded.size))
        degraded = degraded[: audio.size]
        quantized = np.round(np.clip(degraded, -1.0, 1.0) * 2048.0) / 2048.0
        return quantized.astype(np.float32), codec_rate

    def _apply_random_dropouts(self, audio: np.ndarray, max_count: int = 3) -> tuple[np.ndarray, int]:
        if audio.size == 0:
            return audio.astype(np.float32), 0

        count = int(np.random.randint(0, max_count + 1))
        if count == 0:
            return audio.astype(np.float32), 0

        result = audio.astype(np.float32).copy()
        for _ in range(count):
            start = int(np.random.randint(0, max(1, audio.size - 1)))
            length = int(np.random.randint(max(4, self.sample_rate // 400), max(8, self.sample_rate // 20)))
            end = min(audio.size, start + length)
            attenuation = float(np.random.uniform(0.15, 0.65))
            result[start:end] *= attenuation
        return result, count

    def _apply_clicks(self, audio: np.ndarray, max_count: int = 4) -> tuple[np.ndarray, int]:
        if audio.size == 0:
            return audio.astype(np.float32), 0

        count = int(np.random.randint(0, max_count + 1))
        if count == 0:
            return audio.astype(np.float32), 0

        clicks = np.zeros_like(audio, dtype=np.float32)
        for _ in range(count):
            start = int(np.random.randint(0, max(1, audio.size - 1)))
            width = int(np.random.randint(2, max(3, self.sample_rate // 250)))
            end = min(audio.size, start + width)
            win = np.hanning(max(2, end - start)).astype(np.float32)
            amp = float(np.random.uniform(0.08, 0.45) * np.random.choice([-1.0, 1.0]))
            clicks[start:end] += amp * win[: end - start]
        return (audio + clicks).astype(np.float32), count

    def _apply_companding(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        if audio.size == 0:
            return audio.astype(np.float32), 0.0

        drive = float(np.random.uniform(1.1, 2.5))
        compressed = np.tanh(audio * drive) / np.tanh(drive)
        return compressed.astype(np.float32), drive

    def generate_audio(self, text: str, snr_db: float | None = None, noise_type: str = "white",
                       normalize_peak: bool = True) -> np.ndarray:
        """
        noise_type:
        'white' | 'hum' | 'interference' | 'pink' | 'brown' |
        'room' | 'mic' | 'qrm' | 'realistic' | 'realistic_mic'
        """
        audio_segments: list[np.ndarray] = []

        words = str(text).split(' ')
        for w_idx, word in enumerate(words):
            for c_idx, ch in enumerate(word):
                code = self.CODE.get(ch.upper())
                if not code:
                    continue

                for s_idx, sym in enumerate(code):
                    if sym == '.':
                        audio_segments.append(self.generate_tone(self.dot_len))
                    elif sym == '-':
                        audio_segments.append(self.generate_tone(self.dash_len))

                    if s_idx < len(code) - 1:
                        audio_segments.append(self.generate_silence(self.intra_char_space))

                if c_idx < len(word) - 1:
                    audio_segments.append(self.generate_silence(self.inter_char_space))

            if w_idx < len(words) - 1:
                audio_segments.append(self.generate_silence(self.word_space))

        if not audio_segments:
            return np.array([], dtype=np.float32)

        final_audio = np.concatenate(audio_segments).astype(np.float32)
        noise_key = str(noise_type or "white").lower()
        metadata: dict[str, float | int | str] = {
            "noise_type": noise_key,
            "noise_components": "",
            "channel_profile": "clean",
            "pre_silence_s": 0.0,
            "post_silence_s": 0.0,
            "channel_low_hz": 0.0,
            "channel_high_hz": 0.0,
            "reverb_mix": 0.0,
            "codec_rate": 0,
            "qsb_depth": 0.0,
            "dropout_count": 0,
            "click_count": 0,
            "companding_drive": 0.0,
        }

        final_audio, pre_s, post_s = self._add_context_padding(final_audio, noise_key)
        metadata["pre_silence_s"] = pre_s
        metadata["post_silence_s"] = post_s

        if noise_key in self.REALISTIC_NOISE_TYPES or noise_key == "interference":
            final_audio, qsb_depth = self._apply_qsb(final_audio)
            metadata["qsb_depth"] = qsb_depth

        if noise_key in {"room", "mic", "qrm", "realistic", "realistic_mic"}:
            strong_channel = noise_key in {"mic", "qrm", "realistic_mic"}
            final_audio, low_hz, high_hz = self._apply_channel_response(final_audio, strong=strong_channel)
            metadata["channel_profile"] = "speaker_mic_chain" if strong_channel else "room_capture"
            metadata["channel_low_hz"] = low_hz
            metadata["channel_high_hz"] = high_hz

        if noise_key in {"room", "mic", "realistic", "realistic_mic"}:
            final_audio, reverb_mix = self._apply_short_reverb(final_audio)
            metadata["reverb_mix"] = reverb_mix

        if noise_key in {"mic", "qrm", "realistic_mic"}:
            final_audio, codec_rate = self._apply_codec_artifacts(final_audio)
            metadata["codec_rate"] = codec_rate

        if noise_key in {"qrm", "realistic", "realistic_mic"}:
            final_audio, dropout_count = self._apply_random_dropouts(final_audio)
            final_audio, click_count = self._apply_clicks(final_audio)
            metadata["dropout_count"] = dropout_count
            metadata["click_count"] = click_count

        if snr_db is not None:
            final_audio = self.inject_noise(final_audio, float(snr_db), noise_type=noise_type)
            metadata["noise_components"] = "+".join(self.last_noise_components)

        if noise_key in {"mic", "qrm", "realistic_mic"}:
            final_audio, companding_drive = self._apply_companding(final_audio)
            metadata["companding_drive"] = companding_drive

        if normalize_peak:
            peak = float(np.max(np.abs(final_audio))) if final_audio.size else 0.0
            if peak > 0:
                final_audio = (final_audio / peak * 0.9).astype(np.float32)

        self.last_render_metadata = metadata
        return final_audio

    def inject_noise(self, audio: np.ndarray, snr_db: float, noise_type: str = 'white',
                     active_eps: float = 1e-4) -> np.ndarray:
        """
        FIX: SNR computed on ACTIVE samples only (non-silence), otherwise SNR becomes much worse than intended.
        """
        if audio.size == 0:
            return audio

        active = np.abs(audio) > active_eps
        if not np.any(active):
            return audio

        signal_power = float(np.mean(audio[active] ** 2))
        if signal_power <= 0.0:
            return audio

        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        target_noise_std = float(np.sqrt(noise_power))

        n = len(audio)
        t = np.linspace(0, n / self.sample_rate, n, endpoint=False, dtype=np.float32)
        component_kinds = self._resolve_noise_components(str(noise_type or "white").lower())
        self.last_noise_components = component_kinds

        weights = np.random.dirichlet(np.ones(len(component_kinds), dtype=np.float64)).astype(np.float32)
        noise = np.zeros(n, dtype=np.float32)
        for weight, kind in zip(weights, component_kinds):
            component = self._build_noise_component(kind, n, t)
            noise += float(weight) * component

        noise = self._normalize_std(noise) * target_noise_std
        return (audio + noise).astype(np.float32)

    def save(self, audio: np.ndarray, filename: str):
        output_path = Path(filename).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.sample_rate)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="HELLO WORLD")
    parser.add_argument("--out", type=str, default="test_morse.wav")
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument(
        "--noise_type",
        type=str,
        default="white",
        choices=["white", "hum", "interference", "pink", "brown", "room", "mic", "qrm", "realistic", "realistic_mic"],
    )
    args = parser.parse_args()

    gen = MorseGenerator(wpm=20, farnsworth_wpm=15)
    audio = gen.generate_audio(args.text, snr_db=args.snr, noise_type=args.noise_type)
    gen.save(audio, args.out)
    print(f"Saved to {args.out}")
