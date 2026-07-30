import json

import numpy as np
import pytest

from demo_realtime_mic import (
    AdaptiveSilenceGate,
    ActivityDecision,
    BaseRealtimePredictor,
    PredictionOutput,
    RealtimeDebugRecorder,
    RealtimeMicDemo,
    build_parser,
    build_debug_paths,
    load_tuning_profile,
    predictions_are_compatible,
    should_accept_deep_prediction,
    should_accept_traditional_prediction,
    validate_args,
)


class DummyPredictor(BaseRealtimePredictor):
    method_name = "dummy"

    def predict(self, audio_chunk: np.ndarray) -> PredictionOutput:
        duration = float(audio_chunk.size) / 16000.0 if audio_chunk.size else 0.0
        return PredictionOutput(text=f"SEGMENT {duration:.2f}", meta={})


class PaddingDummyPredictor(BaseRealtimePredictor):
    method_name = "dummy"

    def __init__(self, pre_pad_samples: int, post_pad_samples: int):
        self.segment_pre_padding_samples = pre_pad_samples
        self.segment_post_padding_samples = post_pad_samples
        self.seen_lengths: list[int] = []

    def predict(self, audio_chunk: np.ndarray) -> PredictionOutput:
        self.seen_lengths.append(int(audio_chunk.size))
        return PredictionOutput(text=f"LEN {audio_chunk.size}", meta={})


def test_predictions_are_compatible_for_prefix_growth():
    assert predictions_are_compatible("SOS", "SOS CQ")
    assert predictions_are_compatible("SOS CQ", "SOS")
    assert not predictions_are_compatible("SOS", "TEST")


def test_should_accept_deep_prediction_rejects_low_confidence_noise():
    accepted, reason = should_accept_deep_prediction(
        text="A",
        emitted_frame_count=5,
        emitted_frame_ratio=0.03,
        mean_emitted_confidence=0.22,
        mean_blank_probability=0.80,
    )
    assert not accepted
    assert reason == "low_emitted_confidence"


def test_should_accept_traditional_prediction_rejects_sparse_tone_frames():
    accepted, reason = should_accept_traditional_prediction(
        text="SOS",
        meta={
            "is_failure": False,
            "num_frames": 300,
            "num_tone_frames": 2,
        },
    )
    assert not accepted
    assert reason == "too_few_tone_frames"


def test_adaptive_silence_gate_blocks_background_noise_then_allows_tone():
    gate = AdaptiveSilenceGate(base_threshold=0.01, warmup_steps=2)
    background = np.full(16000, 0.002, dtype=np.float32)

    first = gate.evaluate(background)
    second = gate.evaluate(background)
    third = gate.evaluate(background)

    assert not first.should_infer
    assert not second.should_infer
    assert not third.should_infer

    t = np.arange(16000, dtype=np.float32) / 16000.0
    tone = (0.05 * np.sin(2.0 * np.pi * 700.0 * t)).astype(np.float32)
    active = gate.evaluate(tone)

    assert active.should_infer
    assert active.reason == "active_signal"


def test_realtime_demo_requires_confirmation_before_committing_text():
    demo = RealtimeMicDemo(
        predictor=DummyPredictor(),
        sample_rate=16000,
        buffer_duration=3.0,
        step_duration=0.75,
        chunk_duration=0.25,
        silence_threshold=0.01,
        reset_silence_steps=2,
        confirm_steps=2,
        decode_mode="sliding",
        min_segment_duration=1.2,
        max_segment_duration=24.0,
        debug_recorder=None,
        input_device=None,
        show_full_predictions=False,
        show_meta=False,
    )

    assert demo._promote_stable_prediction("SOS") == ""
    assert demo._promote_stable_prediction("SOS") == "SOS"
    assert demo._promote_stable_prediction("SOS CQ") == "SOS CQ"


def test_segment_audio_is_capped_to_max_duration():
    demo = RealtimeMicDemo(
        predictor=DummyPredictor(),
        sample_rate=16000,
        buffer_duration=3.0,
        step_duration=0.75,
        chunk_duration=0.25,
        silence_threshold=0.01,
        reset_silence_steps=2,
        confirm_steps=2,
        decode_mode="segment",
        min_segment_duration=1.0,
        max_segment_duration=2.0,
        debug_recorder=None,
        input_device=None,
        show_full_predictions=False,
        show_meta=False,
    )

    demo._append_segment_audio(np.ones(16000, dtype=np.float32))
    demo._append_segment_audio(np.ones(20000, dtype=np.float32))
    segment = demo._consume_segment_audio()

    assert segment.size == 32000


def test_shutdown_finalizes_pending_segment():
    demo = RealtimeMicDemo(
        predictor=DummyPredictor(),
        sample_rate=16000,
        buffer_duration=3.0,
        step_duration=0.75,
        chunk_duration=0.25,
        silence_threshold=0.01,
        reset_silence_steps=2,
        confirm_steps=2,
        decode_mode="segment",
        min_segment_duration=1.0,
        max_segment_duration=4.0,
        debug_recorder=None,
        input_device=None,
        show_full_predictions=False,
        show_meta=False,
    )

    audio = np.ones(20000, dtype=np.float32) * 0.2
    demo._append_segment_audio(audio)
    demo.last_activity_decision = demo.activity_gate.evaluate(audio)
    demo._finalize_pending_segment_before_shutdown()

    assert demo.last_prediction.startswith("SEGMENT")
    assert demo.segment_sample_count == 0


def test_debug_paths_and_recorder_write_outputs(tmp_path):
    debug_paths = build_debug_paths(tmp_path / "debug_run")
    recorder = RealtimeDebugRecorder(sample_rate=16000, debug_dir=str(debug_paths.root_dir))

    chunk = np.ones(1600, dtype=np.float32) * 0.1
    recorder.append_session_audio(chunk)
    recorder.append_session_audio(chunk)
    recorder.save_segment(
        audio_chunk=np.ones(3200, dtype=np.float32) * 0.2,
        prediction_text="SOS",
        activity=AdaptiveSilenceGate(base_threshold=0.01).evaluate(np.ones(3200, dtype=np.float32) * 0.2),
        method_name="conformer",
    )
    session_wav, session_json = recorder.finalize(method_name="conformer", decode_mode="segment")

    assert session_wav is not None
    assert session_json is not None
    assert debug_paths.session_wav.exists()
    assert debug_paths.session_json.exists()
    assert any(debug_paths.segments_dir.glob("segment_001.wav"))


def test_load_tuning_profile_reads_best_result(tmp_path):
    profile_path = tmp_path / "tuning_result.json"
    profile_path.write_text(
        json.dumps(
            {
                "best": {
                    "model_type": "conformer",
                    "pre_pad_s": 0.25,
                    "post_pad_s": 0.15,
                    "speed": 0.9,
                    "band_low_hz": 850.0,
                    "band_high_hz": 1150.0,
                    "clip_threshold": 0.75,
                    "companding_exponent": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    profile = load_tuning_profile(str(profile_path))

    assert profile.model_type == "conformer"
    assert profile.pre_pad_s == pytest.approx(0.25)
    assert profile.band_low_hz == pytest.approx(850.0)
    assert profile.source_path == profile_path.resolve()


def test_load_tuning_profile_prefers_requested_model_profile(tmp_path):
    profile_path = tmp_path / "tuning_result.json"
    profile_path.write_text(
        json.dumps(
            {
                "best": {
                    "model_type": "conformer",
                    "pre_pad_s": 0.25,
                    "post_pad_s": 0.15,
                    "speed": 0.9,
                    "band_low_hz": 850.0,
                    "band_high_hz": 1150.0,
                    "clip_threshold": 0.75,
                    "companding_exponent": 1.0,
                },
                "best_by_model": {
                    "crnn": {
                        "model_type": "crnn",
                        "pre_pad_s": 0.0,
                        "post_pad_s": 0.25,
                        "speed": 1.0,
                        "band_low_hz": 700.0,
                        "band_high_hz": 1200.0,
                        "clip_threshold": 0.85,
                        "companding_exponent": 0.8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_tuning_profile(str(profile_path), preferred_method="crnn")

    assert profile.model_type == "crnn"
    assert profile.post_pad_s == pytest.approx(0.25)
    assert profile.clip_threshold == pytest.approx(0.85)


def test_validate_args_rejects_mismatched_tuning_method(tmp_path):
    profile_path = tmp_path / "tuning_result.json"
    profile_path.write_text(
        json.dumps(
            {
                "best": {
                    "model_type": "conformer",
                    "pre_pad_s": 0.25,
                    "post_pad_s": 0.25,
                    "speed": 0.9,
                    "band_low_hz": 850.0,
                    "band_high_hz": 1150.0,
                    "clip_threshold": 0.75,
                    "companding_exponent": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(["--method", "crnn", "--tuning-json", str(profile_path)])

    with pytest.raises(SystemExit):
        validate_args(args, parser)


def test_segment_decode_includes_predictor_padding_context():
    predictor = PaddingDummyPredictor(pre_pad_samples=2, post_pad_samples=1)
    demo = RealtimeMicDemo(
        predictor=predictor,
        sample_rate=10,
        buffer_duration=3.0,
        step_duration=0.75,
        chunk_duration=0.25,
        silence_threshold=0.01,
        reset_silence_steps=1,
        confirm_steps=1,
        decode_mode="segment",
        min_segment_duration=0.1,
        max_segment_duration=4.0,
        debug_recorder=None,
        input_device=None,
        show_full_predictions=False,
        show_meta=False,
    )

    demo._remember_recent_step_audio(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    active = ActivityDecision(
        should_infer=True,
        rms=0.2,
        activity_peak=0.2,
        noise_floor_rms=0.01,
        threshold_rms=0.02,
        threshold_peak=0.03,
        reason="active_signal",
    )
    silent = ActivityDecision(
        should_infer=False,
        rms=0.0,
        activity_peak=0.0,
        noise_floor_rms=0.01,
        threshold_rms=0.02,
        threshold_peak=0.03,
        reason="rms_below_threshold",
    )

    demo._run_segment_decode_step(active, np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32))
    demo._run_segment_decode_step(silent, np.array([3.0, 3.0, 3.0], dtype=np.float32))

    assert predictor.seen_lengths == [8]


def test_buffer_pre_context_uses_audio_buffer_before_current_step():
    predictor = PaddingDummyPredictor(pre_pad_samples=6, post_pad_samples=0)
    demo = RealtimeMicDemo(
        predictor=predictor,
        sample_rate=10,
        buffer_duration=3.0,
        step_duration=0.75,
        chunk_duration=0.25,
        silence_threshold=0.01,
        reset_silence_steps=1,
        confirm_steps=1,
        decode_mode="segment",
        min_segment_duration=0.1,
        max_segment_duration=4.0,
        debug_recorder=None,
        input_device=None,
        show_full_predictions=False,
        show_meta=False,
    )

    demo.audio_buffer = np.arange(30, dtype=np.float32)
    step_audio = np.array([25.0, 26.0, 27.0, 28.0, 29.0], dtype=np.float32)

    context = demo._take_buffer_pre_context(demo.audio_buffer.copy(), step_audio, max_samples=6)

    assert np.array_equal(context, np.array([19.0, 20.0, 21.0, 22.0, 23.0, 24.0], dtype=np.float32))
