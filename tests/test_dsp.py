
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.audio_processor import AudioProcessor

def test_dsp():
    processor = AudioProcessor(sample_rate=16000, n_mels=64)
    
    # Use the sample generated in previous step
    sample_path = "data/samples/sample_start.wav"
    if not os.path.exists(sample_path):
        print(f"Sample file not found: {sample_path}")
        return

    print(f"Processing {sample_path}...")
    
    # 1. Load
    audio = processor.load_audio(sample_path)
    print(f"Loaded audio shape: {audio.shape}")
    assert len(audio) > 0, "Audio load failed"

    # 2. Clean
    cleaned = processor.clean_audio(audio)
    print(f"Cleaned audio shape: {cleaned.shape}")
    
    # 3. Log-Mel
    log_mel = processor.compute_log_mel(cleaned)
    print(f"Log-Mel spect shape: {log_mel.shape} (frames, n_mels)")
    assert log_mel.shape[1] == 64, "Incorrect mel bins"
    
    # 4. CMVN
    norm_feat = processor.apply_cmvn(log_mel)
    print(f"Normalized features mean: {np.mean(norm_feat):.4f}, std: {np.std(norm_feat):.4f}")
    
    print("\nDSP Test Passed!")

if __name__ == "__main__":
    test_dsp()
