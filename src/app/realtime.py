import argparse
import time
import sys
import queue
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

import os
# Add src to path if running directly from src/app
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.features.audio_processor import AudioProcessor
from src.utils.device import DEVICE_CHOICES, resolve_torch_device, synchronize_device
from src.utils.text import CHARS, TextTransform
from src.inference import _load_model_for_inference

class RealtimeDecoder:
    def __init__(self, model_path, model_type="conformer",
                 sample_rate=16000, buffer_duration=3.0, overlap_duration=1.0, 
                 silence_threshold=0.01, device_name="auto"):
        self.device = resolve_torch_device(device_name)
        print(f"Using device: {self.device}")
        
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.overlap_duration = overlap_duration
        self.silence_threshold = silence_threshold
        
        # Audio capturing state
        self.q = queue.Queue()
        self.buffer_size = int(buffer_duration * sample_rate)
        self.overlap_size = int(overlap_duration * sample_rate)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        # Models and processing Setup
        self.processor = AudioProcessor(sample_rate=sample_rate, n_mels=64)
        vocab_size = len(CHARS)
        self.model = _load_model_for_inference(model_path, model_type, vocab_size, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.text_transform = TextTransform()
        
        print(f"Loaded {model_type} model from {model_path}")
        print(f"Buffer duration: {buffer_duration}s, Overlap: {overlap_duration}s")
        
    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Put raw audio chunk into the queue
        self.q.put(indata.copy())
        
    def _predict(self, audio_chunk):
        """Runs the model on a NumPy array of audio data."""
        # 1. Check for silence heuristic
        rms = np.sqrt(np.mean(audio_chunk**2))
        if rms < self.silence_threshold:
            return "" # Skip inference if it's just background noise
            
        # 2. Extract features
        cleaned = self.processor.clean_audio(audio_chunk)
        log_mel = self.processor.compute_log_mel(cleaned)
        features = self.processor.apply_cmvn(log_mel)
        features = np.clip(features, -10.0, 10.0)
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        features_tensor = features_tensor.permute(0, 1, 3, 2).to(self.device)  # (1, 1, F, T)
        
        # 3. Model Forward
        with torch.no_grad():
            input_lengths = torch.tensor([features_tensor.shape[3]], dtype=torch.long).to(self.device)
            logits = self.model(features_tensor, input_lengths)
            output = F.log_softmax(logits, dim=2)
            output = output.permute(1, 0, 2)  # (T, B, C)
        synchronize_device(self.device)
            
        # 4. Greedy Decode
        arg_maxes = torch.argmax(output, dim=2)
        decode = []
        prev_idx = 0
        for idx in arg_maxes[:, 0]:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                decode.append(idx)
            prev_idx = idx
            
        result = self.text_transform.int_to_text(decode)
        return result

    def _merge_predictions(self, prev_text, new_text):
        """Basic heuristic to stitch text by finding overlapping suffix/prefix."""
        if not prev_text:
            return new_text
        if not new_text:
            return ""
            
        # Try to find a matching suffix in prev_text that is a prefix in new_text
        # We start looking from the largest possible overlap
        max_overlap = min(len(prev_text), len(new_text))
        for i in range(max_overlap, 0, -1):
            if prev_text[-i:] == new_text[:i]:
                # Found overlap, return the explicitly new part
                return new_text[i:]
                
        # If no overlap found (maybe due to decoding errors or no actual overlap in text),
        # return the new part separated by a space as a fallback.
        # But for continuous morse, often it might just append.
        return new_text

    def start(self):
        print("Starting realtime decoder... (Press Ctrl+C to stop)")
        print("Listening...")
        
        last_prediction = ""
        last_print_time = time.time()
        
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback):
                while True:
                    # Collect data from queue
                    if not self.q.empty():
                        new_data = self.q.get()
                        
                        # Shift buffer left and append new data
                        shift_amount = len(new_data)
                        self.audio_buffer = np.roll(self.audio_buffer, -shift_amount)
                        self.audio_buffer[-shift_amount:] = new_data[:, 0]
                        
                        # Only run inference periodically (e.g. every second)
                        current_time = time.time()
                        if current_time - last_print_time >= 1.0: # 1 second inference step
                            # Run inference on the current buffer
                            current_text = self._predict(self.audio_buffer)
                            
                            if current_text:
                                new_chars = self._merge_predictions(last_prediction, current_text)
                                if new_chars:
                                    print(new_chars, end='', flush=True)
                                    last_prediction = current_text
                                    
                            last_print_time = current_time
                            
                    else:
                        time.sleep(0.01) # Avoid busy loop
                        
        except KeyboardInterrupt:
            print("\nStopped realtime decoding.")
        except Exception as e:
            print(f"\nError initializing audio stream: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Morse Decoder using microphone")
    parser.add_argument("--model", type=str, default="experiments/checkpoints/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="conformer", choices=["crnn", "conformer"])
    parser.add_argument("--buffer", type=float, default=3.0, help="Duration of the audio buffer in seconds (sliding window)")
    parser.add_argument("--overlap", type=float, default=1.0, help="Duration of overlap in seconds")
    parser.add_argument("--threshold", type=float, default=0.01, help="RMS silence threshold to trigger inference")
    parser.add_argument("--device", type=str, default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--list_devices", action="store_true", help="List available audio devices and exit")
    
    args = parser.parse_args()
    
    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)
        
    decoder = RealtimeDecoder(
        model_path=args.model,
        model_type=args.model_type,
        buffer_duration=args.buffer,
        overlap_duration=args.overlap,
        silence_threshold=args.threshold,
        device_name=args.device,
    )
    
    decoder.start()
