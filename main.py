
import sys
import os

# Add src to path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.morse_generator import MorseGenerator

def main():
    print("Initializing Morse Decoder Project Skeleton...")
    
    # Example: Generate a sample file
    text = "SOS CQ TEST"
    print(f"Generating sample audio for: '{text}'")
    
    output_dir = "data/samples"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "sample_start.wav")
    
    gen = MorseGenerator(wpm=25, farnsworth_wpm=18)
    audio = gen.generate_audio(text, snr_db=8, noise_type="realistic_mic")
    gen.save(audio, output_file)
    
    print(f"Sample generated at: {output_file}")
    print("\nSkeleton structure ready:")
    print("- src/data: Data generation and loading")
    print("- src/features: Feature extraction (spectrograms, etc.)")
    print("- src/models: Deep Learning models")
    print("- src/app: Application logic (API/UI)")

if __name__ == "__main__":
    main()
