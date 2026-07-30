
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.crnn import CRNN

def test_model_architecture():
    print("Testing CRNN Architecture...")
    
    n_mels = 64
    num_classes = 40 # A-Z, 0-9, space, punctuation, blank
    hidden_size = 128
    
    model = CRNN(num_classes=num_classes, n_mels=n_mels, hidden_size=hidden_size)
    
    # Create dummy input: Batch=4, Channel=1, Freq=64, Time=500 frames
    batch_size = 4
    time_steps = 500
    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    
    print(f"Input Shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Output Shape: {output.shape}")
    
    # Validation
    # Output should be (Batch, Time', NumClasses)
    # Time' should be Time / 4 due to 2 pooling layers
    expected_time = time_steps // 4
    
    assert output.shape[0] == batch_size, f"Expected batch {batch_size}, got {output.shape[0]}"
    assert output.shape[1] == expected_time, f"Expected time {expected_time}, got {output.shape[1]}"
    assert output.shape[2] == num_classes, f"Expected classes {num_classes}, got {output.shape[2]}"
    
    print("CRNN Architecture verification PASSED!")


def test_crnn_output_lengths():
    model = CRNN(num_classes=40, n_mels=64)
    input_lengths = torch.tensor([1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17])
    output_lengths = model.get_output_lengths(input_lengths)
    assert torch.equal(output_lengths, torch.tensor([1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4]))


def test_crnn_packed_lstm_ignores_batch_padding():
    torch.manual_seed(0)

    model = CRNN(num_classes=40, n_mels=64)
    model.eval()

    # Sample 0 is long, sample 1 is short and zero-padded only for batching.
    batch = torch.randn(2, 1, 64, 120)
    batch[1, :, :, 60:] = 0.0
    input_lengths = torch.tensor([120, 60])

    with torch.no_grad():
        batched_output = model(batch, input_lengths=input_lengths)

    sample_1 = batch[1:2, :, :, :60]
    with torch.no_grad():
        standalone_output = model(sample_1, input_lengths=torch.tensor([60]))

    valid_steps = standalone_output.shape[1]
    assert valid_steps == model.get_output_lengths(torch.tensor([60])).item()
    diff = (batched_output[1:2, :valid_steps] - standalone_output).abs()
    # Packing removes the recurrent leakage from padded tail; any remaining
    # difference should stay small and mostly near the CNN boundary.
    assert float(diff.mean()) < 3e-4
    assert torch.allclose(
        batched_output[1:2, : valid_steps - 2],
        standalone_output[:, : valid_steps - 2],
        atol=1.5e-3,
        rtol=1e-4,
    )

if __name__ == "__main__":
    test_model_architecture()
