import torch
import torch.nn as nn
import torch.optim as optim
from src.models.conformer import Conformer
from src.models.crnn import CRNN
from src.train import _compute_output_lengths, _forward_log_probs

def debug_ctc_sanity():
    print("=== CTC Training Sanity Check (10 steps, CRNN + Conformer) ===")
    
    batch_size = 2
    time = 200
    freq = 64
    num_classes = 38
    
    torch.manual_seed(42)
    data = torch.randn(batch_size, 1, freq, time)
    input_lengths = torch.tensor([time, time // 2], dtype=torch.long)
    targets = torch.randint(1, num_classes, (batch_size, 10))
    target_lengths = torch.tensor([10, 8], dtype=torch.long)
    
    # All models use same blank_bias and same training path
    test_cases = [
        ("CRNN", CRNN(num_classes=num_classes, n_mels=freq, blank_bias=2.0)),
        ("Conformer", Conformer(num_classes=num_classes, input_dim=freq, blank_bias=2.0)),
    ]
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    for name, model in test_cases:
        print(f"\n--- Training {name} ---")
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
        losses = []
        for step in range(1, 11):
            optimizer.zero_grad()
            out_lens = _compute_output_lengths(model, input_lengths, "cpu")
            # All models use unified forward(x, input_lengths)
            log_probs = _forward_log_probs(model, data, input_lengths)
            log_probs_tbc = log_probs.permute(1, 0, 2)
            
            loss = criterion(log_probs_tbc, targets, out_lens, target_lengths)
            losses.append(loss.item())
            
            if not torch.isfinite(loss):
                print(f"  [FAIL] Loss NaN/Inf at step {step}!")
                break
            
            loss.backward()
            optimizer.step()
        
        print(f"  Step 1:  Loss={losses[0]:.4f}")
        print(f"  Step 10: Loss={losses[-1]:.4f}")
        improved = losses[-1] < losses[0]
        print(f"  {'[PASS]' if improved else '[WARN]'} Loss {'decreased' if improved else 'did not decrease'}")

if __name__ == "__main__":
    debug_ctc_sanity()
