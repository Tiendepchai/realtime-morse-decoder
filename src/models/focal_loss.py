import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss for dynamic difficulty weighting.
    Focuses learning on hard-to-recognize characters (distorted dots/dashes).
    
    Reference: Focal Loss for Dense Object Detection (Lin et al., 2017)
    Adapted for CTC by modifying the weighting scheme.
    """
    def __init__(self, blank=0, gamma=2.0, alpha=0.25, reduction='mean', zero_infinity=True):
        super(FocalCTCLoss, self).__init__()
        self.blank = blank
        self.gamma = gamma  # Focusing parameter (typically 2.0)
        self.alpha = alpha  # Balance parameter
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=zero_infinity)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: (Time, Batch, Classes) - log probabilities from model
            targets: (sum(target_lengths),) - concatenated target sequences
            input_lengths: (Batch,) - lengths of input sequences
            target_lengths: (Batch,) - lengths of target sequences
        """
        # Compute standard CTC loss per sample (reduction='none')
        ctc_losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Apply Focal weighting: (1 - exp(-loss))^gamma * loss
        # Higher loss (harder samples) get more weight
        # This is equivalent to focusing on difficult examples
        p_t = torch.exp(-ctc_losses)  # Probability of correct sequence
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha balancing if needed
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight
            
        # Weighted loss
        focal_ctc_loss = focal_weight * ctc_losses
        
        # Reduction
        if self.reduction == 'mean':
            return focal_ctc_loss.mean()
        elif self.reduction == 'sum':
            return focal_ctc_loss.sum()
        else:
            return focal_ctc_loss


if __name__ == "__main__":
    # Test
    focal_loss = FocalCTCLoss(gamma=2.0)
    
    # Dummy data
    T, N, C = 50, 2, 10
    log_probs = torch.randn(T, N, C).log_softmax(2)
    targets = torch.randint(1, C, (20,))
    input_lengths = torch.full((N,), T, dtype=torch.long)
    target_lengths = torch.tensor([10, 10])
    
    loss = focal_loss(log_probs, targets, input_lengths, target_lengths)
    print(f"Focal CTC Loss: {loss.item():.4f}")
