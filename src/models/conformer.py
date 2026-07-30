import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _lengths_to_padding_mask(lengths: torch.Tensor | None, max_len: int) -> torch.Tensor | None:
    if lengths is None:
        return None
    time_index = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return time_index >= lengths.unsqueeze(1)


def _zero_masked_frames(x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return x
    return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def _reduce_lengths_by_stride2(lengths: torch.Tensor, repeats: int) -> torch.Tensor:
    reduced = lengths.clone()
    for _ in range(repeats):
        reduced = torch.div(reduced + 1, 2, rounding_mode="floor")
    return reduced.clamp(min=1)


def infer_conformer_time_reduction_factor(state_dict: dict[str, torch.Tensor]) -> int:
    conv_weight_keys = [
        key
        for key in state_dict
        if key.startswith("time_subsampler.layers.") and key.endswith(".weight")
    ]
    num_layers = len(conv_weight_keys)
    if num_layers == 0:
        return 1
    return 2 ** num_layers


class TemporalConvSubsampler(nn.Module):
    def __init__(self, d_model: int, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("TemporalConvSubsampler requires at least one stride-2 layer")
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.silu(layer(x))
        return x

class ConformerBlock(nn.Module):
    """
    Conformer block: FeedForward -> MultiHeadSelfAttention -> Convolution -> FeedForward
    Reference: Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020)
    """
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, conv_kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()
        
        # Feed Forward Module 1 (Macaron-style)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-Head Self-Attention Module
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Convolution Module
        self.conv_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, 
                                        padding=(conv_kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed Forward Module 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.final_layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, padding_mask: torch.Tensor | None = None):
        # x: (Batch, Time, d_model)
        x = _zero_masked_frames(x, padding_mask)
        
        # Feed Forward 1 (half-step residual)
        x = x + 0.5 * self.ff1(x)
        x = _zero_masked_frames(x, padding_mask)
        
        # Multi-Head Self-Attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x_attn, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout1(x_attn)
        x = _zero_masked_frames(x, padding_mask)
        
        # Convolution Module
        residual = x
        x = self.conv_layer_norm(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # GLU activation
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)  # Swish
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = residual + self.dropout2(x)
        x = _zero_masked_frames(x, padding_mask)
        
        # Feed Forward 2 (half-step residual)
        x = x + 0.5 * self.ff2(x)
        x = _zero_masked_frames(x, padding_mask)
        
        # Final layer norm
        x = self.final_layer_norm(x)
        x = _zero_masked_frames(x, padding_mask)
        
        return x


class Conformer(nn.Module):
    """
    Conformer Encoder for CTC-based Morse code recognition.
    Designed for high accuracy as per thesis section 4.2.2.
    """
    def __init__(
        self,
        num_classes,
        input_dim=64,
        d_model=256,
        nhead=4,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        blank_bias: float = 0.0,
        time_reduction_factor: int = 1,
    ):
        super(Conformer, self).__init__()

        if time_reduction_factor < 1 or time_reduction_factor & (time_reduction_factor - 1):
            raise ValueError("time_reduction_factor must be a power of two >= 1")

        self.input_dim = input_dim
        self.d_model = d_model
        self.time_reduction_factor = time_reduction_factor
        self.num_subsampling_layers = int(math.log2(time_reduction_factor))

        # Input projection (from Mel features to d_model)
        self.input_projection = nn.Linear(input_dim, d_model)

        self.time_subsampler = None
        if self.num_subsampling_layers > 0:
            self.time_subsampler = TemporalConvSubsampler(
                d_model=d_model,
                num_layers=self.num_subsampling_layers,
            )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Stack of Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, nhead, dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, num_classes)

        # Blank bias init (index 0 = blank)
        if blank_bias != 0.0:
            with torch.no_grad():
                self.output_projection.bias[0] = float(blank_bias)
        
    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        if self.num_subsampling_layers == 0:
            return input_lengths.clone()
        return _reduce_lengths_by_stride2(input_lengths, self.num_subsampling_layers)

    def forward(self, x, input_lengths=None):
        """
        Args:
            x: Input tensor (Batch, Channel, Freq, Time) or (Batch, Time, Freq)
            input_lengths: Optional, not used in processing but kept for API compatibility
        Returns:
            out: (Batch, Time, NumClasses) — raw logits for CTC Loss
        """
        # Handle (B, 1, F, T) -> (B, T, F)
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, F, T)
            x = x.permute(0, 2, 1)  # (B, T, F)

        padding_mask = _lengths_to_padding_mask(input_lengths, x.size(1))
        
        # Input projection
        x = self.input_projection(x)  # (B, T, d_model)
        x = _zero_masked_frames(x, padding_mask)

        if self.time_subsampler is not None:
            x = x.transpose(1, 2)
            x = self.time_subsampler(x)
            x = x.transpose(1, 2)
            if input_lengths is not None:
                input_lengths = self.get_output_lengths(input_lengths)
            padding_mask = _lengths_to_padding_mask(input_lengths, x.size(1))
            x = _zero_masked_frames(x, padding_mask)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = _zero_masked_frames(x, padding_mask)
        
        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, padding_mask=padding_mask)
        
        # Output projection — raw logits (log_softmax applied in training loop)
        x = self.output_projection(x)  # (B, T, NumClasses)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based models.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test
    model = Conformer(num_classes=38, input_dim=64, d_model=512, num_layers=8)
    dummy = torch.randn(2, 1, 64, 200)  # (B, C, F, T)
    out = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {out.shape}")  # Should be (2, 200, 38)
