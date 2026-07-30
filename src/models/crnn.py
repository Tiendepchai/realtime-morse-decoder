import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CRNN(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels=1,
        n_mels=64,
        hidden_size=256,
        num_lstm_layers=3,
        cnn_channels=(64, 128, 256),
        blank_bias: float = 0.0,
    ):
        super(CRNN, self).__init__()

        self.n_mels = n_mels

        # Hardcoded pooling scheme:
        if len(cnn_channels) == 3:  # Upgraded
            # 64(P), 128(P), 256
            configs = [(cnn_channels[0], True), (cnn_channels[1], True), (cnn_channels[2], False)]
        elif len(cnn_channels) == 2:  # Legacy
            # 32(P), 64(P)
            configs = [(cnn_channels[0], True), (cnn_channels[1], True)]
        else:
            configs = [(c, True) for c in cnn_channels[:-1]] + [(cnn_channels[-1], False)]

        self.num_pools = sum(1 for _, p in configs if p)  # pools apply to BOTH freq and time

        layers = []
        in_c = input_channels
        for out_c, use_pool in configs:
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ])
            if use_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_c = out_c

        self.conv_layers = nn.Sequential(*layers)

        # RNN input dim: last_channels * (n_mels // 2**num_pools)
        rnn_input_dim = cnn_channels[-1] * (n_mels // (2 ** self.num_pools))

        self.rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # Blank bias init (index 0 = blank)
        if blank_bias != 0.0:
            with torch.no_grad():
                self.fc.bias[0] = float(blank_bias)

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # Pooling halves time each pool => / (2**num_pools)
        div = 2 ** self.num_pools
        return torch.div(input_lengths, div, rounding_mode="floor").clamp(min=1)

    def forward(self, x, input_lengths=None):
        """
        Args:
            x: (B, 1, n_mels, T)
            input_lengths: Optional frame lengths before CNN pooling.
        Returns:
            logits: (B, T', C)  raw logits
        """
        x = self.conv_layers(x)  # (B, C, F, T')
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F)

        if input_lengths is not None:
            if not torch.is_tensor(input_lengths):
                input_lengths = torch.tensor(input_lengths, device=x.device)
            output_lengths = self.get_output_lengths(input_lengths.to(x.device))
            output_lengths = output_lengths.clamp(min=1, max=t)
            packed = pack_padded_sequence(
                x,
                output_lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed, _ = self.rnn(packed)
            x, _ = pad_packed_sequence(packed, batch_first=True, total_length=t)
        else:
            x, _ = self.rnn(x)  # (B, T', 2H)
        logits = self.fc(x)  # (B, T', C)
        return logits


if __name__ == "__main__":
    model = CRNN(num_classes=40, n_mels=64)
    dummy = torch.randn(2, 1, 64, 200)
    out = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output logits: {out.shape}")
