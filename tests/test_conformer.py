import torch

from src.models.conformer import Conformer, infer_conformer_time_reduction_factor


def test_conformer_padding_mask_keeps_valid_prefix_stable():
    torch.manual_seed(0)
    model = Conformer(
        num_classes=8,
        input_dim=64,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    ).eval()

    valid_frames = 48
    padded_frames = 24
    base = torch.randn(1, 1, 64, valid_frames)
    padded = torch.cat([base, torch.zeros(1, 1, 64, padded_frames)], dim=3)

    with torch.no_grad():
        base_logits = model(base, input_lengths=torch.tensor([valid_frames]))
        padded_logits = model(padded, input_lengths=torch.tensor([valid_frames]))

    mean_abs_diff = (base_logits - padded_logits[:, :valid_frames]).abs().mean().item()
    assert mean_abs_diff < 5e-3


def test_conformer_forward_without_lengths_still_runs():
    model = Conformer(
        num_classes=8,
        input_dim=64,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    ).eval()
    dummy = torch.randn(2, 1, 64, 40)

    with torch.no_grad():
        output = model(dummy)

    assert output.shape == (2, 40, 8)


def test_conformer_time_subsampling_reduces_sequence_length():
    model = Conformer(
        num_classes=8,
        input_dim=64,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        time_reduction_factor=4,
    ).eval()
    dummy = torch.randn(2, 1, 64, 41)
    input_lengths = torch.tensor([41, 17])

    with torch.no_grad():
        output = model(dummy, input_lengths=input_lengths)

    assert output.shape == (2, 11, 8)
    assert torch.equal(model.get_output_lengths(input_lengths), torch.tensor([11, 5]))


def test_infer_conformer_time_reduction_factor_from_state_dict():
    model = Conformer(
        num_classes=8,
        input_dim=64,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        time_reduction_factor=4,
    )

    factor = infer_conformer_time_reduction_factor(model.state_dict())
    assert factor == 4
