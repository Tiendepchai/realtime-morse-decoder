from __future__ import annotations

import torch

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


def mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps is not None
        and torch.backends.mps.is_available()
    )


def cuda_is_available() -> bool:
    return bool(torch.cuda.is_available())


def resolve_torch_device(requested_device: str = "auto") -> torch.device:
    requested = str(requested_device).lower().strip()

    if requested == "auto":
        if cuda_is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not cuda_is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")

    if requested == "mps":
        if not mps_is_available():
            raise RuntimeError("MPS was requested, but no MPS device is available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {requested_device}")


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def supports_amp(device: torch.device) -> bool:
    return device.type == "cuda"

