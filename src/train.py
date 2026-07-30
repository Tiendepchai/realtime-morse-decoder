import os
import sys
import argparse
import datetime
import random
import time
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import editdistance

from src.features.audio_processor import AudioProcessor
from src.models.crnn import CRNN
from src.models.conformer import Conformer
from src.models.focal_loss import FocalCTCLoss
from src.utils.device import DEVICE_CHOICES, resolve_torch_device, supports_amp
from src.utils.training_artifacts import (
    EpochMetrics,
    create_tensorboard_writer,
    log_epoch_to_tensorboard,
    log_run_config_to_tensorboard,
    save_training_plots,
    write_metrics_csv,
    write_metrics_json,
)


from src.utils.text import CHARS, CHAR2IDX, IDX2CHAR, TextTransform, greedy_decoder


class MorseDataset(Dataset):
    def __init__(
        self,
        csv_file,
        audio_dir,
        sample_rate=16000,
        n_mels=64,
        augment=False,
        clamp_features=True,
        # AudioProcessor params
        n_fft=512,
        hop_length=160,
        low_cut=400,
        high_cut=1200,
        filter_order=4,
        augment_noise_prob=0.65,
        augment_snr_min_db=6.0,
        augment_snr_max_db=24.0,
        # NEW: dynamic effective hop via time downsample AFTER mel+cmvn
        dynamic_hop=False,
        hop_factors=(1, 2),
        hop_probs=(0.5, 0.5),
        hop_mode="avgpool",  # "avgpool" or "stride"
    ):
        self.csv_file = Path(csv_file).expanduser().resolve()
        self.csv_dir = self.csv_file.parent
        self.split_name = self.csv_file.stem
        self.data = pd.read_csv(self.csv_file)
        self.audio_dir = str(audio_dir or "").strip()
        self.audio_root = Path(self.audio_dir).expanduser().resolve() if self.audio_dir else None
        self.processor = AudioProcessor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            low_cut=low_cut,
            high_cut=high_cut,
            filter_order=filter_order,
            augment_noise_prob=augment_noise_prob,
            augment_snr_min_db=augment_snr_min_db,
            augment_snr_max_db=augment_snr_max_db,
        )
        self.text_transform = TextTransform()
        self.augment = augment
        self.clamp_features = clamp_features

        self.dynamic_hop = dynamic_hop
        self.hop_factors = list(hop_factors)
        self.hop_probs = list(hop_probs)
        self.hop_mode = hop_mode

    def __len__(self):
        return len(self.data)

    def _ensure_path_context(self):
        csv_file = getattr(self, "csv_file", None)
        if csv_file is not None and not isinstance(csv_file, Path):
            csv_file = Path(csv_file)
            self.csv_file = csv_file

        if getattr(self, "csv_file", None) is not None:
            resolved_csv_file = Path(self.csv_file).expanduser().resolve()
            self.csv_file = resolved_csv_file
            if not hasattr(self, "csv_dir"):
                self.csv_dir = resolved_csv_file.parent
            if not hasattr(self, "split_name"):
                self.split_name = resolved_csv_file.stem

        if not hasattr(self, "csv_dir"):
            self.csv_dir = Path.cwd()
        if not hasattr(self, "split_name"):
            self.split_name = ""
        if not hasattr(self, "audio_root"):
            audio_dir = str(getattr(self, "audio_dir", "") or "").strip()
            self.audio_root = Path(audio_dir).expanduser().resolve() if audio_dir else None

    def _resolve_audio_path(self, raw_path: str) -> str:
        self._ensure_path_context()
        raw = str(raw_path or "").strip()
        if not raw:
            raise FileNotFoundError(f"Empty audio path found in {self.csv_file}")

        raw_path_obj = Path(raw).expanduser()
        candidates: list[Path] = []

        def add_candidate(path: Path | None):
            if path is None:
                return
            try:
                candidate = path.expanduser()
            except Exception:
                candidate = path
            if candidate not in candidates:
                candidates.append(candidate)

        add_candidate(raw_path_obj)
        if not raw_path_obj.is_absolute():
            add_candidate((Path.cwd() / raw_path_obj).resolve())
            add_candidate((self.csv_dir / raw_path_obj).resolve())

        basename = raw_path_obj.name
        parent_name = raw_path_obj.parent.name if raw_path_obj.parent and raw_path_obj.parent.name not in {"", "."} else ""

        search_roots = [self.csv_dir]
        if self.audio_root is not None:
            search_roots.insert(0, self.audio_root)

        for root in search_roots:
            add_candidate(root / basename)
            add_candidate(root / self.split_name / basename)
            if parent_name:
                add_candidate(root / parent_name / basename)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        preview = ", ".join(str(path) for path in candidates[:6])
        raise FileNotFoundError(f"Could not resolve audio path '{raw}'. Tried: {preview}")

    def _downsample_time(self, features_np: np.ndarray, factor: int) -> np.ndarray:
        # features_np: (T, F)
        if factor <= 1 or features_np.shape[0] <= factor:
            return features_np

        if self.hop_mode == "stride":
            return features_np[::factor, :]

        # default avgpool (smooth, less aliasing)
        x = torch.from_numpy(features_np).float().T.unsqueeze(0)  # (1, F, T)
        x = F.avg_pool1d(x, kernel_size=factor, stride=factor, ceil_mode=False)
        return x.squeeze(0).T.numpy()  # (T', F)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = self._resolve_audio_path(row["path"])
        text = str(row["text"])

        audio = self.processor.load_audio(file_path)
        cleaned = self.processor.clean_audio(audio)

        if self.augment:
            cleaned = self.processor.augment_audio(cleaned)

        log_mel = self.processor.compute_log_mel(cleaned)      # (T, F)
        features = self.processor.apply_cmvn(log_mel)          # (T, F)

        # NEW: dynamic effective hop (train only)
        if self.dynamic_hop and len(self.hop_factors) > 0:
            factor = int(np.random.choice(self.hop_factors, p=self.hop_probs))
            features = self._downsample_time(features, factor)

        if self.clamp_features:
            features = np.clip(features, -10.0, 10.0)

        features = torch.FloatTensor(features)  # (T, F)

        label_ints = self.text_transform.text_to_int(text)
        if len(label_ints) == 0:
            label_ints = [CHAR2IDX[" "]]
        label = torch.LongTensor(label_ints)

        return features, label


def collate_fn(batch):
    features, labels = zip(*batch)

    input_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)

    # (B,T,F) -> (B,1,F,T)
    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True)  # (B,T,F)
    features_padded = features_padded.unsqueeze(1).permute(0, 1, 3, 2).contiguous()  # (B,1,F,T)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concatenated = torch.cat(labels, dim=0)

    return features_padded, labels_concatenated, input_lengths, label_lengths


def _compute_output_lengths(model, input_lengths, device):
    input_lengths = input_lengths.to(device)
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "get_output_lengths"):
        return m.get_output_lengths(input_lengths).to(device)
    # fallback: assume 2 time pools
    return torch.div(input_lengths, 4, rounding_mode="floor").clamp(min=1)


def _forward_log_probs(model, data, input_lengths, blank_logit_bias: float = 0.0):
    """
    All models output raw logits with unified forward(x, input_lengths).
    Apply blank bias then log_softmax once.
    """
    out = model(data, input_lengths)

    if out.dim() != 3:
        raise RuntimeError(f"Model output must be (B,T,C). Got {tuple(out.shape)}")

    if blank_logit_bias and blank_logit_bias != 0.0:
        out = out.clone()
        out[:, :, 0] -= float(blank_logit_bias)

    return out.log_softmax(dim=2)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return nullcontext()


def _prepare_ctc_loss_inputs(
    log_probs_tbc: torch.Tensor,
    targets: torch.Tensor,
    output_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    device: torch.device,
):
    """
    MPS still has uneven operator coverage across PyTorch releases.
    Keep the model forward pass on MPS, but evaluate CTC on CPU for stability.
    """
    if device.type == "mps":
        return (
            log_probs_tbc.float().cpu(),
            targets.detach().cpu(),
            output_lengths.detach().cpu(),
            target_lengths.detach().cpu(),
        )

    return (
        log_probs_tbc,
        targets,
        output_lengths.detach().cpu(),
        target_lengths.detach().cpu(),
    )


def _prepare_decode_inputs(
    log_probs_tbc: torch.Tensor,
    output_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    device: torch.device,
):
    if device.type == "mps":
        return (
            log_probs_tbc.detach().cpu(),
            output_lengths.detach().cpu(),
            targets.detach().cpu(),
            target_lengths.detach().cpu(),
        )
    return log_probs_tbc, output_lengths, targets, target_lengths





def train_one_epoch(model, device, train_loader, criterion, optimizer, epoch, log_interval=10, grad_clip=5.0,
                    blank_logit_bias: float = 0.0, use_amp: bool = False, grad_accum_steps: int = 1,
                    scaler: torch.cuda.amp.GradScaler | None = None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (data, targets, input_lengths, target_lengths) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        with _autocast_context(device, enabled=use_amp):
            output_lengths = _compute_output_lengths(model, input_lengths, device)
            log_probs_btc = _forward_log_probs(model, data, input_lengths.to(device), blank_logit_bias=blank_logit_bias)
            log_probs_tbc = log_probs_btc.permute(1, 0, 2).contiguous()
            loss_log_probs, loss_targets, loss_input_lengths, loss_target_lengths = _prepare_ctc_loss_inputs(
                log_probs_tbc,
                targets,
                output_lengths,
                target_lengths,
                device,
            )

            loss = criterion(
                loss_log_probs,
                loss_targets,
                loss_input_lengths,
                loss_target_lengths,
            )

        if not torch.isfinite(loss):
            print("[WARN] non-finite loss, skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        raw_loss = loss
        scaled_loss = loss / max(1, grad_accum_steps)

        if scaler is not None and use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = ((batch_idx + 1) % max(1, grad_accum_steps) == 0) or ((batch_idx + 1) == len(train_loader))
        grad_norm_for_log = 0.0
        if should_step:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm_for_log = total_norm ** 0.5
            if scaler is not None and use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item())

        if batch_idx % log_interval == 0:
            seen = batch_idx * data.size(0)
            total = len(train_loader.dataset)
            pct = 100.0 * batch_idx / max(1, len(train_loader))
            
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']
            
            # Get Grad Norm (approximate if clipped, or calc before clip if possible. 
            # Here we just check what's currently in params - which is post-backward, pre-step)
            print(
                f"Train Epoch: {epoch} [{seen}/{total} ({pct:.0f}%)]\t"
                f"Loss: {raw_loss.item():.6f}\tLR: {current_lr:.2e}\t"
                f"GradNorm: {grad_norm_for_log:.2f}\tAccum: {grad_accum_steps}"
            )

    return total_loss / max(1, len(train_loader))


def validate(model, device, val_loader, criterion, text_transform: TextTransform, debug_first_batch=True,
             blank_logit_bias: float = 0.0, use_amp: bool = False):
    model.eval()
    val_loss = 0.0
    total_cer = 0
    total_chars = 0
    first_batch = True

    with torch.no_grad():
        for data, targets, input_lengths, target_lengths in val_loader:
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            with _autocast_context(device, enabled=use_amp):
                output_lengths = _compute_output_lengths(model, input_lengths, device)
                log_probs_btc = _forward_log_probs(model, data, input_lengths.to(device), blank_logit_bias=blank_logit_bias)
                log_probs_tbc = log_probs_btc.permute(1, 0, 2).contiguous()
                loss_log_probs, loss_targets, loss_input_lengths, loss_target_lengths = _prepare_ctc_loss_inputs(
                    log_probs_tbc,
                    targets,
                    output_lengths,
                    target_lengths,
                    device,
                )

                loss = criterion(
                    loss_log_probs,
                    loss_targets,
                    loss_input_lengths,
                    loss_target_lengths,
                )
            val_loss += float(loss.item())

            decode_log_probs, decode_output_lengths, decode_targets, decode_target_lengths = _prepare_decode_inputs(
                log_probs_tbc,
                output_lengths,
                targets,
                target_lengths,
                device,
            )

            decoded_preds, decoded_targets = greedy_decoder(
                decode_log_probs,
                decode_output_lengths,
                decode_targets,
                decode_target_lengths,
                text_transform,
            )

            if debug_first_batch and first_batch:
                T0 = int(output_lengths[0].item())
                first_sample = log_probs_tbc[:T0, 0, :]
                argmax_preds = first_sample.argmax(dim=1)
                blank_ratio = (argmax_preds == 0).float().mean().item()

                unique, counts = torch.unique(argmax_preds, return_counts=True)
                pred_dist = {int(u): int(c) for u, c in zip(unique, counts)}

                probs0 = first_sample[0].exp()
                topv, topi = probs0.topk(min(5, probs0.numel()))
                top5 = list(zip(topi.tolist(), [float(v) for v in topv.tolist()]))

                ratio = (output_lengths.float() / target_lengths.float()).cpu()
                print(f"\n[DEBUG] Data Stats: Mean={data.mean().item():.3f}, Min={data.min().item():.3f}, Max={data.max().item():.3f}")
                print(f"[DEBUG] Output Stats: Mean={log_probs_tbc.mean().item():.3f}, Min={log_probs_tbc.min().item():.3f}, Max={log_probs_tbc.max().item():.3f}")
                print(f"[DEBUG] Length ratio (T_model/T_target): mean={ratio.mean().item():.2f}, median={ratio.median().item():.2f}, min={ratio.min().item():.2f}, max={ratio.max().item():.2f}")
                print(f"[DEBUG] Argmax distribution (first sample, valid T={T0}): {pred_dist}")
                print(f"[DEBUG] Blank ratio (first sample): {blank_ratio:.3f}")
                print(f"[DEBUG] t=0 top5 (class, prob): {top5}")
                print(f"[DEBUG] Target: '{decoded_targets[0]}'")
                print(f"[DEBUG] Pred:   '{decoded_preds[0]}'")
                print(f"[DEBUG] T_model: {int(output_lengths[0].item())}, T_target: {int(target_lengths[0].item())}")
                first_batch = False

            for pred, target in zip(decoded_preds, decoded_targets):
                total_cer += editdistance.eval(pred, target)
                total_chars += len(target)

    avg_loss = val_loss / max(1, len(val_loader))
    avg_cer = (total_cer / total_chars) if total_chars > 0 else 0.0
    print(f"\nValidation set: Average loss: {avg_loss:.4f}, CER: {avg_cer:.4f}")
    return avg_loss, avg_cer


def build_model(
    model_type: str,
    vocab_size: int,
    blank_bias: float = 0.0,
    conformer_time_reduction: int = 4,
):
    num_classes = vocab_size + 1  # +1 for blank
    if model_type == "conformer":
        model = Conformer(
            num_classes=num_classes,
            input_dim=64,
            d_model=256,
            num_layers=8,
            blank_bias=blank_bias,
            time_reduction_factor=conformer_time_reduction,
        )
        print(
            "Using Conformer model "
            f"(d_model=256, num_layers=8, time_reduction={conformer_time_reduction}) "
            f"[blank_bias={blank_bias}]"
        )
        return model
    model = CRNN(num_classes=num_classes, blank_bias=blank_bias)
    print(f"Using C-RNN model (Lightweight) [blank_bias={blank_bias}]")
    return model


def _extract_checkpoint_state_dict(checkpoint: object, checkpoint_path: str) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        known_keys = ("state_dict", "model_state_dict", "model", "net", "weights")
        for key in known_keys:
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict) and candidate:
                if all(isinstance(k, str) for k in candidate.keys()):
                    return candidate

        if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
            tensor_like_values = [torch.is_tensor(value) for value in checkpoint.values()]
            if tensor_like_values and all(tensor_like_values):
                return checkpoint

    raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}")


def load_pretrained_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = True) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_checkpoint_state_dict(checkpoint, checkpoint_path)
    state_dict = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
    target_model = model.module if hasattr(model, "module") else model
    incompatible = target_model.load_state_dict(state_dict, strict=strict)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    return missing_keys, unexpected_keys


def _parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _resolve_pretrained_arg(args) -> str:
    load_pretrained = str(getattr(args, "load_pretrained", "") or "").strip()
    init_checkpoint = str(getattr(args, "init_checkpoint", "") or "").strip()

    if load_pretrained and init_checkpoint:
        left = str(Path(load_pretrained).expanduser())
        right = str(Path(init_checkpoint).expanduser())
        if left != right:
            raise ValueError("Use only one of --load_pretrained or --init_checkpoint, or pass the same path to both.")

    return load_pretrained or init_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--valid_csv", required=True)
    parser.add_argument("--audio_dir", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_dir", default="experiments/checkpoints")
    parser.add_argument("--load_pretrained", default="", help="Optional pretrained checkpoint to load before fine-tuning")
    parser.add_argument("--init_checkpoint", default="", help=argparse.SUPPRESS)
    parser.add_argument("--pretrained_non_strict", action="store_true", help="Allow missing/unexpected keys when loading pretrained weights")
    parser.add_argument("--device", type=str, default="auto", choices=DEVICE_CHOICES)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="conformer", choices=["crnn", "conformer"])
    parser.add_argument("--conformer_time_reduction", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--no_feature_clamp", action="store_true")

    # generic blank bias (init + runtime, works for all models)
    parser.add_argument("--blank_logit_bias", type=float, default=2.0)

    # augmentation warmup
    parser.add_argument("--augment_warmup_epochs", type=int, default=2)

    # AudioProcessor knobs
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=160)  # base hop
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--low_cut", type=int, default=400)
    parser.add_argument("--high_cut", type=int, default=1200)
    parser.add_argument("--filter_order", type=int, default=4)
    parser.add_argument("--augment_noise_prob", type=float, default=0.65)
    parser.add_argument("--augment_snr_min_db", type=float, default=6.0)
    parser.add_argument("--augment_snr_max_db", type=float, default=24.0)

    # NEW: dynamic effective hop in TRAIN (downsample time after mel)
    parser.add_argument("--dynamic_hop", action="store_true")
    parser.add_argument("--hop_factors", type=str, default="1,2")     # effective hop = base_hop * factor
    parser.add_argument("--hop_probs", type=str, default="0.5,0.5")
    parser.add_argument("--hop_mode", type=str, default="avgpool", choices=["avgpool", "stride"])
    parser.set_defaults(amp=True)

    args = parser.parse_args()
    pretrained_checkpoint_arg = _resolve_pretrained_arg(args)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training Run Timestamp: {run_timestamp}")

    args.save_dir = os.path.join(args.save_dir, args.model_type, run_timestamp)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {args.save_dir}")
    tensorboard_dir = Path(args.save_dir) / "tensorboard"

    requested_device = str(args.device).lower()
    if requested_device in {"auto", "cuda"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device_ids = list(range(len(args.devices.split(",")))) if args.devices.strip() else []

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Random seed set to {seed}")

    device = resolve_torch_device(args.device)
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")
    if use_cuda:
        print(f"[INFO] CUDA_VISIBLE_DEVICES={args.devices}")
    elif args.devices.strip():
        print(f"[INFO] ignoring --devices={args.devices} because device={device.type}")
    if device.type == "mps":
        print("[INFO] MPS enabled for model forward/backward; CTC loss will run on CPU for compatibility.")

    use_amp = bool(args.amp and supports_amp(device))
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    print(f"[INFO] amp={'ON' if use_amp else 'OFF'} | grad_accum_steps={max(1, args.grad_accum_steps)}")
    writer = create_tensorboard_writer(tensorboard_dir)
    if writer is None:
        print("[WARN] TensorBoard writer unavailable. Install tensorboard to enable event logs.")
    else:
        log_run_config_to_tensorboard(
            writer,
            {
                "train_csv": args.train_csv,
                "valid_csv": args.valid_csv,
                "audio_dir": args.audio_dir,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "load_pretrained": pretrained_checkpoint_arg,
                "pretrained_non_strict": bool(args.pretrained_non_strict),
                "model_type": args.model_type,
                "conformer_time_reduction": args.conformer_time_reduction,
                "devices": args.devices,
                "num_workers": args.num_workers,
                "blank_logit_bias": args.blank_logit_bias,
                "augment_warmup_epochs": args.augment_warmup_epochs,
                "n_fft": args.n_fft,
                "hop_length": args.hop_length,
                "n_mels": args.n_mels,
                "low_cut": args.low_cut,
                "high_cut": args.high_cut,
                "filter_order": args.filter_order,
                "augment_noise_prob": args.augment_noise_prob,
                "augment_snr_min_db": args.augment_snr_min_db,
                "augment_snr_max_db": args.augment_snr_max_db,
                "dynamic_hop": bool(args.dynamic_hop),
                "hop_factors": args.hop_factors,
                "hop_probs": args.hop_probs,
                "hop_mode": args.hop_mode,
                "amp": use_amp,
                "grad_accum_steps": max(1, args.grad_accum_steps),
                "save_dir": args.save_dir,
            },
        )

    # parse dynamic hop lists
    hop_factors = _parse_int_list(args.hop_factors)
    hop_probs = _parse_float_list(args.hop_probs)
    if len(hop_factors) != len(hop_probs):
        raise ValueError("--hop_factors and --hop_probs must have the same length")
    ps = sum(hop_probs)
    hop_probs = [p / ps for p in hop_probs] if ps > 0 else [1.0 / len(hop_probs)] * len(hop_probs)

    clamp = not args.no_feature_clamp

    train_dataset = MorseDataset(
        args.train_csv,
        args.audio_dir,
        augment=True,  # will be warmed up below
        clamp_features=clamp,
        sample_rate=16000,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        low_cut=args.low_cut,
        high_cut=args.high_cut,
        filter_order=args.filter_order,
        augment_noise_prob=args.augment_noise_prob,
        augment_snr_min_db=args.augment_snr_min_db,
        augment_snr_max_db=args.augment_snr_max_db,
        dynamic_hop=args.dynamic_hop,
        hop_factors=hop_factors,
        hop_probs=hop_probs,
        hop_mode=args.hop_mode,
    )

    valid_dataset = MorseDataset(
        args.valid_csv,
        args.audio_dir,
        augment=False,
        clamp_features=clamp,
        sample_rate=16000,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        low_cut=args.low_cut,
        high_cut=args.high_cut,
        filter_order=args.filter_order,
        augment_noise_prob=args.augment_noise_prob,
        augment_snr_min_db=args.augment_snr_min_db,
        augment_snr_max_db=args.augment_snr_max_db,
        dynamic_hop=False,  # IMPORTANT: keep validation fixed
    )

    text_transform = TextTransform()

    pin_memory = bool(use_cuda)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    vocab_size = len(CHARS)
    model = build_model(
        args.model_type,
        vocab_size,
        blank_bias=args.blank_logit_bias,
        conformer_time_reduction=args.conformer_time_reduction,
    )

    if use_cuda and len(device_ids) > 1:
        print(f"Using {len(device_ids)} GPUs with DataParallel")
        model = nn.DataParallel(model, device_ids=device_ids)

    model = model.to(device)
    if pretrained_checkpoint_arg:
        pretrained_checkpoint = Path(pretrained_checkpoint_arg).expanduser().resolve()
        missing_keys, unexpected_keys = load_pretrained_checkpoint(
            model,
            str(pretrained_checkpoint),
            strict=not args.pretrained_non_strict,
        )
        print(f"Loaded pretrained checkpoint: {pretrained_checkpoint}")
        if args.pretrained_non_strict:
            print(
                "[INFO] pretrained_non_strict=True "
                f"| missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)}"
            )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.use_focal_loss:
        criterion = FocalCTCLoss(blank=0, gamma=2.0, alpha=0.25, zero_infinity=True)
        print("Using Focal CTC Loss")
    else:
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        print("Using Standard CTC Loss")

    best_loss = float("inf")
    epoch_history: list[EpochMetrics] = []
    metrics_csv_path = Path(args.save_dir) / "metrics.csv"
    metrics_json_path = Path(args.save_dir) / "metrics.json"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        # warm up augmentation
        train_dataset.augment = epoch > args.augment_warmup_epochs
        if epoch <= args.augment_warmup_epochs:
            print(f"[INFO] augment=False (warmup epoch {epoch}/{args.augment_warmup_epochs})")
        else:
            print("[INFO] augment=True")

        dyn = "ON" if args.dynamic_hop else "OFF"
        if args.dynamic_hop:
            print(f"[INFO] blank_logit_bias={args.blank_logit_bias:.3f} | dynamic_hop={dyn} "
                  f"| factors={hop_factors} probs={[round(p,3) for p in hop_probs]} mode={args.hop_mode}")
        else:
            print(f"[INFO] blank_logit_bias={args.blank_logit_bias:.3f} | dynamic_hop={dyn}")

        train_loss = train_one_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=10,
            grad_clip=args.grad_clip,
            blank_logit_bias=args.blank_logit_bias,
            use_amp=use_amp,
            grad_accum_steps=max(1, args.grad_accum_steps),
            scaler=scaler,
        )
        scheduler.step()

        val_loss, val_cer = validate(
            model=model,
            device=device,
            val_loader=val_loader,
            criterion=criterion,
            text_transform=text_transform,
            debug_first_batch=True,
            blank_logit_bias=args.blank_logit_bias,
            use_amp=use_amp,
        )
        epoch_time_sec = time.perf_counter() - epoch_start
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            val_cer=float(val_cer),
            learning_rate=learning_rate,
            epoch_time_sec=float(epoch_time_sec),
            augment_enabled=bool(train_dataset.augment),
            blank_logit_bias=float(args.blank_logit_bias),
            grad_accum_steps=max(1, args.grad_accum_steps),
            amp_enabled=bool(use_amp),
        )
        epoch_history.append(epoch_metrics)
        write_metrics_csv(epoch_history, metrics_csv_path)
        write_metrics_json(epoch_history, metrics_json_path)
        save_training_plots(epoch_history, args.save_dir)
        log_epoch_to_tensorboard(writer, epoch_metrics)

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Val CER={val_cer:.4f}, Epoch Time={epoch_time_sec:.2f}s"
        )

        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state_dict, os.path.join(args.save_dir, "last_model.pth"))
        torch.save(state_dict, os.path.join(args.save_dir, f"epoch_{epoch}.pth"))

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(state_dict, os.path.join(args.save_dir, "best_model.pth"))
            print(f"Saved Best Model: epoch_{epoch}.pth (Val Loss: {val_loss:.4f})")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
