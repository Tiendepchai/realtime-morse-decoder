import re

import torch

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}


def normalize_text(text: str, allowed_chars: str = CHARS, restrict_charset: bool = True) -> str:
    """
    Normalize text consistently across training-time labels and evaluation-time predictions.
    """
    allowed = set(allowed_chars)
    normalized_chars = []

    for char in str(text).upper():
        if char.isspace():
            normalized_chars.append(" ")
        elif not restrict_charset or char in allowed:
            normalized_chars.append(char)

    normalized = "".join(normalized_chars)
    normalized = re.sub(r" +", " ", normalized)
    return normalized.strip()

class TextTransform:
    def text_to_int(self, text: str):
        seq = []
        for c in str(text).upper():
            if c in CHAR2IDX:
                seq.append(CHAR2IDX[c])
        return seq

    def int_to_text(self, sequence):
        return "".join([IDX2CHAR.get(int(i), "") for i in sequence])

def greedy_decoder(log_probs_tbc, output_lengths, labels, label_lengths, text_transform: TextTransform):
    arg_maxes = torch.argmax(log_probs_tbc, dim=2)  # (T, B)
    decodes, targets = [], []

    start = 0
    for b, args in enumerate(arg_maxes.transpose(0, 1)):  # (B, T)
        T_b = int(output_lengths[b].item())
        args = args[:T_b]

        decode = []
        prev = 0
        for idx in args.tolist():
            if idx != 0 and idx != prev:
                decode.append(idx)
            prev = idx
        decodes.append(text_transform.int_to_text(decode))

        if labels is not None and label_lengths is not None:
            tlen = int(label_lengths[b].item())
            tgt = labels[start:start + tlen].tolist()
            targets.append(text_transform.int_to_text(tgt))
            start += tlen
        else:
            targets.append("")

    return decodes, targets
