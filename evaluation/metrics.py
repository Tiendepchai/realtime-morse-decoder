import editdistance

def calculate_cer(pred: str, target: str) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance.
    CER = (Substitutions + Deletions + Insertions) / Total Characters
    """
    if len(target) == 0:
        return 0.0
    return editdistance.eval(pred, target) / len(target)

def calculate_wer(pred: str, target: str) -> float:
    """
    Calculate Word Error Rate (WER) using Levenshtein distance.
    WER = (Substitutions + Deletions + Insertions) / Total Words
    """
    pred_words = pred.strip().split()
    target_words = target.strip().split()
    if len(target_words) == 0:
        return 0.0
    return editdistance.eval(pred_words, target_words) / len(target_words)
