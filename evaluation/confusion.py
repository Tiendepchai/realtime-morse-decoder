from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.utils.text import CHARS

INS_TOKEN = "<INS>"
DEL_TOKEN = "<DEL>"
SPACE_TOKEN = "<SPACE>"


def display_token(token: str) -> str:
    return SPACE_TOKEN if token == " " else token


def align_token_sequences(
    reference_tokens: Sequence[str],
    prediction_tokens: Sequence[str],
) -> list[tuple[str, str]]:
    rows = len(reference_tokens)
    cols = len(prediction_tokens)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]

    for row in range(1, rows + 1):
        dp[row][0] = row
    for col in range(1, cols + 1):
        dp[0][col] = col

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            substitution_cost = 0 if reference_tokens[row - 1] == prediction_tokens[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + substitution_cost,
            )

    alignment: list[tuple[str, str]] = []
    row = rows
    col = cols
    while row > 0 or col > 0:
        if row > 0 and col > 0:
            substitution_cost = 0 if reference_tokens[row - 1] == prediction_tokens[col - 1] else 1
            if dp[row][col] == dp[row - 1][col - 1] + substitution_cost:
                alignment.append((reference_tokens[row - 1], prediction_tokens[col - 1]))
                row -= 1
                col -= 1
                continue

        if row > 0 and dp[row][col] == dp[row - 1][col] + 1:
            alignment.append((reference_tokens[row - 1], DEL_TOKEN))
            row -= 1
            continue

        alignment.append((INS_TOKEN, prediction_tokens[col - 1]))
        col -= 1

    alignment.reverse()
    return alignment


def build_confusion_counter(
    records: Iterable[Mapping[str, Any]],
    reference_key: str = "reference_normalized",
    prediction_key: str = "prediction_normalized",
) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for record in records:
        reference = str(record.get(reference_key, ""))
        prediction = str(record.get(prediction_key, ""))
        for pair in align_token_sequences(list(reference), list(prediction)):
            counter[pair] += 1
    return counter


def confusion_counter_to_dataframe(
    counter: Counter[tuple[str, str]],
    vocabulary: str = CHARS,
) -> pd.DataFrame:
    row_tokens = [*vocabulary, INS_TOKEN]
    column_tokens = [*vocabulary, DEL_TOKEN]
    dataframe = pd.DataFrame(
        0,
        index=[display_token(token) for token in row_tokens],
        columns=[display_token(token) for token in column_tokens],
        dtype=int,
    )

    for (reference_token, prediction_token), count in counter.items():
        dataframe.at[display_token(reference_token), display_token(prediction_token)] += int(count)
    return dataframe


def top_confusion_pairs(
    counter: Counter[tuple[str, str]],
    top_n: int = 20,
    predicate: Callable[[str, str], bool] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (reference_token, prediction_token), count in counter.most_common():
        if reference_token == prediction_token and reference_token not in {INS_TOKEN, DEL_TOKEN}:
            continue
        if predicate is not None and not predicate(reference_token, prediction_token):
            continue
        rows.append(
            {
                "reference": display_token(reference_token),
                "prediction": display_token(prediction_token),
                "count": int(count),
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def is_space_related(reference_token: str, prediction_token: str) -> bool:
    return reference_token == " " or prediction_token == " "


def is_digit_related(reference_token: str, prediction_token: str) -> bool:
    return reference_token.isdigit() or prediction_token.isdigit()
