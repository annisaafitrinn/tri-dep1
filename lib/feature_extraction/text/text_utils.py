"""Shared helpers for loading and sanitising transcription text."""

from pathlib import Path

import pandas as pd


def load_transcription_texts(subject_path: str | Path) -> list[str]:
    """Load the second CSV column as non-empty plain Python strings."""
    subject_path = Path(subject_path)
    csv_files = sorted(subject_path.glob("*.csv"))
    if not csv_files:
        return []

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path, keep_default_na=False)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} has fewer than two columns")

    texts = [str(value).strip() for value in df.iloc[:, 1].tolist()]
    texts = [text for text in texts if text]
    return texts or ["[NO TRANSCRIPTION]"]
