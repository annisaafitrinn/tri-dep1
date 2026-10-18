"""Generate EEG and audio spectrogram PNGs for baseline feature extraction.

The script uses the repository's flat subject-folder layout and fixed fold
assignments. Outputs are written inside each subject folder as
``eeg_stft_spectrogram2/`` and ``audio_spectrogram/``, the directories expected
by the ViT and DenseNet baseline extractors.

Examples:
    python scripts/baselines/generate_spectrograms.py --modality all
    python scripts/baselines/generate_spectrograms.py --modality eeg
    python scripts/baselines/generate_spectrograms.py --modality audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "split_dataset_june"
DEFAULT_FOLD_FILE = DEFAULT_DATA_DIR / "fold_assignments.json"
EEG_FILENAME = "processed_segmented_eeg.npy"
AUDIO_SUBDIR = "processed_audio"
EEG_OUTPUT_SUBDIR = "eeg_stft_spectrogram2"
AUDIO_OUTPUT_SUBDIR = "audio_spectrogram"


def subject_ids_from_folds(fold_file: Path) -> list[str]:
    """Return every unique subject referenced by the fixed fold assignments."""
    with fold_file.open(encoding="utf-8") as file:
        folds = json.load(file)
    return sorted(
        {
            subject_id
            for fold in folds.values()
            for split_name in ("train", "val", "test")
            for subject_id in fold[split_name]
        }
    )


def load_eeg_for_spectrogram(eeg_file: Path) -> np.ndarray:
    """Load EEG as ``(channels, time)`` from raw or segmented repository data.

    Current preprocessing outputs ``(segments, channels, samples)``. Segments
    are concatenated in temporal order to make one full-recording signal per
    channel, matching the notebook's one-image-per-channel approach.
    """
    eeg = np.load(eeg_file)
    if eeg.ndim == 3:
        # (segments, channels, samples) -> (channels, segments * samples)
        eeg = eeg.transpose(1, 0, 2).reshape(eeg.shape[1], -1)
    elif eeg.ndim != 2:
        raise ValueError(
            f"Expected a 2-D or 3-D EEG array at {eeg_file}; got shape {eeg.shape}"
        )

    if eeg.shape[0] > eeg.shape[1] and eeg.shape[1] <= 256:
        eeg = eeg.T
    if eeg.shape[0] > 256:
        raise ValueError(
            f"Could not identify the channel axis in EEG shape {eeg.shape} at {eeg_file}"
        )
    return np.asarray(eeg, dtype=np.float32)


def save_spectrogram_image(
    spectrogram: np.ndarray,
    output_path: Path,
    *,
    sample_rate: int,
    hop_length: int,
    y_axis: str | None = None,
) -> None:
    """Render a spectrogram with the notebook's magma, axis-free styling."""
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.set_axis_off()
    librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        hop_length=hop_length,
        y_axis=y_axis,
        x_axis=None,
        cmap="magma",
        ax=axis,
    )
    figure.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )
    plt.close(figure)


def generate_eeg_spectrograms(
    subject_dir: Path,
    *,
    sample_rate: int = 250,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> int:
    """Save one STFT image per EEG channel; return the number written."""
    eeg_file = subject_dir / EEG_FILENAME
    if not eeg_file.is_file():
        print(f"Skipping EEG for {subject_dir.name}: missing {eeg_file.name}")
        return 0

    eeg = load_eeg_for_spectrogram(eeg_file)
    output_dir = subject_dir / EEG_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for channel_index, channel in enumerate(eeg, start=1):
        stft = librosa.stft(channel, n_fft=n_fft, hop_length=hop_length)
        spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        save_spectrogram_image(
            spectrogram_db,
            output_dir / f"eeg_channel_{channel_index}.png",
            sample_rate=sample_rate,
            hop_length=hop_length,
            y_axis="log",
        )
        written += 1
    return written


def generate_audio_spectrograms(
    subject_dir: Path,
    *,
    audio_subdir: str = AUDIO_SUBDIR,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
) -> int:
    """Save one Mel spectrogram image per preprocessed recording."""
    audio_dir = subject_dir / audio_subdir
    if not audio_dir.is_dir():
        print(f"Skipping audio for {subject_dir.name}: missing {audio_dir}")
        return 0

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"Skipping audio for {subject_dir.name}: no WAV files in {audio_dir}")
        return 0

    output_dir = subject_dir / AUDIO_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_index, audio_file in enumerate(audio_files):
        try:
            waveform, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
            mel_power = librosa.feature.melspectrogram(
                y=waveform,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmax=sample_rate / 2,
            )
            mel_db = librosa.power_to_db(mel_power, ref=np.max)
            save_spectrogram_image(
                mel_db,
                output_dir / f"audio_{image_index:02d}.png",
                sample_rate=sample_rate,
                hop_length=hop_length,
            )
            written += 1
        except Exception as error:
            print(f"Skipping audio {audio_file}: {error}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate baseline EEG STFT and audio Mel spectrogram images."
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fold_file", type=Path, default=DEFAULT_FOLD_FILE)
    parser.add_argument(
        "--modality",
        choices=("eeg", "audio", "all"),
        default="all",
        help="Which spectrograms to generate (default: all).",
    )
    parser.add_argument(
        "--audio_subdir",
        default=AUDIO_SUBDIR,
        help="Subject-relative folder containing WAVs (default: processed_audio).",
    )
    parser.add_argument("--eeg_sample_rate", type=int, default=250)
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=64)
    args = parser.parse_args()

    if not args.fold_file.is_file():
        parser.error(f"Fold assignment file not found: {args.fold_file}")
    subject_ids = subject_ids_from_folds(args.fold_file)
    if not subject_ids:
        parser.error(f"No subject IDs found in fold file: {args.fold_file}")

    totals = {"eeg": 0, "audio": 0}
    for subject_id in subject_ids:
        subject_dir = args.data_dir / subject_id
        if not subject_dir.is_dir():
            print(f"Skipping unknown subject directory: {subject_dir}")
            continue
        if args.modality in ("eeg", "all"):
            totals["eeg"] += generate_eeg_spectrograms(
                subject_dir,
                sample_rate=args.eeg_sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            )
        if args.modality in ("audio", "all"):
            totals["audio"] += generate_audio_spectrograms(
                subject_dir,
                audio_subdir=args.audio_subdir,
                sample_rate=args.audio_sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
            )

    print(
        f"Finished. Wrote {totals['eeg']} EEG-channel images and "
        f"{totals['audio']} audio images across {len(subject_ids)} assigned subjects."
    )


if __name__ == "__main__":
    main()
