"""Vision Transformer (ViT-B/16) baseline for multimodal depression detection.

Two-phase pipeline:
  1. Extract ViT-B/16 features (768-dim) from EEG and audio spectrogram images
     and encode raw EEG channel data with :class:`~lib.models.models.EEG1DEncoder`.
  2. Train :class:`~lib.models.models.ConvPoolReLUClassifier` on the combined
     embeddings using 5-fold subject-level cross-validation.

Usage
-----
    # Phase 1 — feature extraction
    python scripts/baselines/baseline_vit.py extract \\
        --source_dir data/split_dataset_june

    # Phase 2 — training & evaluation
    python scripts/baselines/baseline_vit.py train \\
        --feat_dir data/split_dataset_june \\
        --fold_file data/split_dataset_june/fold_assignments.json \\
        --output_csv predictions/baseline_vit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from lib.baselines.datasets import SpectrogramEmbeddingDataset, collate_spectrogram
from lib.baselines.models import ConvPoolReLUClassifier, EEG1DEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Reproducibility ───────────────────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Feature extraction ────────────────────────────────────────────────────────


def build_vit(device: torch.device) -> tuple[nn.Module, transforms.Compose]:
    """Load ViT-B/16 pretrained on ImageNet and its preprocessing transform.

    Args:
        device: Torch device for inference.

    Returns:
        Tuple of (vit_model, preprocess_transform).
    """
    vit = models.vit_b_16(pretrained=True).to(device)
    vit.heads = nn.Identity()
    vit.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return vit, preprocess


def extract_image_feature(
    image_path: str,
    model: nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    """Extract a 768-dim ViT feature vector from a spectrogram image.

    Args:
        image_path: Path to a PNG spectrogram image.
        model: ViT model with heads replaced by Identity.
        preprocess: ImageNet preprocessing transform.
        device: Torch device for inference.

    Returns:
        1-D CPU tensor of shape ``(768,)``.
    """
    image = Image.open(image_path).convert("RGB")
    inp = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(inp)
    return feat.squeeze(0).cpu()


def extract_all_features(
    source_dir: str,
    target_dir: str | None,
    device: torch.device,
) -> None:
    """Extract ViT and EEG1D features for all subjects and save as .npy.

    Reads EEG spectrogram images from ``eeg_stft_spectrogram2/``, audio
    spectrograms from ``audio_spectrogram/``, and raw EEG from
    ``<subject_id>_processed.npy`` inside each subject directory.

    Args:
        source_dir: Root directory containing flat subject folders.
        target_dir: Root directory containing subject folders for features.
            Defaults to *source_dir*, so features sit beside each subject's data.
        device: Torch device for inference.
    """
    vit, preprocess = build_vit(device)
    eeg1d = EEG1DEncoder().to(device)
    eeg1d.eval()

    target_dir = target_dir or source_dir
    os.makedirs(target_dir, exist_ok=True)
    subject_ids = sorted(
        name
        for name in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, name))
        and name[:4] in {"0201", "0202", "0203"}
    )

    for subject_id in tqdm(subject_ids, desc="Extracting subjects"):
        subject_path = os.path.join(source_dir, subject_id)
        eeg_dir = os.path.join(subject_path, "eeg_stft_spectrogram2")
        audio_dir = os.path.join(subject_path, "audio_spectrogram")
        if not os.path.isdir(eeg_dir) or not os.path.isdir(audio_dir):
            print(f"Skipping {subject_id} — missing EEG/audio spectrogram directory.")
            continue

        eeg_files = sorted(
            os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith(".png")
        )
        audio_files = sorted(
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".png")
        )

        eeg_embs: list[torch.Tensor] = []
        for path in eeg_files:
            try:
                eeg_embs.append(extract_image_feature(path, vit, preprocess, device))
            except Exception as exc:
                print(f"Skipping EEG image {path}: {exc}")

        audio_embs: list[torch.Tensor] = []
        for path in audio_files:
            try:
                audio_embs.append(extract_image_feature(path, vit, preprocess, device))
            except Exception as exc:
                print(f"Skipping audio image {path}: {exc}")

        if not eeg_embs or not audio_embs:
            print(f"Skipping {subject_id} — insufficient embeddings.")
            continue

        eeg_stack = torch.stack(eeg_embs)
        audio_stack = torch.stack(audio_embs)
        eeg_len, audio_len = eeg_stack.size(0), audio_stack.size(0)
        if audio_len < eeg_len:
            audio_stack = torch.cat(
                [audio_stack, torch.zeros(eeg_len - audio_len, audio_stack.size(1))],
                dim=0,
            )
        elif eeg_len < audio_len:
            eeg_stack = torch.cat(
                [eeg_stack, torch.zeros(audio_len - eeg_len, eeg_stack.size(1))],
                dim=0,
            )

        raw_eeg_path = os.path.join(subject_path, f"{subject_id}_processed.npy")
        if not os.path.isfile(raw_eeg_path):
            raw_eeg_path = os.path.join(subject_path, "processed_segmented_eeg.npy")
        channel_embs: list[np.ndarray] = []
        if os.path.exists(raw_eeg_path):
            eeg_raw = np.load(raw_eeg_path)
            if eeg_raw.ndim == 3:
                # Current format: (segments, channels, samples).
                eeg_raw = eeg_raw.transpose(1, 0, 2).reshape(eeg_raw.shape[1], -1)
            if eeg_raw.ndim != 2:
                raise ValueError(
                    f"Expected 2-D/3-D EEG input for {subject_id}, got {eeg_raw.shape}"
                )
            if eeg_raw.shape[0] > eeg_raw.shape[1] and eeg_raw.shape[1] <= 256:
                eeg_raw = eeg_raw.T
            eeg_raw = (eeg_raw - eeg_raw.mean(axis=1, keepdims=True)) / (
                eeg_raw.std(axis=1, keepdims=True) + 1e-6
            )
            with torch.no_grad():
                for channel in range(eeg_raw.shape[0]):
                    channel_tensor = (
                        torch.tensor(eeg_raw[channel], dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                    channel_embs.append(eeg1d(channel_tensor).cpu().numpy())

        save_dir = os.path.join(target_dir, subject_id)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "vit_eeg_embedding.npy"), eeg_stack.numpy())
        np.save(os.path.join(save_dir, "vit_audio_embedding.npy"), audio_stack.numpy())
        if channel_embs:
            np.save(
                os.path.join(save_dir, "vit_eeg1d_embedding.npy"),
                np.stack(channel_embs),
            )


# ── Training ──────────────────────────────────────────────────────────────────


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 70,
) -> nn.Module:
    """Train with early-stopping on validation accuracy.

    Args:
        model: Classifier to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Parameter optimiser.
        device: Torch device.
        epochs: Maximum training epochs. Default 70.

    Returns:
        Model loaded with the best-validation-accuracy weights.
    """
    best_state, best_val_acc = None, 0.0
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def write_predictions(predictions: list[dict], output_csv: Path) -> None:
    """Write out-of-fold predictions in the repository's standard CSV format."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "fold",
        "ground_truth",
        "predicted_label",
        "probability",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            predictions, key=lambda item: (item["fold"], item["subject_id"])
        ):
            writer.writerow(
                {
                    **row,
                    "probability": f"{float(row['probability']):.6f}",
                }
            )


def run_cv(
    feat_dir: str,
    fold_file: str,
    device: torch.device,
    output_csv: Path,
) -> None:
    """Run 5-fold cross-validation, print metrics, and save OOF predictions.

    Args:
        feat_dir: Root directory containing pre-extracted per-subject .npy files.
        fold_file: Path to ``fold_assignments.json``.
        device: Torch device for training.
        output_csv: Destination for all out-of-fold subject predictions.
    """
    with open(fold_file) as f:
        folds = json.load(f)

    subject_ids = sorted(
        {
            subject_id
            for fold in folds.values()
            for split in ("train", "val", "test")
            for subject_id in fold[split]
        }
    )
    subject_path_map = {
        subject_id: os.path.join(feat_dir, subject_id) for subject_id in subject_ids
    }
    missing_features = [
        os.path.join(subject_path, filename)
        for subject_path in subject_path_map.values()
        for filename in (
            "vit_eeg_embedding.npy",
            "vit_audio_embedding.npy",
            "vit_eeg1d_embedding.npy",
        )
        if not os.path.isfile(os.path.join(subject_path, filename))
    ]
    if missing_features:
        raise FileNotFoundError(
            "Missing ViT features; run the extract phase first. First missing file: "
            f"{missing_features[0]}"
        )

    results: list[tuple[float, float, float, float]] = []
    all_predictions: list[dict] = []
    for fold_idx in range(5):
        fn = f"fold_{fold_idx + 1}"
        train_subjs = [subject_path_map[s] for s in folds[fn]["train"]]
        val_subjs = [subject_path_map[s] for s in folds[fn]["val"]]
        test_subjs = [subject_path_map[s] for s in folds[fn]["test"]]

        train_loader = DataLoader(
            SpectrogramEmbeddingDataset(
                train_subjs,
                include_eeg1d=True,
                eeg_filename="vit_eeg_embedding.npy",
                audio_filename="vit_audio_embedding.npy",
                eeg1d_filename="vit_eeg1d_embedding.npy",
            ),
            batch_size=100, shuffle=True, collate_fn=collate_spectrogram,
        )
        val_loader = DataLoader(
            SpectrogramEmbeddingDataset(
                val_subjs,
                include_eeg1d=True,
                eeg_filename="vit_eeg_embedding.npy",
                audio_filename="vit_audio_embedding.npy",
                eeg1d_filename="vit_eeg1d_embedding.npy",
            ),
            batch_size=100, collate_fn=collate_spectrogram,
        )
        test_loader = DataLoader(
            SpectrogramEmbeddingDataset(
                test_subjs,
                include_eeg1d=True,
                eeg_filename="vit_eeg_embedding.npy",
                audio_filename="vit_audio_embedding.npy",
                eeg1d_filename="vit_eeg1d_embedding.npy",
            ),
            batch_size=100, collate_fn=collate_spectrogram,
        )

        model = ConvPoolReLUClassifier(input_dim=2048).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0004)
        criterion = nn.CrossEntropyLoss()

        trained = train_model(model, train_loader, val_loader, criterion, optimizer, device)
        trained.eval()

        fold_preds, fold_labels = [], []
        fold_predictions: list[dict] = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = trained(x)
                probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1)
                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(y.cpu().numpy())
                start = len(fold_predictions)
                batch_subjects = test_subjs[start : start + y.size(0)]
                for subject_path, label, prediction, probability in zip(
                    batch_subjects,
                    y.cpu().numpy(),
                    preds.cpu().numpy(),
                    probabilities,
                ):
                    fold_predictions.append(
                        {
                            "subject_id": os.path.basename(subject_path),
                            "fold": fold_idx + 1,
                            "ground_truth": int(label),
                            "predicted_label": int(prediction),
                            "probability": float(probability),
                        }
                    )

        acc  = sum(p == l for p, l in zip(fold_preds, fold_labels)) / len(fold_labels)
        prec = precision_score(fold_labels, fold_preds, average="macro")
        rec  = recall_score(fold_labels, fold_preds, average="macro")
        f1   = f1_score(fold_labels, fold_preds, average="macro")
        print(f"Fold {fold_idx+1}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        results.append((acc, prec, rec, f1))
        all_predictions.extend(fold_predictions)

    arr = np.array(results)
    print("\n5-Fold CV Summary:")
    for i, name in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        print(f"  {name}: {arr[:, i].mean():.4f} ± {arr[:, i].std():.4f}")
    write_predictions(all_predictions, output_csv)
    print(f"\nOut-of-fold predictions saved to: {output_csv}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and dispatch to extract or train phase."""
    parser = argparse.ArgumentParser(description="ViT baseline for depression detection")
    sub = parser.add_subparsers(dest="phase", required=True)

    ext = sub.add_parser("extract", help="Extract ViT + EEG1D features from spectrograms")
    ext.add_argument("--source_dir", default="data/split_dataset_june")
    ext.add_argument(
        "--target_dir",
        default=None,
        help="Feature root with per-subject folders (default: save beside each source subject).",
    )
    ext.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    trn = sub.add_parser("train", help="Train classifier on pre-extracted features")
    trn.add_argument("--feat_dir", default="data/split_dataset_june")
    trn.add_argument("--fold_file", default="data/split_dataset_june/fold_assignments.json")
    trn.add_argument(
        "--output_csv",
        default="predictions/baseline_vit.csv",
        help="Path for subject-level out-of-fold predictions.",
    )
    trn.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    set_seed(42)
    device = torch.device(args.device)

    if args.phase == "extract":
        extract_all_features(args.source_dir, args.target_dir, device)
    else:
        output_csv = Path(args.output_csv)
        if not output_csv.is_absolute():
            output_csv = PROJECT_ROOT / output_csv
        run_cv(args.feat_dir, args.fold_file, device, output_csv)


if __name__ == "__main__":
    main()
