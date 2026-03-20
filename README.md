# TRI-DEP: Trimodal Depression Detection

Depression detection from EEG, speech, and text using the [MODMA dataset](http://modma.lzu.edu.cn).
Experiments are conducted with 5-fold subject-level cross-validation on 38 aligned subjects (17 MDD, 21 HC).

## Results summary

| Modality / Fusion | Best config | Macro-F1 |
|---|---|---|
| Text (MacBERT) | `macbert_lstm` | 0.843 |
| Speech (XLSR-53) | `hubert_bigru_conv` | 0.764 |
| EEG (CBraMod-Mumtaz) | `cbramod_mumtaz_conv` | 0.639 |
| Early fusion (concat) | `early_concat` | 0.748 |
| Intermediate fusion | `intermediate_concat` | 0.748 |
| **Late fusion (WA)** | **`WA: EEG+Speech+Text`** | **0.864** |

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Obtain the dataset

Request access to MODMA at [modma.lzu.edu.cn](http://modma.lzu.edu.cn).
Place the raw archives in a directory with this structure:

```
dataset/
├── audio_lanzhou_2015-2/          # raw .wav recordings
└── EEG_128channels_resting_lanzhou_2015/   # raw .mat EEG files
```

---

## Pipeline

### Step 1 — Build the aligned dataset

Find subjects present in both EEG and audio modalities and create a unified directory tree.

```bash
python lib/preprocessing/create_dataset.py \
    --eeg_dir dataset/EEG_128channels_resting_lanzhou_2015 \
    --audio_dir dataset/audio_lanzhou_2015-2 \
    --output_dir data/split_dataset_june
```

Expected output: **38 aligned subjects**.

The resulting layout:
```
data/split_dataset_june/
├── fold_assignments.json
└── <subject_id>/
    ├── eeg/          *.mat
    └── audio/        *.wav
```

---

### Step 2 — Preprocessing

#### EEG

Bandpass-filter (0.5–50 Hz), average-reference, and segment into 10-second epochs (30 segments per subject).

```bash
python lib/preprocessing/eeg_preprocessing.py \
    --input_dir data/split_dataset_june \
    --output_dir data/split_dataset_june
```

Output per subject: `processed_segmented_eeg.npy` — shape `(30, 29, 2500)`.

#### Speech

Normalise amplitude, trim silence, and segment into overlapping 5-second clips (2.5 s stride).

```bash
python lib/preprocessing/speech_preprocessing.py
```

Output per subject: `processed_audio/*.wav` and `segmented_audio/*.wav`.

#### Transcription

Transcribe all audio files with WhisperX (Mandarin Chinese).

```bash
python lib/preprocessing/audio_transcription.py
```

Output per subject: `transcriptions_<subject_id>.csv`.

---

### Step 3 — Feature extraction

All features are saved as `.npy` files inside each subject's directory under `data/split_dataset_june/`.

#### EEG features

**CBraMod (original pretrained weights)**
```bash
python lib/feature_extraction/eeg/extract_cbramod.py \
    --PRETRAINED_WEIGHTS cbramod_pretrained_weights/pretrained-weights.pth
```

**CBraMod (Mumtaz fine-tuned weights — used in paper)**
```bash
python lib/feature_extraction/eeg/extract_cbramod.py \
    --PRETRAINED_WEIGHTS cbramod_pretrained_weights/pretrained-weights2.pth
```

Output: `cbramod_embeddings.npy` or `cbramod_mumtaz_embeddings.npy` — shape `(30, 200)`.

**LaBraM**
```bash
python lib/feature_extraction/eeg/extract_labram.py
```

Output: `labram_embeddings.npy`.

**Handcrafted EEG features** (10 per channel per segment)
```bash
python lib/feature_extraction/eeg/extract_handcrafted_features.py
```

Output: `eeg_handcrafted_features.npy` — shape `(30, 29, 10)`.

#### Text features

Encode transcriptions with all four language models:

```bash
python lib/feature_extraction/text/extract_features_text.py \
    --models macbert bert mpnet xlnet \
    --base_dir data/split_dataset_june \
    --save_dir data/split_dataset_june
```

Output per subject: `text_embedding_macbert.npy`, `text_embedding_bert.npy`, etc.

#### Speech features

**Pretrained model embeddings + encoder**
```bash
python lib/feature_extraction/speech/extract_features_speech.py \
    --base_dir data/split_dataset_june \
    --feature_extractor xslr53 \
    --encoder cnn_bigru \
    --input_dim 1024 \
    --encoder_output_dim 256
```

`--feature_extractor` options: `xslr53`, `chinese_hubert`, `mfcc`
`--encoder` options: `cnn_bigru`, `cnn_lstm`, `cnn_gru`, `cnn_bilstm`, `cnn`

Output: `audio_<extractor>_encoded_<encoder>.npy` — shape `(29, 256)`.

**Handcrafted speech features** (46 per segment)
```bash
python lib/feature_extraction/speech/extract_handcrafted_features_speech.py \
    --base_dir data/split_dataset_june
```

Output: `raw_audio_features.npy` — object array of shape `(29,)` with per-recording feature arrays.

---

### Step 4 — Training & evaluation (unimodal / early / intermediate fusion)

All experiments use YAML configs under `configs/training/` and are run via `scripts/inference.py`.

**Run a single named configuration:**
```bash
python scripts/inference.py --config configs/training/text.yaml --name macbert_lstm
python scripts/inference.py --config configs/training/speech.yaml --name hubert_bigru_conv
python scripts/inference.py --config configs/training/eeg.yaml --name cbramod_mumtaz_conv
python scripts/inference.py --config configs/training/early_fusion.yaml --name early_concat
python scripts/inference.py --config configs/training/intermediate_fusion.yaml --name intermediate_concat
```

**Run all configurations in a file:**
```bash
python scripts/inference.py --config configs/training/eeg.yaml --all
```

**List available configurations:**
```bash
python scripts/inference.py --config configs/training/text.yaml --list
```

**Override YAML values from the CLI (OmegaConf dot-notation):**
```bash
python scripts/inference.py --config configs/training/eeg.yaml --all data.base_dir=/custom/path
```

Predictions are saved to `predictions/<output_csv>`.
Checkpoints are saved to `checkpoints/<config_name>/fold_{1-5}.pt`.

Available config files:

| File | Modality | Configs |
|---|---|---|
| `configs/training/text.yaml` | Text | 8 (MPNet, MacBERT, BERT, XLNet × LSTM / ConvPool) |
| `configs/training/speech.yaml` | Speech | 21 (HuBERT, XLSR-53, MFCC, handcrafted × encoder) |
| `configs/training/eeg.yaml` | EEG | 12 (CBraMod, CBraMod-Mumtaz, LaBraM, handcrafted × classifier) |
| `configs/training/early_fusion.yaml` | Early fusion | 6 (concat, bottleneck × feature groups) |
| `configs/training/intermediate_fusion.yaml` | Intermediate fusion | 4 (concat, gated) |

---

### Step 5 — Late (decision-level) fusion

**Run all 12 predefined fusion configurations:**
```bash
python scripts/fusion.py --config configs/training/fusion.yaml
```

**Grid-search for optimal weights:**
```bash
python scripts/fusion_grid_search.py --config configs/training/fusion.yaml
```

Fusion configs (weights, modality files, prior) are defined in `configs/training/fusion.yaml`.
Fused prediction CSVs are saved alongside the unimodal CSVs in `predictions/`.

---

## Project structure

```
tri-dep1/
├── configs/
│   ├── preprocessing.yaml
│   ├── feature_extraction/
│   │   ├── eeg.yaml
│   │   ├── speech.yaml
│   │   └── text.yaml
│   └── training/
│       ├── eeg.yaml
│       ├── speech.yaml
│       ├── text.yaml
│       ├── early_fusion.yaml
│       ├── intermediate_fusion.yaml
│       └── fusion.yaml
├── lib/
│   ├── datasets.py               # TextDataset, TrimodalDataset, RawAudioDataset
│   ├── models/
│   │   ├── models.py             # 15 classifier architectures
│   │   ├── encoders/             # CNN-BiGRU, CNN-LSTM, CNN-GRU, CNN-BiLSTM, CNN
│   │   └── cbramod/              # CBraMod pretrained model
│   ├── preprocessing/
│   │   ├── create_dataset.py
│   │   ├── eeg_preprocessing.py
│   │   ├── speech_preprocessing.py
│   │   └── audio_transcription.py
│   └── feature_extraction/
│       ├── eeg/                  # CBraMod, LaBraM, handcrafted
│       ├── speech/               # XLSR-53, HuBERT, MFCC, handcrafted
│       └── text/                 # MacBERT, BERT, MPNet, XLNet
├── scripts/
│   ├── inference.py              # training + evaluation loop
│   ├── fusion.py                 # decision-level fusion
│   ├── fusion_grid_search.py     # weight optimisation
│   └── fusion_significance.py    # McNemar / permutation tests
├── utils/
│   ├── speech/                   # speech feature extractor wrappers
│   └── text/                     # text encoding functions
├── cbramod_pretrained_weights/
├── requirements.txt
└── setup.py
```

---

## Citation

If you use this code, please cite the MODMA dataset:

> Cai, H., et al. (2020). *A multi-modal open dataset for mental-disorder analysis*.
> Scientific Data, 9, 178. https://doi.org/10.1038/s41597-022-01211-x
