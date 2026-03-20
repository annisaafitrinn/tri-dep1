"""Factory function for instantiating audio encoders by name.

This module is provided for convenience; the recommended approach is to
import encoder classes directly via ``utils.encoders``.
"""

from utils.encoders.cnn import AudioCNNEncoder
from utils.encoders.cnn_lstm import CNNLSTMEncoder
from utils.encoders.cnn_gru import CNNGRUEncoder
from utils.encoders.cnn_bigru import BiGRUAttentionEncoder
from utils.encoders.cnn_bilstm import AudioTemporalBiLSTMEncoder

import torch.nn as nn


def get_encoder(name: str | None) -> nn.Module | None:
    """Instantiate an encoder by name with default hyperparameters.

    Args:
        name: One of ``"cnn"``, ``"cnn_lstm"``, ``"cnn_gru"``,
            ``"cnn_bigru"``, ``"cnn_bilstm"`` (case-insensitive), or
            ``None`` to return ``None``.

    Returns:
        An encoder instance in eval mode, or ``None`` if *name* is
        ``None``.

    Raises:
        ValueError: If *name* is not a recognised encoder identifier.
    """
    if name is None:
        return None

    name = name.lower()
    if name == "cnn":
        return AudioCNNEncoder().eval()
    elif name == "cnn_lstm":
        return CNNLSTMEncoder().eval()
    elif name == "cnn_gru":
        return CNNGRUEncoder().eval()
    elif name == "cnn_bigru":
        return BiGRUAttentionEncoder().eval()
    elif name == "cnn_bilstm":
        return AudioTemporalBiLSTMEncoder().eval()
    else:
        raise ValueError(f"Unknown encoder: {name}")
