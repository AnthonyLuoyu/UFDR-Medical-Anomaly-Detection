"""Shared test configuration for UFDR validation."""

from copy import deepcopy

import pytest


@pytest.fixture
def canonical_config():
    """Return an independent copy of the canonical portable configuration."""
    config = {
        "seed": 0,
        "device": "cuda",
        "data": {
            "root": "data/example",
            "image_size": 256,
            "channels": 3,
            "batch_size": 64,
            "workers": 4,
        },
        "model": {
            "encoder": "dinov3_convnext_tiny",
            "weights": "weights/dinov3_convnext_tiny_lvd1689m.pth",
            "aux_view": "rot180",
            "freeze_encoder_epochs": 100,
            "cosine_weight": 0.5,
            "mse_weight": 0.05,
        },
        "pucl": {
            "temperature": 0.1,
            "weight": 0.01,
            "label_mode": "class",
            "group_size": 8,
            "eps": 1e-12,
        },
        "tgdr": {
            "window_size": 20,
            "base_l2_lambda": 3e-4,
            "max_lambda": 0.03,
            "target": "decoder1",
            "reliability": "corr_gap",
        },
        "train": {
            "epochs": 600,
            "lr_encoder": 5e-6,
            "lr_projection": 1e-4,
            "lr_decoder": 1e-3,
            "output_dir": "outputs/ufdr",
        },
    }
    return deepcopy(config)
