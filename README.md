# UFDR: Uncertainty-Guided Feature Distribution Refinement for Unsupervised Medical Image Anomaly Detection

Official implementation of **UFDR** for unsupervised medical image anomaly detection.

UFDR is a normal-only learning framework that refines normal-feature distributions through uncertainty-guided feature modeling and improves anomaly-sensitive reconstruction through adaptive decoder regulation.

## Repository Structure

```text
UFDR/
├── configs/ufdr.yaml        # Example configuration
├── scripts/
│   ├── train.py             # Training entry
│   └── test.py              # Image-level evaluation
├── ufdr/                    # Model, training, and data modules
├── tests/                   # Lightweight tests
├── requirements.txt
└── THIRD_PARTY_NOTICES.md

```text

---

## Core Components

- **PUCL**: Pairwise Uncertainty-guided Curriculum Learning  
  Refines normal-feature distributions by modeling uncertainty in pairwise relations and progressively organizing normal features into compact local modes.

- **TGDR**: Trajectory-Guided Decoder Regulation  
  Dynamically adjusts decoder regularization according to training-validation trajectory inconsistency to prevent abnormal-feature over-reconstruction.

- **RCA**: Re-parameterized Calibration Attention  
  Enhances skip features by introducing non-local structural context during feature reconstruction.



