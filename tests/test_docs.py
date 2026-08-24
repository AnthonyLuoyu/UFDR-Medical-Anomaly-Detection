"""Static contracts for the portable GitHub documentation."""

from pathlib import Path
import re


PACKAGE_ROOT = Path(__file__).parents[1]
TITLE = (
    "UFDR: Uncertainty-Guided Feature Distribution Refinement for "
    "Unsupervised Medical Image Anomaly Detection"
)
LEGACY_BASELINE = "DCSR" + "-AD"
LEGACY_CURRICULUM = "S" + "CL"
LEGACY_ATTENTION = "RCA" + "/SASC"
LOCAL_MOUNT_PREFIX = "/" + "mnt/" + "YLUO"
LOCAL_HOME_PREFIX = "/" + "home/"


def _read(relative_path: str) -> str:
    return (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_defines_current_method_and_mechanisms_exactly():
    readme = _read("README.md")

    assert readme.startswith(f"# {TITLE}\n")
    assert TITLE in readme
    assert "PUCL = Pairwise Uncertainty-guided Curriculum Learning" in readme
    assert "TGDR = Trajectory-Guided Decoder Regulation" in readme
    assert "RCA = Re-parameterized calibration attention" in readme
    assert LEGACY_BASELINE not in readme
    assert "SpatialCL" not in readme
    assert re.search(r"\b" + LEGACY_CURRICULUM + r"\b", readme) is None
    assert LEGACY_ATTENTION not in readme


def test_readme_commands_match_the_cli_and_document_portability():
    readme = _read("README.md")

    assert "python scripts/train.py --config configs/ufdr.yaml" in readme
    assert (
        "python scripts/test.py --config configs/ufdr.yaml "
        "--checkpoint outputs/ufdr/best.pt"
    ) in readme
    assert "python -m pytest -q" in readme
    assert "lightly_train._models.dinov3.dinov3_src.hub.backbones" in readme
    assert "相对于包根目录" in readme
    for output_key in ("auc", "average_precision", "num_samples"):
        assert f"`{output_key}`" in readme


def test_readme_documents_scope_data_flow_and_label_modes():
    readme = _read("README.md")

    for term in (
        "原 MedIAnomaly UFDR 实现",
        "不是重新设计",
        "shared DINOv3 ConvNeXt-Tiny encoder",
        "orig",
        "rot180",
        "two independent decoders",
        "4 个 RCA",
        "feature reconstruction",
        "decoder1",
        "反向旋转",
        "class",
        "group",
        "instance",
        "Mamba",
        "Flow",
        "Memory",
        "IQA",
        "WNet",
    ):
        assert term in readme


def test_readme_documents_source_aligned_training_policy():
    readme = _read("README.md")

    for term in (
        "encoder projection",
        "RCA 随 decoder",
        "AdamW",
        "weight_decay=1e-4",
        "10% linear warmup",
        "CosineAnnealingLR",
        "eta_min=1e-7",
        "clip_grad_norm_=0.5",
        "loss_base",
        "避免正则反馈回路",
    ):
        assert term in readme


def test_requirements_are_only_direct_runtime_and_test_dependencies():
    requirements = {
        line.split(">=")[0]
        for line in _read("requirements.txt").splitlines()
        if line and not line.startswith("#")
    }

    assert requirements == {
        "numpy",
        "Pillow",
        "PyYAML",
        "scikit-learn",
        "torch",
        "pytest",
    }
    assert "lightly-train" not in requirements
    assert "lightly_train" not in requirements


def test_third_party_notice_preserves_pucl_source_license_and_boundaries():
    notice = _read("THIRD_PARTY_NOTICES.md")

    assert "extra_network/SpatialCL-master" in notice
    assert "Copyright (c) 2025 Olemou Felix" in notice
    assert "MIT License" in notice
    assert "Permission is hereby granted, free of charge" in notice
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in notice
    assert "DINOv3" in notice
    assert "lightly_train" in notice
    assert "not bundled" in notice


def test_gitignore_covers_generated_assets_without_hiding_source():
    ignored = set(_read(".gitignore").splitlines())

    assert {"data/", "weights/", "outputs/", "__pycache__/", ".pytest_cache/"} <= ignored
    assert not {"ufdr/", "configs/", "scripts/", "tests/"} & ignored


def test_publishable_text_contains_no_local_absolute_paths():
    for relative_path in ("README.md", "THIRD_PARTY_NOTICES.md"):
        text = _read(relative_path)
        assert LOCAL_MOUNT_PREFIX not in text
        assert LOCAL_HOME_PREFIX not in text
