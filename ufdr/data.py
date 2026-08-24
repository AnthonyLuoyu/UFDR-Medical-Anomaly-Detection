"""Portable folder datasets and deterministic data loaders for UFDR."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_SPLITS = {"train", "val", "test"}


def _image_paths(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        ),
        key=lambda path: (path.relative_to(directory).as_posix().lower(), path.name),
    )


class UFDRFolderDataset(Dataset):
    """Read the small, generic folder layout documented by this package."""

    def __init__(
        self,
        root,
        split,
        image_size: int = 256,
        channels: int = 3,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError("split must be one of: train, val, test")
        if (
            isinstance(image_size, bool)
            or not isinstance(image_size, int)
            or image_size <= 0
        ):
            raise ValueError("image_size must be a positive integer")
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels not in (1, 3)
        ):
            raise ValueError("channels must be 1 or 3")

        self.root = Path(root).expanduser()
        self.split = split
        self.image_size = image_size
        self.channels = channels
        classes = (("normal", 0),) if split != "test" else (
            ("normal", 0),
            ("anomaly", 1),
        )

        samples: list[tuple[Path, int]] = []
        for class_name, label in classes:
            directory = self.root / split / class_name
            if not directory.is_dir():
                relative = Path(split) / class_name
                raise FileNotFoundError(
                    f"required dataset directory does not exist: {relative} "
                    f"(under {self.root})"
                )
            paths = _image_paths(directory)
            if not paths:
                raise ValueError(
                    f"no supported images found for {split}/{class_name} "
                    f"under {self.root}"
                )
            samples.extend((path, label) for path in paths)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        path, label = self.samples[index]
        mode = "L" if self.channels == 1 else "RGB"
        with Image.open(path) as image:
            image = image.convert(mode)
            image = image.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
            array = np.array(image, dtype=np.float32, copy=True)

        if self.channels == 1:
            array = array[None, :, :]
        else:
            array = array.transpose(2, 0, 1)
        tensor = torch.from_numpy(array).div_(127.5).sub_(1.0)
        return {
            "image": tensor,
            "label": label,
            "contrast_label": label,
            "path": str(path),
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(config, generator=None) -> dict[str, DataLoader]:
    """Build deterministic loaders for all three required dataset splits."""

    data = config["data"]
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(config["seed"])
    elif not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator or None")
    if generator.device.type != "cpu":
        raise ValueError("generator must be a CPU torch.Generator")
    initial_generator_state = generator.get_state()

    split_generators = {}
    for split in ("train", "val", "test"):
        split_generator = torch.Generator()
        split_generator.set_state(initial_generator_state)
        split_generators[split] = split_generator

    common = {
        "image_size": data["image_size"],
        "channels": data["channels"],
    }
    datasets = {
        split: UFDRFolderDataset(data["root"], split, **common)
        for split in ("train", "val", "test")
    }
    pin_memory = str(config["device"]).startswith("cuda")
    return {
        split: DataLoader(
            dataset,
            batch_size=data["batch_size"],
            shuffle=split == "train",
            num_workers=data["workers"],
            pin_memory=pin_memory,
            drop_last=False,
            generator=split_generators[split],
            worker_init_fn=_seed_worker,
        )
        for split, dataset in datasets.items()
    }


__all__ = ["UFDRFolderDataset", "build_dataloaders"]
