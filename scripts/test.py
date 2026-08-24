#!/usr/bin/env python3
"""Evaluate a portable UFDR checkpoint."""

import argparse
import json
from pathlib import Path
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ufdr.engine import evaluate, load_config, resolve_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UFDR on a folder dataset")
    parser.add_argument("--config", required=True, help="path to a UFDR YAML config")
    parser.add_argument("--checkpoint", required=True, help="path to best.pt")
    args = parser.parse_args()
    try:
        config = resolve_paths(load_config(args.config), args.config)
        result = evaluate(config, args.checkpoint)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
