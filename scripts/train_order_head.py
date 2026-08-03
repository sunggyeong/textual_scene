from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textual_scene.config import load_config
from textual_scene.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an order-head experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")
    args = parser.parse_args()
    result = run_training(load_config(args.config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
