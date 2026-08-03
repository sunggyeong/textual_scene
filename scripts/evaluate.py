from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from textual_scene.config import load_config
from textual_scene.evaluator import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an order experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config file.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory, for example outputs/name/checkpoints/last.")
    args = parser.parse_args()
    result = run_evaluation(load_config(args.config), checkpoint=args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
