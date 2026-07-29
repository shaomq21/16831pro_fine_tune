#!/usr/bin/env python3
"""Point pi05 policy_preprocessor.json at a local PaliGemma tokenizer directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    args = parser.parse_args()

    ckpt = Path(args.ckpt_dir)
    tok = Path(args.tokenizer_dir).resolve()
    if not tok.is_dir():
        raise FileNotFoundError(tok)
    if not (tok / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Missing tokenizer_config.json in {tok}")

    for name in ("policy_preprocessor.json",):
        path = ckpt / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for step in data.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                step.setdefault("config", {})["tokenizer_name"] = str(tok)
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Patched {path} -> tokenizer_name={tok}")


if __name__ == "__main__":
    main()
