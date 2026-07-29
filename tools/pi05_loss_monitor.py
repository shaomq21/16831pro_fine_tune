#!/usr/bin/env python3
"""Poll a lerobot train log and exit 0 when loss <= target."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


def latest_loss(log_path: Path) -> tuple[int, float] | None:
    step = None
    loss = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.search(r"step:(\d+[K]?) .* loss:([\d.]+)", line)
        if not m:
            continue
        step_s, loss_s = m.group(1), m.group(2)
        step = int(step_s.replace("K", "000"))
        loss = float(loss_s)
    if step is None or loss is None:
        return None
    return step, loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--target_loss", type=float, required=True)
    parser.add_argument("--poll_sec", type=int, default=60)
    parser.add_argument("--train_pid", type=int, default=0)
    args = parser.parse_args()

    log_path = Path(args.log)
    while True:
        if args.train_pid and _pid_dead(args.train_pid):
            print("[loss_monitor] training process exited", file=sys.stderr)
            raise SystemExit(1)

        if log_path.is_file():
            row = latest_loss(log_path)
            if row is not None:
                step, loss = row
                print(f"[loss_monitor] step={step} loss={loss:.4f} target<={args.target_loss}")
                if loss <= args.target_loss:
                    print(f"[loss_monitor] target reached at step={step} loss={loss:.4f}")
                    raise SystemExit(0)
        time.sleep(args.poll_sec)


def _pid_dead(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
        return False
    except OSError:
        return True


if __name__ == "__main__":
    main()
