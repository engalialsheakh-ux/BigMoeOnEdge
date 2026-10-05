#!/usr/bin/env python3
"""Recreate the exact 17,663-token coding-agent prompt used in this report."""

import argparse
import hashlib
import subprocess
from pathlib import Path


SOURCE_REVISION = "4b6e755:core/src/engine/session.cpp"
PREFIX = (r"You are a coding agent reviewing a local inference engine. Read the following C++ source "
          r"carefully and preserve the exact behavior of model state, expert routing, and GPU placement.\n\n")
SUFFIX = (r"\n\nTask: Write a detailed code review with numbered findings, exact function references, "
          r"and concrete fixes. Explain the state-cache correctness conditions and runtime performance risks. "
          r"Continue for approximately 1300 tokens.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = subprocess.check_output(["git", "show", SOURCE_REVISION]).decode("utf-8")
    prompt = (PREFIX + source[:65000] + SUFFIX).encode("utf-8")
    args.output.write_bytes(prompt)
    print(f"bytes={len(prompt)} sha256={hashlib.sha256(prompt).hexdigest()}")


if __name__ == "__main__":
    main()
