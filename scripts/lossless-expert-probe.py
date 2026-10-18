#!/usr/bin/env python3
"""Measure one lossless compression sample from stored trunk expert bytes."""

import argparse
import time
import zlib
from pathlib import Path

import gguf


LAYERS = (0, 7, 23, 39)
EXPERTS = (0, 17, 31, 63, 95, 127, 191, 255)
PROJECTIONS = ("gate", "up", "down")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="existing production GGUF; read-only")
    args = parser.parse_args()
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(str(args.model)).tensors}
    ranges = []
    for layer in LAYERS:
        for projection in PROJECTIONS:
            name = f"blk.{layer}.ffn_{projection}_exps.weight"
            tensor = tensors[name]
            n_expert = int(tensor.shape[-1])
            if max(EXPERTS) >= n_expert or tensor.n_bytes % n_expert:
                raise ValueError(f"unexpected expert layout for {name}")
            unit = int(tensor.n_bytes) // n_expert
            for expert in EXPERTS:
                ranges.append((int(tensor.data_offset) + expert * unit, unit))

    original = compressed = 0
    decompression_seconds = 0.0
    verified = True
    with args.model.open("rb", buffering=0) as model:
        for offset, size in ranges:
            model.seek(offset)
            source = model.read(size)
            if len(source) != size:
                raise IOError(f"short GGUF read at {offset}")
            packed = zlib.compress(source, level=1)
            start = time.perf_counter()
            restored = zlib.decompress(packed)
            decompression_seconds += time.perf_counter() - start
            verified &= restored == source
            original += len(source)
            compressed += len(packed)

    print(f"sample: {len(LAYERS)} trunk layers x {len(EXPERTS)} experts x {len(PROJECTIONS)} projections")
    print(f"original_bytes={original} compressed_bytes={compressed} ratio={original / compressed:.4f}")
    print(f"decompression_mib_s={original / decompression_seconds / 1048576:.2f} bitwise_verified={verified}")
    if not verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
