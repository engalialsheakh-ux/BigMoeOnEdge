#!/usr/bin/env python3
"""Build a deterministic, exact expert sidecar from a model and frozen layout.

The binary stores two-expert groups with gate slices first and the remaining
up/down slices second. Both the source GGUF and payload are SHA-256 hashed.
Generation is atomic: an incomplete temporary file is never the final sidecar.
"""

import argparse
import hashlib
import json
import os
import runpy
import struct
from pathlib import Path

import gguf


HEADER = struct.Struct("<8sIIIIQ32s32sQ")
LAYER = struct.Struct("<QQQ")
ENTRY = struct.Struct("<QQii")
MAGIC = b"BMOESD01"
VERSION = 1
ALIGNMENT = 4096


def aligned(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def sha256_file(path, start=0):
    digest = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        stream.seek(start)
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def verify_gguf_ranges(model, ranges, n_expert):
    tensors = {tensor.name: tensor for tensor in gguf.GGUFReader(str(model)).tensors}
    for layer, projections in enumerate(ranges):
        for index, projection in enumerate(("gate", "up", "down")):
            name = f"blk.{layer}.ffn_{projection}_exps.weight"
            tensor = tensors.get(name)
            base, unit, _ = projections[index]
            if tensor is None or tensor.data_offset != base or tensor.n_bytes != unit * n_expert:
                raise ValueError(f"access trace does not match GGUF tensor {name}")


def build(model, trace, layout_path, output):
    replay = runpy.run_path(str(Path(__file__).with_name("expert-access-replay.py")))
    n_expert, _, _, units, ranges, _ = replay["read_trace"](trace)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    n_layer = len(units)
    if layout.get("version") != VERSION or layout.get("n_layer") != n_layer or \
            layout.get("n_expert") != n_expert or len(layout.get("order", [])) != n_layer:
        raise ValueError("sidecar layout does not match access trace")
    for layer, order in enumerate(layout["order"]):
        if sorted(order) != list(range(n_expert)):
            raise ValueError(f"layer {layer} is not an expert permutation")
    model_bytes = model.stat().st_size
    for layer, projections in enumerate(ranges):
        if len(projections) != 3 or any(unit <= 0 or file_index != 0 or
                                        base + n_expert * unit > model_bytes
                                        for base, unit, file_index in projections):
            raise ValueError(f"layer {layer} is not a single-file split-three expert layout")
    verify_gguf_ranges(model, ranges, n_expert)
    model_hash = sha256_file(model)
    metadata_bytes = HEADER.size + n_layer * (LAYER.size + n_expert * ENTRY.size)
    payload_offset = aligned(metadata_bytes)
    metadata = bytearray(payload_offset)
    HEADER.pack_into(metadata, 0, MAGIC, VERSION, n_layer, n_expert, 0, model_bytes,
                     model_hash, bytes(32), payload_offset)
    cursor = HEADER.size
    for layer in range(n_layer):
        LAYER.pack_into(metadata, cursor, *(unit for _, unit, _ in ranges[layer]))
        cursor += LAYER.size + n_expert * ENTRY.size
    assert cursor == metadata_bytes

    temporary = output.with_name(output.name + ".partial")
    payload_hash = hashlib.sha256()
    with model.open("rb", buffering=8 * 1024 * 1024) as source, \
            temporary.open("wb", buffering=8 * 1024 * 1024) as target:
        target.write(metadata)
        for layer in range(n_layer):
            projections = ranges[layer]
            order = layout["order"][layer]
            table = HEADER.size + layer * (LAYER.size + n_expert * ENTRY.size) + LAYER.size
            for pair_start in range(0, n_expert, 2):
                pair = order[pair_start:pair_start + 2]
                padding = aligned(target.tell()) - target.tell()
                if padding:
                    target.write(bytes(padding))
                    payload_hash.update(bytes(padding))
                pair_offset = target.tell()
                gate_size = projections[0][1]
                rest_size = projections[1][1] + projections[2][1]
                for rank, expert in enumerate(pair):
                    gate_offset = pair_offset + rank * gate_size
                    rest_offset = pair_offset + len(pair) * gate_size + rank * rest_size
                    partner = pair[1 - rank] if len(pair) == 2 else -1
                    ENTRY.pack_into(metadata, table + expert * ENTRY.size, gate_offset,
                                    rest_offset, partner, rank)
                for projection in (0, 1, 2):
                    for expert in pair:
                        # Gate is pair-major; the second wave is expert-major.
                        if projection == 2:
                            continue
                        if projection == 1:
                            for remainder in (1, 2):
                                base, unit, _ = projections[remainder]
                                source.seek(base + expert * unit)
                                chunk = source.read(unit)
                                if len(chunk) != unit:
                                    raise ValueError("truncated GGUF expert slice")
                                target.write(chunk)
                                payload_hash.update(chunk)
                        else:
                            base, unit, _ = projections[0]
                            source.seek(base + expert * unit)
                            chunk = source.read(unit)
                            if len(chunk) != unit:
                                raise ValueError("truncated GGUF gate slice")
                            target.write(chunk)
                            payload_hash.update(chunk)
        target.flush()
        os.fsync(target.fileno())
        HEADER.pack_into(metadata, 0, MAGIC, VERSION, n_layer, n_expert, 0, model_bytes,
                         model_hash, payload_hash.digest(), payload_offset)
        target.seek(0)
        target.write(metadata)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, output)
    return model_hash.hex(), payload_hash.hexdigest(), output.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("access_trace", type=Path)
    parser.add_argument("layout_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model_hash, payload_hash, size = build(args.model, args.access_trace, args.layout_json,
                                          args.output)
    print(f"sidecar_bytes={size} model_sha256={model_hash} payload_sha256={payload_hash}")


if __name__ == "__main__":
    main()
