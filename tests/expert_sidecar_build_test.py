"""Exact byte-layout and fingerprint gate for the derived sidecar builder."""

import argparse
import hashlib
import json
import runpy
import struct
import tempfile
import unittest
from pathlib import Path


class SidecarBuildTest(unittest.TestCase):
    def test_pair_layout_preserves_every_projection(self):
        builder = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                     "scripts" / "expert-sidecar-build.py"))
        # The byte-layout fixture is intentionally a raw tensor region. The CTest
        # fixture below exercises the GGUF directory check on a real tiny model.
        builder["build"].__globals__["verify_gguf_ranges"] = lambda *_: None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "source.gguf"
            trace = root / "source.access"
            layout = root / "layout.json"
            output = root / "derived.bmoeside"
            n_expert = 4
            slice_size = 4096
            bases = (0, n_expert * slice_size, 2 * n_expert * slice_size)
            source = bytearray()
            for projection in range(3):
                for expert in range(n_expert):
                    source.extend(bytes([projection * 16 + expert]) * slice_size)
            model.write_bytes(source)
            with trace.open("wb") as stream:
                stream.write(b"BMOA")
                stream.write(struct.pack("<IIIQI", 3, 1, n_expert, 4400 * 1024 * 1024, 1))
                stream.write(struct.pack("<Q", 3 * slice_size))
                for base in bases:
                    stream.write(struct.pack("<QQi", base, slice_size, 0))
            layout.write_text(json.dumps({"version": 1, "n_layer": 1, "n_expert": n_expert,
                                          "order": [[2, 0, 3, 1]]}), encoding="utf-8")
            builder["build"](model, trace, layout, output)
            data = output.read_bytes()
            header = builder["HEADER"].unpack_from(data)
            self.assertEqual(header[0], b"BMOESD01")
            self.assertEqual(header[6], hashlib.sha256(source).digest())
            self.assertEqual(header[7], hashlib.sha256(data[header[8]:]).digest())
            entry_base = builder["HEADER"].size + builder["LAYER"].size
            for expert in range(n_expert):
                gate, rest, partner, rank = builder["ENTRY"].unpack_from(
                    data, entry_base + expert * builder["ENTRY"].size)
                self.assertEqual(data[gate:gate + slice_size],
                                 source[bases[0] + expert * slice_size:
                                        bases[0] + (expert + 1) * slice_size])
                self.assertEqual(data[rest:rest + slice_size],
                                 source[bases[1] + expert * slice_size:
                                        bases[1] + (expert + 1) * slice_size])
                self.assertEqual(data[rest + slice_size:rest + 2 * slice_size],
                                 source[bases[2] + expert * slice_size:
                                        bases[2] + (expert + 1) * slice_size])
                self.assertEqual(rank, 0 if expert in (2, 3) else 1)
                self.assertEqual(partner, {2: 0, 0: 2, 3: 1, 1: 3}[expert])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(SidecarBuildTest))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.model and args.trace and args.output:
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        replay = runpy.run_path(str(scripts / "expert-access-replay.py"))
        n_expert, _, _, units, _, _ = replay["read_trace"](args.trace)
        with tempfile.TemporaryDirectory() as temporary:
            layout = Path(temporary) / "layout.json"
            layout.write_text(json.dumps({"version": 1, "n_layer": len(units), "n_expert": n_expert,
                                          "order": [list(range(n_expert)) for _ in units]}),
                              encoding="utf-8")
            builder = runpy.run_path(str(scripts / "expert-sidecar-build.py"))
            builder["build"](args.model, args.trace, layout, args.output)
