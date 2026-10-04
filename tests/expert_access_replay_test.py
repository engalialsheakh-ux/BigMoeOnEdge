"""Small binary fixture with independently checked prefill eviction and reread counts."""

import importlib.util
import pathlib
import struct
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("access_replay", ROOT / "scripts" / "expert-access-replay.py")
REPLAY = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


class AccessReplayTest(unittest.TestCase):
    def test_known_trace(self):
        mib = 1024 * 1024
        batches = [
            (0, 0, (0, 1), (2, 0, 0, 0, 4 * mib)),
            (0, 1, (0,), (1, 0, 0, 1, 3 * mib)),
            (0, 0, (0,), (1, 0, 1, 1, 2 * mib)),
            (1, 1, (0,), (1, 1, 0, 0, 0)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "known.access"
            with path.open("wb") as output:
                output.write(b"BMOA" + struct.pack("<IIIQI", 3, 2, 2, 6 * mib, 1))
                for unit in (2 * mib, 3 * mib):
                    output.write(struct.pack("<Q", unit))
                    for proj in range(3):
                        output.write(struct.pack("<QQi", proj * 16 * mib, unit if proj == 0 else 0, 0))
                for phase, layer, ids, observed in batches:
                    output.write(b"B" + struct.pack("<BHIIIIIQ", phase, layer, len(ids), *observed))
                    output.write(struct.pack("<" + "i" * len(ids), *ids))
            n_expert, budget, overlap, units, ranges, parsed = REPLAY.read_trace(path)
            self.assertEqual((n_expert, budget, overlap, units), (2, 6 * mib, True, [2 * mib, 3 * mib]))
            self.assertEqual(len(ranges), 2)
            sim = REPLAY.Replay("current", n_expert, budget, overlap, units)
            for batch in parsed:
                self.assertEqual(sim.batch(batch), batch.observed)
            self.assertEqual(sim.counts[0]["rereads"], 1)
            self.assertEqual(sim.counts[1]["hits"], 1)
            self.assertEqual(REPLAY.estimate_io([0, 1], [(3584, 512 * 1024, 0), (0, 0, 0), (0, 0, 0)]),
                             (2 * 512 * 1024 + 8192, 2 * 512 * 1024 + 4096, 2, 1))


if __name__ == "__main__":
    unittest.main()
