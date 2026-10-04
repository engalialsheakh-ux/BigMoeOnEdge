"""Extract actual routed expert GGUF offsets for the isolated storage probe."""

import argparse
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("access_trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=5000)
    args = parser.parse_args()
    replay = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "expert-access-replay.py"))
    _, _, _, _, ranges, batches = replay["read_trace"](args.access_trace)
    offsets = []
    seen = set()
    for batch in batches:
        if batch.phase != 0:
            continue
        for expert in batch.ids:
            for base, unit, _ in ranges[batch.layer]:
                if not unit:
                    continue
                offset = base + expert * unit
                if offset not in seen:
                    offsets.append(offset)
                    seen.add(offset)
                    if len(offsets) >= args.count:
                        args.output.write_text("\n".join(map(str, offsets)) + "\n", encoding="ascii")
                        return
    args.output.write_text("\n".join(map(str, offsets)) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
