"""Generate labelled controls and manipulations without external datasets."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fourierforensics.synthetic import variants


def generate(output: Path, count: int = 24, seed: int = 1000) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    output.mkdir(parents=True, exist_ok=False)
    (output / "Au").mkdir()
    (output / "Tp").mkdir()
    (output / "masks").mkdir()
    rows = []
    for i in range(count):
        for operation, (pixels, data) in variants(seed + i).items():
            label = int(operation not in ("original", "single_jpeg"))
            relative = (
                Path("Tp" if label else "Au") / f"{seed + i}_{operation}.{'jpg' if data else 'png'}"
            )
            if data:
                (output / relative).write_bytes(data)
            else:
                Image.fromarray(pixels).save(output / relative)
            rows.append(
                {
                    "path": relative.as_posix(),
                    "label": label,
                    "group": str(seed + i),
                    "operation": operation,
                }
            )
            if operation == "splice":
                mask = np.zeros(pixels.shape[:2], dtype=np.uint8)
                size = len(mask)
                mask[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 255
                Image.fromarray(mask).save(output / "masks" / f"{seed + i}.png")
    (output / "manifest.json").write_text(
        json.dumps({"seed": seed, "count": count, "samples": rows}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    generate(args.output, args.count, args.seed)
