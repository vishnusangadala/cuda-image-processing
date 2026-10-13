#!/usr/bin/env python3
"""Run the CPU/CUDA comparison across operations and image resolutions."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def generate(generator: Path, image: Path, width: int, height: int) -> None:
    subprocess.run(
        [sys.executable, str(generator), str(image), "--width", str(width), "--height", str(height)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=Path("build/cuda-image-processor"))
    parser.add_argument("--csv", type=Path, default=Path("benchmark-results/results.csv"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--resolutions", nargs="+", default=["640x480", "1280x720", "1920x1080", "3840x2160"])
    args = parser.parse_args()
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"executable not found: {executable}")

    root = Path(__file__).resolve().parents[1]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.csv.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="cip-benchmark-") as temp:
        temp_dir = Path(temp)
        for resolution in args.resolutions:
            try:
                width, height = (int(part) for part in resolution.lower().split("x", 1))
            except ValueError as error:
                parser.error(f"invalid resolution {resolution!r}; expected WIDTHxHEIGHT")
            source = temp_dir / f"{width}x{height}.ppm"
            generate(root / "tools" / "generate_sample.py", source, width, height)
            for operation in ("grayscale", "gaussian", "sobel"):
                output = temp_dir / f"{width}x{height}-{operation}.png"
                command = [str(executable), "--input", str(source), "--output", str(output),
                           "--operation", operation, "--backend", "both", "--warmup", "3",
                           "--iterations", str(args.iterations), "--csv", str(args.csv)]
                print("+", " ".join(command), flush=True)
                subprocess.run(command, check=True)
    print(f"Benchmark data written to {args.csv}")


if __name__ == "__main__":
    main()
