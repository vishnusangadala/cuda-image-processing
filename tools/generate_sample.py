#!/usr/bin/env python3
"""Generate a dependency-free PPM test image with edges and smooth gradients."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("dimensions must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as image:
        image.write(f"P6\n{args.width} {args.height}\n255\n".encode())
        for y in range(args.height):
            row = bytearray()
            for x in range(args.width):
                checker = 55 if ((x // 64) + (y // 64)) % 2 else 0
                circle = 80 if (x - args.width // 2) ** 2 + (y - args.height // 2) ** 2 < (min(args.width, args.height) // 4) ** 2 else 0
                row.extend(((x * 255 // args.width + checker) % 256,
                            (y * 255 // args.height + circle) % 256,
                            ((x + y) * 127 // max(1, args.width + args.height) + checker) % 256))
            image.write(row)


if __name__ == "__main__":
    main()
