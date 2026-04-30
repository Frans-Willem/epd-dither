#!/usr/bin/env python3
"""Generate a palette-indexed PNG of solid-color test patches.

Patches are laid out left-to-right, top-to-bottom in palette-index order, so
the driver-order palette you pass on the command line is the order you'll
encounter the patches when measuring with the spectrometer.

Example (Spectra 6 placeholder palette):
  ./make_calibration_patches.py 1872 1404 \\
      --palette 3a0042 b3d0c8 d7e900 97262c 3d2698 606856 \\
      -o patches.png
"""
import argparse
import math
import sys

import numpy as np
from PIL import Image


def parse_hex(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    if len(s) != 6:
        raise argparse.ArgumentTypeError(f"expected 6 hex digits, got {s!r}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("width", type=int)
    ap.add_argument("height", type=int)
    ap.add_argument(
        "--palette",
        nargs="+",
        type=parse_hex,
        required=True,
        metavar="RRGGBB",
        help="palette colors as hex, in driver order",
    )
    ap.add_argument(
        "--cols",
        type=int,
        default=None,
        help="grid columns (default: ceil(sqrt(N)))",
    )
    ap.add_argument("-o", "--output", default="patches.png")
    args = ap.parse_args()

    n = len(args.palette)
    if n > 256:
        ap.error("palette must be <= 256 entries")
    cols = args.cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Edges chosen so leftover pixels go to the last row/col rather than
    # leaving an unindexed strip.
    col_edges = [args.width * c // cols for c in range(cols + 1)]
    row_edges = [args.height * r // rows for r in range(rows + 1)]

    idx = np.zeros((args.height, args.width), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        idx[row_edges[r] : row_edges[r + 1], col_edges[c] : col_edges[c + 1]] = i

    img = Image.fromarray(idx, mode="P")
    flat = [v for rgb in args.palette for v in rgb]
    flat.extend([0] * (768 - len(flat)))
    img.putpalette(flat)
    img.save(args.output, format="PNG", optimize=True)

    print(
        f"wrote {args.output}: {args.width}x{args.height}, "
        f"{n} patches in {cols}x{rows} grid"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
