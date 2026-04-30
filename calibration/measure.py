#!/usr/bin/env python3
"""Collect spectral reflectance measurements from a Spectra 6 panel.

Drives ArgyllCMS spotread interactively. Runs one calibration against the
instrument's matched white tile, then walks you through N reads per patch
(move the head to a different region of the patch between reads), saving
each spectrum to <out_dir>/<label>_<n>.sp. Averaging and conversion to
XYZ/Lab/sRGB is done downstream against the saved files.

Example:
  ./measure.py -o calibration/spectra6/2026-04-30 \\
      --labels K W Y R B G --reps 3
"""
import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


SPOTREAD_BASE = ["spotread", "-v", "-H"]


def run_spotread(extra_args: list[str]) -> str:
    """Run spotread, feeding a single Enter on stdin to advance any prompt."""
    res = subprocess.run(
        SPOTREAD_BASE + extra_args,
        input="\n",
        capture_output=True,
        text=True,
    )
    return res.stdout + res.stderr


def parse_instrument_info(text: str) -> dict[str, str]:
    keys = {
        "Instrument Type",
        "Serial Number",
        "Firmware version",
        "CPLD version",
        "Date manufactured",
        "U.V. filter",
        "Total lamp usage",
    }
    info = {}
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Za-z\.\? ]+?)\s*:\s+(.+?)\s*$", line)
        if not m:
            continue
        key = m.group(1).rstrip("?").strip()
        if key in keys:
            info[key] = m.group(2).strip()
    return info


def find_xyz_lab(text: str) -> tuple[float, ...] | None:
    m = re.search(
        r"Result is XYZ:\s+(\S+)\s+(\S+)\s+(\S+),\s+D50 Lab:\s+(\S+)\s+(\S+)\s+(\S+)",
        text,
    )
    return tuple(map(float, m.groups())) if m else None


def calibrate() -> dict[str, str]:
    print(">>> Place the instrument on its MATCHED white calibration tile, then press Enter.")
    input()
    out = run_spotread(["-O"])
    if "Calibration complete" not in out:
        sys.stderr.write(out)
        sys.exit("calibration failed")
    print("    calibration ok")
    return parse_instrument_info(out)


def measure(label: str, n: int, total: int, out_dir: Path) -> None:
    sp = out_dir / f"{label}_{n}.sp"
    print(f">>> Position head on patch '{label}' (read {n}/{total}); press Enter.")
    input()
    out = run_spotread(["-N", "-O", str(sp)])
    if not sp.exists():
        sys.stderr.write(out)
        sys.exit(f"measurement failed for {label} #{n}")
    vals = find_xyz_lab(out)
    if vals:
        X, Y, Z, L, a, b = vals
        print(
            f"    {sp.name}: XYZ=({X:.3f}, {Y:.3f}, {Z:.3f}) "
            f"D50 Lab=({L:.2f}, {a:.2f}, {b:.2f})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-o", "--out-dir", required=True, type=Path)
    ap.add_argument("--labels", nargs="+", default=["K", "W", "Y", "R", "B", "G"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--note", default="", help="free-text note for metadata.json")
    ap.add_argument(
        "--skip-cal",
        action="store_true",
        help="skip calibration (instrument already calibrated this session)",
    )
    args = ap.parse_args()

    if shutil.which("spotread") is None:
        sys.exit("spotread not in PATH (install argyllcms)")

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        sys.exit(f"{args.out_dir} is not empty; refusing to overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    instrument_info = {} if args.skip_cal else calibrate()

    for label in args.labels:
        for n in range(1, args.reps + 1):
            measure(label, n, args.reps, args.out_dir)

    metadata = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "labels": args.labels,
        "reps": args.reps,
        "spotread_invocation": " ".join(SPOTREAD_BASE) + " [-N] -O <fname.sp>",
        "instrument": instrument_info,
        "note": args.note,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"done — wrote {len(args.labels) * args.reps} spectra + metadata.json "
        f"to {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
