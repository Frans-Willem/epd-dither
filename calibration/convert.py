#!/usr/bin/env python3
"""Average measured panel spectra and convert to sRGB palettes for one
or more viewing illuminants.

Reads <calib_dir>/metadata.json to find the patch labels and rep count,
parses each `<label>_<n>.sp` file, averages the per-rep reflectance
spectra, and computes:

  - XYZ under the chosen illuminant + CIE 1931 2° observer
  - L*a*b* (referenced to the same illuminant)
  - sRGB 8-bit (chromatic adaptation: source illuminant → D65, Bradford)

Output: per-illuminant table on stdout, plus an HTML report with swatches
and per-patch reflectance plots written to <calib_dir>/report.html.
"""
import argparse
import datetime
import html
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import colour
    except ImportError:
        sys.exit("requires `colour-science`; pip install -r requirements.txt")
warnings.filterwarnings("ignore", category=colour.utilities.ColourRuntimeWarning)
warnings.filterwarnings("ignore", category=colour.utilities.ColourUsageWarning)


@dataclass
class Patch:
    label: str
    wavelengths: np.ndarray
    mean_refl: np.ndarray
    std_refl: np.ndarray
    nreps: int
    sd: object  # colour.SpectralDistribution


@dataclass
class PatchResult:
    label: str
    XYZ: np.ndarray
    Lab: np.ndarray
    rgb: tuple[int, int, int]


def parse_sp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelengths_nm, reflectance_0_to_1)."""
    text = path.read_text()
    bands = int(re.search(r'SPECTRAL_BANDS\s+"(\d+)"', text).group(1))
    start = float(re.search(r'SPECTRAL_START_NM\s+"([\d.]+)"', text).group(1))
    end = float(re.search(r'SPECTRAL_END_NM\s+"([\d.]+)"', text).group(1))
    norm = float(re.search(r'SPECTRAL_NORM\s+"([\d.]+)"', text).group(1))
    block = re.search(r"\nBEGIN_DATA\n(.*?)\nEND_DATA\b", text, re.DOTALL).group(1)
    values = np.array(block.split(), dtype=float)
    if values.size != bands:
        raise ValueError(f"{path}: expected {bands} values, got {values.size}")
    return np.linspace(start, end, bands), values / norm


def load_patch(label: str, files: list[Path]) -> Patch:
    arrs = [parse_sp(p) for p in files]
    wls = arrs[0][0]
    for w, _ in arrs[1:]:
        if not np.allclose(w, wls):
            raise ValueError(f"wavelength grids differ across reps for {label}")
    refls = np.stack([r for _, r in arrs])
    mean_r = refls.mean(axis=0)
    std_r = refls.std(axis=0)
    sd = colour.SpectralDistribution(dict(zip(wls, mean_r)), name=label)
    return Patch(label, wls, mean_r, std_r, len(files), sd)


def xyz_to_srgb_u8(
    XYZ_0_100: np.ndarray, illuminant_xy: np.ndarray
) -> tuple[int, int, int]:
    rgb = colour.XYZ_to_sRGB(
        XYZ_0_100 / 100,
        illuminant=illuminant_xy,
        chromatic_adaptation_transform="Bradford",
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    return tuple(int(round(float(c) * 255)) for c in rgb)


def compute_under(
    patch: Patch, ill_sd, cmfs, ill_xy
) -> PatchResult:
    XYZ = colour.sd_to_XYZ(patch.sd, cmfs, ill_sd, method="Integration")
    rgb = xyz_to_srgb_u8(XYZ, ill_xy)
    Lab = np.asarray(colour.XYZ_to_Lab(XYZ / 100, ill_xy))
    return PatchResult(patch.label, np.asarray(XYZ), Lab, rgb)


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Quick perceptual luminance for picking text color."""
    r, g, b = (c / 255 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def spectrum_svg(
    patch: Patch, width: int = 320, height: int = 90, pad: int = 4
) -> str:
    wls = patch.wavelengths
    refl = patch.mean_refl
    wmin, wmax = 380, 730
    rmax = 1.0
    pts = []
    for w, r in zip(wls, refl):
        if w < wmin or w > wmax:
            continue
        x = pad + (w - wmin) / (wmax - wmin) * (width - 2 * pad)
        y = (height - pad) - r / rmax * (height - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    path = " ".join(pts)
    # tick marks at 400, 500, 600, 700 nm
    ticks = []
    for nm in (400, 500, 600, 700):
        x = pad + (nm - wmin) / (wmax - wmin) * (width - 2 * pad)
        ticks.append(
            f'<line x1="{x:.1f}" y1="{height-pad}" x2="{x:.1f}" y2="{height-pad+2}" stroke="#888"/>'
            f'<text x="{x:.1f}" y="{height-pad+11}" font-size="8" text-anchor="middle" fill="#666">{nm}</text>'
        )
    # 50% grid line
    y50 = (height - pad) - 0.5 * (height - 2 * pad)
    return (
        f'<svg width="{width}" height="{height + 12}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#fff;border:1px solid #ddd">'
        f'<line x1="{pad}" y1="{y50}" x2="{width - pad}" y2="{y50}" stroke="#eee"/>'
        f'<polyline fill="none" stroke="#222" stroke-width="1" points="{path}"/>'
        + "".join(ticks) +
        "</svg>"
    )


def render_html(
    calib_dir: Path,
    metadata: dict,
    patches: list[Patch],
    by_illuminant: dict[str, list[PatchResult]],
) -> str:
    started = metadata.get("started", "?")
    note = metadata.get("note", "") or ""
    instr = metadata.get("instrument", {})
    instr_str = " · ".join(f"{k}: {v}" for k, v in instr.items()) or "(no instrument metadata)"

    illuminant_sections = []
    for ill_name, results in by_illuminant.items():
        cards = []
        for patch, res in zip(patches, results):
            r, g, b = res.rgb
            hex6 = f"#{r:02x}{g:02x}{b:02x}"
            text_color = "#fff" if relative_luminance(res.rgb) < 0.5 else "#111"
            mean_std = float(patch.std_refl.mean())
            sigma_pct = mean_std * 100
            svg = spectrum_svg(patch)
            cards.append(f"""
                <div class="card">
                  <div class="swatch" style="background:{hex6};color:{text_color}">
                    <div class="label">{html.escape(patch.label)}</div>
                    <div class="hex">{hex6}</div>
                  </div>
                  <table class="meta">
                    <tr><th>sRGB</th><td>({r}, {g}, {b})</td></tr>
                    <tr><th>L* a* b*</th><td>({res.Lab[0]:.2f}, {res.Lab[1]:.2f}, {res.Lab[2]:.2f})</td></tr>
                    <tr><th>XYZ</th><td>({res.XYZ[0]:.3f}, {res.XYZ[1]:.3f}, {res.XYZ[2]:.3f})</td></tr>
                    <tr><th>within-rep σ</th><td>{sigma_pct:.3f}% over {patch.nreps} reads</td></tr>
                  </table>
                  <div class="spectrum">{svg}</div>
                </div>
            """)
        rust = _rust_block(ill_name, results)
        illuminant_sections.append(f"""
            <section>
              <h2>Illuminant {html.escape(ill_name)}</h2>
              <p class="note">CIE 1931 2° observer, sRGB via Bradford CAT to D65,
              integration on the as-measured spectral grid. Lab is referenced to the same illuminant.</p>
              <div class="grid">{''.join(cards)}</div>
              <pre class="rust"><code>{html.escape(rust)}</code></pre>
            </section>
        """)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spectra 6 calibration · {html.escape(calib_dir.name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2em; max-width: 1400px; color: #111; }}
  h1 {{ font-weight: 500; margin-bottom: 0.2em; }}
  h2 {{ font-weight: 500; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: 0.3em; }}
  .meta-block {{ color: #555; font-size: 0.9em; line-height: 1.5; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 1em; margin-top: 1em; }}
  .card {{ width: 340px; border: 1px solid #ddd; padding: 0.6em; border-radius: 4px; background: #fafafa; }}
  .swatch {{ height: 100px; display: flex; flex-direction: column; justify-content: center; align-items: center;
            border-radius: 3px; font-family: ui-monospace, monospace; }}
  .swatch .label {{ font-size: 1.4em; font-weight: 600; }}
  .swatch .hex {{ font-size: 0.95em; opacity: 0.85; }}
  table.meta {{ width: 100%; margin-top: 0.5em; border-collapse: collapse; font-size: 0.85em; }}
  table.meta th {{ text-align: left; font-weight: 500; color: #555; padding: 2px 0; width: 40%; }}
  table.meta td {{ font-family: ui-monospace, monospace; }}
  .spectrum {{ margin-top: 0.5em; }}
  .note {{ color: #666; font-size: 0.9em; }}
  pre.rust {{ background: #f4f4f4; padding: 1em; border-radius: 4px; overflow-x: auto; font-size: 0.85em; }}
  code {{ font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
  <h1>Spectra 6 calibration</h1>
  <div class="meta-block">
    <div>Source: <code>{html.escape(str(calib_dir))}</code></div>
    <div>Captured: {html.escape(started)}</div>
    <div>Instrument: {html.escape(instr_str)}</div>
    {f'<div>Note: {html.escape(note)}</div>' if note else ''}
    <div>Rendered: {html.escape(datetime.datetime.now().isoformat(timespec='seconds'))}</div>
  </div>
  {''.join(illuminant_sections)}
</body>
</html>
"""


def _rust_block(ill_name: str, results: list[PatchResult]) -> str:
    lines = [
        f"// Spectra 6, viewing illuminant {ill_name}, CIE 1931 2°.",
        f"pub const SPECTRA6_{ill_name}: [[u8; 3]; {len(results)}] = [",
    ]
    for res in results:
        r, g, b = res.rgb
        lines.append(f"    [{r:>3}, {g:>3}, {b:>3}], // {res.label}")
    lines.append("];")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("calib_dir", type=Path)
    ap.add_argument(
        "--illuminants",
        nargs="+",
        default=["D50", "D65"],
        help="any colour-science illuminant key (e.g. D50 D65 A F11)",
    )
    ap.add_argument(
        "--html",
        type=Path,
        default=None,
        help="HTML report path (default: <calib_dir>/report.html; use '-' to skip)",
    )
    args = ap.parse_args()

    metadata = json.loads((args.calib_dir / "metadata.json").read_text())
    labels = metadata["labels"]

    patches = []
    for label in labels:
        files = sorted(args.calib_dir.glob(f"{label}_*.sp"))
        if not files:
            sys.exit(f"no .sp files for label {label}")
        patches.append(load_patch(label, files))

    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    chromaticities = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]

    by_illuminant: dict[str, list[PatchResult]] = {}
    for ill_name in args.illuminants:
        try:
            ill_sd = colour.SDS_ILLUMINANTS[ill_name]
            ill_xy = chromaticities[ill_name]
        except KeyError:
            sys.exit(f"unknown illuminant {ill_name!r}")
        by_illuminant[ill_name] = [
            compute_under(p, ill_sd, cmfs, ill_xy) for p in patches
        ]

    for ill_name, results in by_illuminant.items():
        print(
            f"== Illuminant {ill_name} (CIE 1931 2°, "
            f"sRGB via Bradford CAT to D65) =="
        )
        print(
            f"  {'label':<8} {'sRGB':<18} {'hex':<9} "
            f"{'Lab':<24} {'mean σ':<8} reps"
        )
        for patch, res in zip(patches, results):
            r, g, b = res.rgb
            mean_std = float(patch.std_refl.mean())
            print(
                f"  {patch.label:<8} ({r:>3}, {g:>3}, {b:>3})    "
                f"#{r:02x}{g:02x}{b:02x}   "
                f"({res.Lab[0]:6.2f}, {res.Lab[1]:6.2f}, {res.Lab[2]:6.2f})    "
                f"{mean_std:.4f}   {patch.nreps}"
            )
        print()

    print("== Rust ==")
    for ill_name, results in by_illuminant.items():
        print(_rust_block(ill_name, results))
        print()

    html_path = args.html
    if html_path is None:
        html_path = args.calib_dir / "report.html"
    if str(html_path) != "-":
        html_text = render_html(args.calib_dir, metadata, patches, by_illuminant)
        html_path.write_text(html_text)
        print(f"wrote HTML report to {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
