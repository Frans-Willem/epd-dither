# TODO

## Documentation

### Refresh example images
The README/documentation images predate the naive-decomposer rework. Regenerate them so they reflect current output.

### Compare noise sources
Document the differences between white noise, Bayer matrices, interleaved gradient noise, and blue noise — when each is preferable and what artefacts each introduces.

## Code

### Audit dither.rs vs library boundary against epd-photoframe-server
Now that `epd-photoframe-server` (sibling crate) consumes `epd-dither` as a library, walk through `src/bin/dither.rs` and identify functionality that's currently CLI-private but would be useful as library API. Likely candidates: parsing decomposition strategies from strings (the `DecomposeStrategy` enum and its `value_enum` plumbing), RGB → brightness/luminance helpers, anything else duplicated between the two crates. Move what makes sense up into `epd-dither` and reduce duplication.

## Measurement

### Measure E1001 4-level grayscale
`GRAYSCALE4` (and the other `GRAYSCALE2/16` placeholders) is linearly spaced. Drive each grayscale stage on the E1001 panel, capture spectra with the same `calibration/measure.py` workflow used for Spectra 6, and replace with measured reflectance values.
