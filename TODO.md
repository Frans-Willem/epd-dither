# TODO

## Documentation

### Refresh example images
The README/documentation images predate the naive-decomposer rework. Regenerate them so they reflect current output.

### Compare noise sources
Document the differences between white noise, Bayer matrices, interleaved gradient noise, and blue noise — when each is preferable and what artefacts each introduces.

## Measurement

### Measure E1001 4-level grayscale
`GRAYSCALE4` (and the other `GRAYSCALE2/16` placeholders) is linearly spaced. Drive each grayscale stage on the E1001 panel, capture spectra with the same `calibration/measure.py` workflow used for Spectra 6, and replace with measured reflectance values.
