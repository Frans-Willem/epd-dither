# TODO

## Documentation

### Refresh example images
The README/documentation images predate the naive-decomposer rework. Regenerate them so they reflect current output.

### Compare noise sources
Document the differences between white noise, Bayer matrices, interleaved gradient noise, and blue noise — when each is preferable and what artefacts each introduces.

## Measurement

### Measure E1001 4-level grayscale
`GRAYSCALE4` (and the other `GRAYSCALE2/16` placeholders) is linearly spaced. Drive each grayscale stage on the E1001 panel, capture spectra with the same `calibration/measure.py` workflow used for Spectra 6, and replace with measured reflectance values.

### Collect dither test data to research better decomposition
Build a measurement pipeline that maps dither vectors (decomposed colors) to measured panel colors, then fit a forward model and invert it for a principled decomposer.

- **Firmware**: special build of the reterminal_e100x/epd-photoframe firmware that accepts a dither vector over UART (e.g. 6 integers summing to 64 for Spectra 6), renders it on the panel as an 8×8 Bayer-tiled patch, and replies on UART when the refresh completes.
- **Host script**: Python driver that iterates over dither vectors, triggers the firmware, takes a spectrometer reading per patch, and stores `(dither_vector, spectrum/XYZ)` samples.
- **Forward fit**: fit a function from dither vector → XYZ (or another colorspace) from the collected samples.
- **Inverse / decomposer**: investigate inverting the forward model — given a target XYZ, produce a dither vector. Likely path: find or construct an intermediate colorspace in which dithering acts as a linear mix, decompose linearly there, then map back. Empirically sRGB is close but not theoretically justified; the deviation is probably dot-gain or a related reflective-halftone effect (cf. Yule-Nielsen).
