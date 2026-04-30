# TODO

## Documentation

### Refresh example images
The README/documentation images predate the naive-decomposer rework. Regenerate them so they reflect current output.

### Compare noise sources
Document the differences between white noise, Bayer matrices, interleaved gradient noise, and blue noise — when each is preferable and what artefacts each introduces.

## Code

### dither-rgb vs dither-decompose
Allow Floyd-Steinberg (and other diffusion methods) to operate either on the raw RGB residual or on per-component decomposition weights — expose both modes (e.g. `dither-rgb`, `dither-decompose`) so the caller can choose.

### no-alloc N-channel naive decomposer
Add `NaiveDecomposerN<T, const MAX: usize>` alongside the alloc-required `NaiveDecomposer`. Deferred during initial `Decomposer`-trait design until the gray decomposer landed; unblocked now.

## Measurement

### Recalibrate Spectra 6 / Epdoptimize
Replace the placeholder values in `decompose::octahedron::SPECTRA6` and `decompose::naive::EPDOPTIMIZE` with i1Pro-measured panel reflectance values once the unit arrives.

### Measure grayscale palette levels
`GRAYSCALE2/4/16` are linearly-spaced placeholders. Replace with measured reflectance values for the actual e-paper grayscale stages.
