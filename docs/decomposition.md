# Decomposition

Dithering in this library is a two-stage pipeline:

1. **Decompose** each input colour into a set of non-negative weights over palette entries that sum to one — i.e. a probability distribution over palette colours whose weighted average reproduces the input.
2. **Sample** that distribution at each pixel using a noise source (blue noise, Bayer matrix, IGN, white noise) and/or diffuse the residual with Floyd-Steinberg (or other diffusion matrices).

This document covers stage 1 — the `Decomposer` implementations, when each applies, and how they resolve the ambiguity that arises when more than one valid decomposition exists.

## Why decomposition is ambiguous

A point in N-dimensional colour space has a unique barycentric decomposition over an (N+1)-vertex simplex (a triangle in 2-D, a tetrahedron in 3-D). Once the palette has more vertices than that, almost every input colour can be expressed as a weighted average of palette colours in **many different ways**, and the decomposer has to pick one.

For RGB (3-D space) and a 6-colour palette like Spectra 6:

- There are `C(6,4) = 15` ways to pick four palette colours forming a tetrahedron. A generic interior colour lies inside several of them — each gives a different valid decomposition.
- For the symmetric Spectra 6 / NaiveRGB6 octahedron there are exactly 3 "central-axis" decompositions (one per antipodal pair: K↔W, Y↔B, R↔G).

For 1-D grayscale palettes the bracketing pair `[L_i, L_{i+1}]` gives a unique linear decomposition, but we can also intentionally introduce extra weight on neighbouring levels to break up banding — that's a different kind of multiplicity (a free parameter rather than a discrete choice). As an example, in a 4 level grayscale palette (0%, 33%, 66%, 100%), you could mix to 50% with equal amounts of 33% and 66%, but you could also reach 50% with equal amounts 0% and 100%.

The picking strategy controls **what kind of dither pattern you get** for a given input colour. Two decompositions with the same mean produce visibly different output: one might mix red and green to make yellow (lots of speckle), another might just emit yellow pixels (clean but only if yellow is in the palette).

## RGB decomposers

### `OctahedronDecomposer`

For palettes whose six colours form a regular convex octahedron — three antipodal pairs sharing a centroid. `SPECTRA6` and `NAIVE_RGB6` both qualify; `EPDOPTIMIZE` does not (its black is `0x191e21`, not antipodal to its white).

Construction validates the structure (`OctahedronDecomposer::new` returns `None` if the palette is not octahedral). Internally it precomputes three `OctahedronDecomposerAxis` instances, one per antipodal pair, each treating that pair as the central axis of a square bipyramid.

Per-axis decomposition is closed-form and very cheap: project onto the axis line, then barycentric-project onto the four equatorial vertices. On a single ESP32-S3 core this runs an 800×480 `f32` image in under 5 seconds.

**Strategies** (`OctahedronDecomposerAxisStrategy`):

| Strategy        | What it does                                                                                                                  | When to use                                                                |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `Axis(i)`       | Always use axis `i`.                                                                                                          | Debugging, or when you specifically want one axis to dominate.             |
| `Closest`       | Pick the axis whose central line is closest (in 3-D RGB distance) to the input. **Default.**                                  | General use. Inputs near the K↔W axis dither in grayscale, etc.            |
| `Furthest`      | Pick the axis farthest from the input.                                                                                        | Stress-testing / artistic effect; pulls in maximally-distant palette pairs. |
| `Average`       | Average the weight vectors from all three axes (only when the input is inside the octahedron; otherwise falls back to axis 0). | Smoother spatial transitions at the cost of more colours mixed per pixel.  |

`Closest` is a good default because the axis whose central line is nearest the input is also the axis whose decomposition involves the smallest detour through the equatorial colours — i.e. the most "natural" dither.

### `NaiveDecomposer`

For arbitrary convex-hull palettes — anything where you don't have the octahedral structure. Required for `EPDOPTIMIZE`; usable on Spectra 6 too if you want a different dither character.

Enumerates **every** combination of 4 vertices (tetrahedra), 3 vertices (faces), and 2 vertices (edges) up front. For each input:

1. Find every tetrahedron the point is inside (all weights ≥ 0).
2. If any: pick one according to strategy. Done.
3. Otherwise the point is outside the convex hull of the palette: project onto the closest face or edge.

The enumeration is `O(N⁴)` in palette size and lives behind the `alloc` Cargo feature because the lists are `Vec`-backed. For the 6-colour case the tetrahedra list has 15 entries — small. The `NaiveDecomposerN<T, const MAX: usize>` no-alloc variant tracked in `TODO.md` would let this run on embedded targets too.

**Strategies** (`NaiveDecomposerStrategy`):

| Strategy         | What it does                                                                                                            | Visual effect                                                                                |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `FavorMix`       | Among containing tetrahedra, pick the one whose **largest** weight is **smallest**. **Default.**                        | Spreads weight over more palette colours → more speckle, less posterisation in flat regions. |
| `FavorDominant`  | Pick the tetrahedron whose largest weight is largest.                                                                   | Concentrates weight on one or two colours → cleaner solids, more visible level transitions.  |
| `TetraBlend(p)`  | Smooth blend over **all** containing tetrahedra, weighted by `α ∝ (∏_j w_j)^p`. `p = 0` averages equally; higher `p` concentrates on the most-interior tetrahedron. | Continuous in input (no per-pixel hard pick) at the cost of recruiting more palette colours per pixel. See `tetra-blend-research.md` for the derivation. |

`FavorMix` minimises the most-likely-sampled colour, which keeps any single palette entry from dominating; `FavorDominant` does the opposite. `TetraBlend` is qualitatively different — instead of picking one tetrahedron per pixel it blends them, removing the discrete strategy-flip discontinuity that `FavorMix`/`FavorDominant` produce inside tetrahedron-overlap regions.

## Grayscale decomposers

For 1-D palettes (an ascending list of grayscale levels). Both decomposers are storage-generic over `AsRef<[T]>` so they work without an allocator: pass a `Vec<T>`, `[T; N]`, `&[T]`, `tinyvec::ArrayVec<…>`, etc.

Both reduce to the same plain bracket decomposition (input lands between `L[left]` and `L[right]`, weights are linear in the gap) when their parameter is zero. The parameter controls how aggressively to spread weight beyond that pair to break up banding.

### `PureSpreadGrayDecomposer`

Asymmetric mean-preserving spread. The amount of weight redistributed beyond the bracket is largest when the input coincides with a palette level (when banding would otherwise be most visible — a flat plateau of one pure colour) and zero at the midpoint between two levels (where bracket dither already produces dense alternation).

`spread_ratio ∈ [0, 1]` scales the maximum redistribution. At `spread_ratio = 0` it's plain bracket decomposition. At `spread_ratio = 1`, an input exactly on level `L[i]` gets all pushed out to `L[i-1]` and `L[i+1]` (with proportions chosen so the mean stays at `L[i]`). An input exactly between two levels will not spread its endpoints. Inputs between the midpoint and a pure palette color will increase their spreading linearly.

Spreads weight across **up to three** adjacent levels.

### `OffsetBlendGrayDecomposer`

Linearly interpolates between the bracket decompositions of `input - distance/2` and `input + distance/2`, with the blend factor chosen so the combined mean lands back at `input` even when one sample point clamps to an edge level.

`distance` is the input-space separation between the two sample points; the caller is responsible for clamping. At `distance = 0` it reduces to plain bracket decomposition.

Spreads weight across **up to four** adjacent levels (vs. three for `PureSpread`), and the spread peaks symmetrically around each level rather than asymmetrically. Different visual character — try both and pick whichever looks better on your panel.

## Picking a decomposer + strategy

| Palette                            | Decomposer                                              | Notes                                                                            |
| ---------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `SPECTRA6`, `NAIVE_RGB6`           | `OctahedronDecomposer` (`Closest`)                      | Structurally optimal. Use `NaiveDecomposer` if you want the FavorMix character.  |
| `EPDOPTIMIZE`                      | `NaiveDecomposer` (`FavorMix`)                          | Octahedron decomposer would refuse the palette (or produce wrong weights).       |
| `GRAYSCALE2/4/16`                  | `OffsetBlendGrayDecomposer` or `PureSpreadGrayDecomposer` | Both work; pick by visual preference. Set parameter > 0 to attack banding.        |

The CLI exposes the same matrix via `--strategy octahedron-closest`, `--strategy naive-mix`, `--strategy gray-pure-spread:0.25`, etc. — see `dither --help`.

## See also

- The `Decomposer` trait (`src/decompose/mod.rs`) — minimal slice-based interface so hot loops can pre-allocate the weight buffer once.
- Stage 2 sampling lives in `src/dither/`. The decomposer is independent of the noise source: the same weight distribution can be dithered by any of blue noise, Bayer, IGN, or Floyd-Steinberg-style diffusion.
