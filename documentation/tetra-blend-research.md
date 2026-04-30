# Smooth tetrahedron blending for `NaiveDecomposer`

Research note. Background: see `TODO.md` ("Smooth tetrahedron blending in
NaiveDecomposer") and `documentation/decomposition.md`.

## Problem

For an N-colour palette in 3-D RGB the current `NaiveDecomposer` enumerates
all `C(N, 4)` tetrahedra and, per pixel, picks one containing tetrahedron via
`FavorMix` (smallest max-weight) or `FavorDominant` (largest max-weight).
This is a hard discrete choice. On a smooth gradient the picked tetrahedron
can flip from `T_i` to `T_j` across an arbitrary surface inside the overlap
region `T_i ∩ T_j`, even though both decompositions are mean-correct. The
flip is visible as a seam.

Note: at a *geometric* face shared between two tetrahedra (where `T_i` ends
and `T_j` begins) both decompositions reduce to the same face-decomposition,
so no seam appears there. Seams come from strategy switching inside overlap
volumes, not from tetra-tetra geometric boundaries.

## Mathematical setup

For each containing tetrahedron `T_i` with vertex indices `I_i ⊂ {0..N}`,
let `w_i ∈ ℝ⁴` be the barycentric weights of input `x`. Then `w_{i,j} ≥ 0`,
`Σ_j w_{i,j} = 1`, and `Σ_j w_{i,j}·p_{I_i[j]} = x`.

A convex combination
```
W = Σ_i α_i · w_i,    Σ_i α_i = 1,    α_i ≥ 0
```
is automatically a valid decomposition: non-negative, sums to one, mean equals
`x`. So *any* per-tetrahedron weighting gives a valid output; the question
is which choice gives smoothness and visually-pleasing dither.

## Choice of α

Intuition: a barycentric weight `w_{i,j}` close to zero means `x` is close
to the face of `T_i` opposite vertex `j` — the decomposition is "almost
using fewer colours" — so `T_i` should contribute less.

### Why `min` is C⁰ but not smooth

`α_i ∝ min_j w_{i,j}` formalises the intuition. It is continuous but has
kinks where the argmin-vertex changes inside `T_i`, so the output is C⁰ at
best.

### Why `product` is the right choice

```
α_i ∝ ∏_{j=1..4} w_{i,j}
```

Properties:

- **C∞ in x**: polynomial in barycentric coordinates, which are linear in `x`.
- **Vanishes on every face** of `T_i` (some `w_{i,j} → 0`).
- **Vanishes faster on edges** (two `w → 0`) and at vertices (three `w → 0`).
  Good — those are the places where multiple tetrahedra meet, and you don't
  want any one of them dominating.
- **Single-tetrahedron case**: `α_i` normalises to 1, output `= w_i` exactly.

### Sharpness knob

Generalise to
```
α_i ∝ (∏_j w_{i,j})^p,    p ∈ ℕ
```

- `p = 0`: equal weights — averages all containing tetrahedra (most diffuse).
- `p = 1`: plain product.
- `p → ∞`: peaks on the single most-interior tetrahedron — recovers a soft
  argmax / soft `FavorMix`.

Same role as `distance` for `OffsetBlendGrayDecomposer` and `spread_ratio`
for `PureSpreadGrayDecomposer`.

## Behaviour at boundaries

### Geometric face shared by `T_i` and `T_j`

Both `α_i` and `α_j` vanish linearly in distance-to-face. The α-ratio limit
is `0/0`, but the *output* `W = (α_i w_i + α_j w_j) / (α_i + α_j)` has a
clean limit because the weights on the "off-face" vertices `p_d` (in `T_i`
only) and `p_e` (in `T_j` only) are scaled by `α·w`, which vanishes as
`O(ε^{p+1})`. So `W → face-decomposition` continuously.

### Strategy-flip surface inside overlap region (the seam culprit in `FavorMix`)

`α` is a smooth function of `x` via the `w`'s, which are linear in `x`. Both
contributions are present throughout the overlap region; their ratio shifts
gradually. **No seam.**

## Cost

Per containing tetrahedron we already compute `w_i` (FavorMix needs it too).
Added work: one product (4 muls), `p` more muls for the power, accumulate
into the output buffer. Final pass: one division per palette entry. Same
`O(#containing tetrahedra)` per pixel as today.

## Open questions

1. **More palette colours per pixel.** Hard-picking limits each pixel's
   sampling support to one tetrahedron (4 colours). Blending recruits more —
   could increase visible noise if the noise source isn't dense enough.
   Compare against blue noise vs. Bayer.
2. **Outside-the-hull fallback** stays unchanged: no containing tetrahedra
   to blend.
3. **Degenerate `total = 0`**: only possible if every containing tetrahedron
   has at least one zero barycentric weight at `x` — i.e. `x` lies on a
   palette edge or vertex. Measure-zero subset; defensive code falls through
   to the existing face/edge logic.
4. **Tie-breaking with degenerate tetrahedra**: a near-coplanar tetrahedron
   has tiny volume and inflated barycentric weights; the product can blow up
   asymmetrically. `TetrahedronProjector::new` already rejects fully-degenerate
   tetrahedra. Borderline cases may want normalisation by tetrahedron volume
   — defer until measured.

## Empirical question

The math says the output is correct (mean-preserving, non-negative,
seam-free). What's left is whether the additional colour mixing helps or
hurts perceived quality at typical viewing distance. Implement, run through
`tests/regression.sh`, eyeball gradients.
