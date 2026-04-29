use crate::decompose::Decomposer;
use crate::helpers::partial_clamp;
use core::ops::{Add, AddAssign, Div, Mul, Sub};
use num_traits::{One, Zero};

/// Decomposes a scalar into per-level weights for a 1D palette by linearly
/// interpolating between the bracket decompositions of `input - offset` and
/// `input + offset`, choosing the interpolation factor so the combined mean
/// lands back at `input`.
///
/// Each bracket decomposition `B(x)` is mean-preserving — its weights yield
/// effective mean `E(x) = x` for `x` in `[levels[0], levels[last]]`, or
/// clamp to the nearest edge level otherwise. The output is
/// `ratio_lower_br · B(input-offset) + ratio_upper_br · B(input+offset)`
/// with `ratio_lower_br + ratio_upper_br = 1` and the two ratios chosen to
/// satisfy `ratio_lower_br · E_lower + ratio_upper_br · E_upper = input`.
/// So the combined mean equals `input` exactly whenever `input` itself is
/// in range, even if one of `input ± offset` was clamped. Inputs outside
/// the palette range collapse onto the nearest edge level — same edge
/// behaviour as [`super::PureSpreadGrayDecomposer`].
///
/// Compared to PureSpread, this spreads dither weight across up to four
/// adjacent levels (vs. three) and the spread peaks symmetrically around
/// each level rather than asymmetrically with mean preservation. `offset` is
/// in input-space units; caller is responsible for clamping.
///
/// Levels are supplied sorted strictly ascending. `L` is the levels storage;
/// any `AsRef<[T]>` works (`Vec<T>`, `[T; N]`, `&[T]`,
/// `tinyvec::ArrayVec<[T; N]>`, …) so the decomposer is usable without an
/// allocator.
pub struct OffsetBlendGrayDecomposer<L, T> {
    levels: L,
    offset: T,
}

impl<L, T> OffsetBlendGrayDecomposer<L, T>
where
    L: AsRef<[T]>,
    T: Clone + PartialOrd + Zero,
{
    /// Construct a decomposer for the given strictly-ascending levels.
    /// Returns `None` if `levels.len() < 2` or the levels are not sorted
    /// ascending (also rejects palettes containing NaN levels).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(levels: L) -> Option<Self> {
        let slice = levels.as_ref();
        if slice.len() < 2 {
            return None;
        }
        for w in slice.windows(2) {
            if !(w[0] < w[1]) {
                return None;
            }
        }
        Some(Self {
            levels,
            offset: T::zero(),
        })
    }
}

impl<L, T> OffsetBlendGrayDecomposer<L, T> {
    /// Set the offset used by
    /// [`Decomposer::decompose_into`](super::super::Decomposer::decompose_into).
    /// Caller is responsible for clamping to the input-space range.
    pub fn with_offset(mut self, offset: T) -> Self {
        self.offset = offset;
        self
    }
}

impl<L, T> Decomposer<T> for OffsetBlendGrayDecomposer<L, T>
where
    L: AsRef<[T]>,
    T: Clone + PartialOrd + Zero + One,
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    T: AddAssign,
{
    type Input = T;

    fn palette_size(&self) -> usize {
        self.levels.as_ref().len()
    }

    fn decompose_into(&self, input: &T, out: &mut [T]) {
        for slot in out.iter_mut() {
            *slot = T::zero();
        }
        let levels = self.levels.as_ref();

        // Quick out: in case of no offset or if the input is outside the
        // levels, just collapse to the single bracket case.
        let input_br = bracket(levels, input);
        if self.offset <= T::zero() || input_br.left == input_br.right {
            out[input_br.left] += input_br.ratio_left;
            out[input_br.right] += input_br.ratio_right;
            return;
        }

        let lower_br = bracket(levels, &(input.clone() - self.offset.clone()));
        let upper_br = bracket(levels, &(input.clone() + self.offset.clone()));

        // Pick `ratio_lower_br` so that
        //   ratio_lower_br · E_lower + ratio_upper_br · E_upper = input,
        // where ratio_upper_br = 1 - ratio_lower_br. When the two effective
        // means coincide (both decompositions clamped to the same edge —
        // can't happen for an in-range input with offset > 0, but kept as
        // a defensive guard) the bracket decomps are identical and any
        // value gives the same output; pick 1/2 for symmetry. Defensive
        // clamp handles floating-point rounding pushing the ratio outside
        // [0, 1].
        let ratio_lower_br = if lower_br.effective == upper_br.effective {
            T::one() / (T::one() + T::one())
        } else {
            partial_clamp(
                (input.clone() - upper_br.effective.clone())
                    / (lower_br.effective - upper_br.effective),
                T::zero(),
                T::one(),
            )
        };
        let ratio_upper_br = T::one() - ratio_lower_br.clone();

        out[lower_br.left] += ratio_lower_br.clone() * lower_br.ratio_left;
        out[lower_br.right] += ratio_lower_br * lower_br.ratio_right;
        out[upper_br.left] += ratio_upper_br.clone() * upper_br.ratio_left;
        out[upper_br.right] += ratio_upper_br * upper_br.ratio_right;
    }
}

/// Bracket decomposition of `input` against `levels`: the surrounding pair
/// of levels and the linear ratio between them. Inputs outside
/// `[levels[0], levels[last]]` collapse to a degenerate bracket at the
/// nearest edge level (`left == right`, `ratio_right == 1`). `effective`
/// is `ratio_left·levels[left] + ratio_right·levels[right]` — equal to
/// `input` for in-range inputs, clamped to the edge level otherwise.
struct Bracket<T> {
    left: usize,
    right: usize,
    ratio_left: T,
    ratio_right: T,
    effective: T,
}

#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn bracket<T>(levels: &[T], input: &T) -> Bracket<T>
where
    T: Clone + PartialOrd + Zero + One,
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let num_levels = levels.len();
    // Negated comparison (`!(l < input)` rather than `l >= input`) so a NaN
    // input lands on the upper-edge collapse path rather than producing
    // weights at undefined positions.
    let right = levels
        .iter()
        .position(|l| !(l < input))
        .unwrap_or(num_levels);
    if right == 0 {
        return Bracket {
            left: 0,
            right: 0,
            ratio_left: T::zero(),
            ratio_right: T::one(),
            effective: levels[0].clone(),
        };
    }
    if right >= num_levels {
        let last = num_levels - 1;
        return Bracket {
            left: last,
            right: last,
            ratio_left: T::zero(),
            ratio_right: T::one(),
            effective: levels[last].clone(),
        };
    }
    let left = right - 1;
    let value_left = levels[left].clone();
    let value_right = levels[right].clone();
    let ratio_right = partial_clamp(
        (input.clone() - value_left.clone()) / (value_right.clone() - value_left.clone()),
        T::zero(),
        T::one(),
    );
    let ratio_left = T::one() - ratio_right.clone();
    let effective = ratio_left.clone() * value_left + ratio_right.clone() * value_right;
    Bracket {
        left,
        right,
        ratio_left,
        ratio_right,
        effective,
    }
}
