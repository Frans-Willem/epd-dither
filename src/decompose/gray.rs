/// Linearly-spaced 2-level grayscale palette (placeholder; replace with
/// measured panel reflectance values — see TODO.md).
pub const GRAYSCALE2: [u8; 2] = [0, 255];

/// Linearly-spaced 4-level grayscale palette (placeholder).
pub const GRAYSCALE4: [u8; 4] = [0, 85, 170, 255];

/// Linearly-spaced 16-level grayscale palette (placeholder).
pub const GRAYSCALE16: [u8; 16] = [
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
];

use crate::decompose::Decomposer;
use crate::helpers::{partial_clamp, partial_max};
use core::ops::{Add, AddAssign, Div, Mul, Sub};
use num_traits::{One, Zero};

/// Decomposes a scalar into per-level weights for a 1D palette such as a
/// grayscale e-paper display.
///
/// Levels are supplied sorted strictly ascending. The `spread_ratio`
/// (in `[0, 1]`) controls how much weight is redistributed beyond the
/// bracketing level pair, mitigating visible banding at level boundaries.
///
/// `L` is the levels storage; any `AsRef<[T]>` works (`Vec<T>`, `[T; N]`,
/// `&[T]`, `tinyvec::ArrayVec<[T; N]>`, …) so the decomposer is usable
/// without an allocator.
pub struct GrayDecomposer<L, T> {
    levels: L,
    spread_ratio: T,
}

impl<L, T> GrayDecomposer<L, T>
where
    L: AsRef<[T]>,
    T: Clone + PartialOrd + Zero,
{
    /// Construct a decomposer for the given strictly-ascending levels.
    /// Returns `None` if `levels.len() < 2` or the levels are not sorted
    /// ascending (which also rejects palettes containing NaN levels).
    // `!(a < b)` rather than `a >= b` so NaN comparisons (which return false)
    // get rejected via the negation rather than silently admitted.
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
            spread_ratio: T::zero(),
        })
    }
}

impl<L, T> GrayDecomposer<L, T> {
    /// Set the spread ratio used by
    /// [`Decomposer::decompose_into`](super::Decomposer::decompose_into).
    /// Caller is responsible for clamping to `[0, 1]`.
    pub fn with_spread_ratio(mut self, spread_ratio: T) -> Self {
        self.spread_ratio = spread_ratio;
        self
    }
}

impl<L, T> Decomposer<T> for GrayDecomposer<L, T>
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
        let num_levels = levels.len();
        // Bracket: input ∈ [levels[left], levels[right]] with right = left + 1.
        // Inputs outside the palette range collapse onto the nearest edge
        // bracket (u clamps to 0 or 1 below). Negated comparison (rather than
        // `l >= input`) so a NaN input lands on the upper-edge collapse path
        // rather than producing weights at undefined positions.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let right = levels
            .iter()
            .position(|l| !(l < input))
            .unwrap_or(num_levels);
        // If input falls below the lowest level, just return the lowest level
        if right == 0 {
            out[right] = T::one();
            return;
        }
        // If input falls above highest level, just return the highest level
        if right >= num_levels {
            out[num_levels - 1] = T::one();
            return;
        }
        let left = right - 1;
        let value_left = levels[left].clone();
        let value_right = levels[right].clone();

        // How much of right and left we should dither together to produce this input
        let ratio_right = partial_clamp(
            (input.clone() - value_left.clone()) / (value_right.clone() - value_left.clone()),
            T::zero(),
            T::one(),
        );
        let ratio_left = T::one() - ratio_right.clone();

        let two = T::one() + T::one();
        // How much of the 'spread out' version of left we should dither in.
        let ratio_spread_left = self.spread_ratio.clone()
            * partial_max(T::zero(), ratio_left.clone() * two.clone() - T::one())
            * ratio_left.clone();
        // How much of the pure left we should dither in.
        let ratio_pure_left = ratio_left - ratio_spread_left.clone();

        out[left] += ratio_pure_left;
        if left == 0 {
            out[left] += ratio_spread_left;
        } else {
            // Asymmetric mean-preserving split: weights at levels[left-1] and
            // levels[right] sum to ratio_spread_left and have mean L[left].
            // Solving:
            //   spread_to_left_left * L[left-1] + spread_to_right * L[right] = ratio_spread_left * L[left]
            //   spread_to_left_left + spread_to_right                        = ratio_spread_left
            // gives:
            //   spread_to_left_left = ratio_spread_left * (L[right] - L[left])    / (L[right] - L[left-1])
            //   spread_to_right     = ratio_spread_left * (L[left]  - L[left-1])  / (L[right] - L[left-1])
            let value_left_left = levels[left - 1].clone();
            let span = value_right.clone() - value_left_left;
            let spread_to_left_left =
                ratio_spread_left.clone() * (value_right.clone() - value_left.clone()) / span;
            let spread_to_right = ratio_spread_left - spread_to_left_left.clone();
            out[left - 1] += spread_to_left_left;
            out[right] += spread_to_right;
        }

        // Mirror for the right endpoint.
        let ratio_spread_right = self.spread_ratio.clone()
            * partial_max(T::zero(), ratio_right.clone() * two - T::one())
            * ratio_right.clone();
        let ratio_pure_right = ratio_right - ratio_spread_right.clone();

        out[right] += ratio_pure_right;
        if right == num_levels - 1 {
            out[right] += ratio_spread_right;
        } else {
            // Mirror split: weights at levels[left] and levels[right+1] sum to
            // ratio_spread_right and have mean L[right].
            //   spread_to_left          = ratio_spread_right * (L[right+1] - L[right]) / (L[right+1] - L[left])
            //   spread_to_right_right   = ratio_spread_right * (L[right]   - L[left])  / (L[right+1] - L[left])
            let value_right_right = levels[right + 1].clone();
            let span = value_right_right - value_left.clone();
            let spread_to_right_right =
                ratio_spread_right.clone() * (value_right - value_left) / span;
            let spread_to_left = ratio_spread_right - spread_to_right_right.clone();
            out[left] += spread_to_left;
            out[right + 1] += spread_to_right_right;
        }
    }
}
