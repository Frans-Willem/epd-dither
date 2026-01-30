use num_traits::float::FloatCore;
use num_traits::identities::Zero;
use num_traits::zero;

pub fn interleaved_gradient_noise<F>(x: F, y: F) -> F
where
    F: FloatCore + From<f32>,
{
    // InterleavedGradientNoise[x_, y_] := FractionalPart[52.9829189*FractionalPart[0.06711056*x + 0.00583715*y]]
    let inner1: F = (x * (0.06711056).into()) + (y * (0.00583715).into());
    let inner2: F = inner1.fract() * 52.983_917.into();
    inner2.fract()
}

const BAYER_MATRIX: [[f32; 2]; 2] = [[0.0, 2.0], [3.0, 1.0]];

pub fn bayer_inf<F>(x: usize, y: usize) -> F
where
    F: From<f32> + FloatCore + Zero,
{
    let base_multiplier: F = (0.25).into();
    let mut ret: F = zero();
    let mut x = x;
    let mut y = y;
    let mut multiplier = base_multiplier;
    while x > 0 || y > 0 {
        ret = ret + (multiplier * BAYER_MATRIX[y % 2][x % 2].into());
        x /= 2;
        y /= 2;
        multiplier = multiplier * base_multiplier;
    }
    ret
}

pub fn bayer<F>(x: usize, y: usize, max_depth: usize) -> F
where
    F: From<f32> + FloatCore + Zero,
{
    let base_multiplier: F = (0.25).into();
    let mut ret: F = zero();
    let mut x = x;
    let mut y = y;
    let mut max_depth = max_depth;
    let mut multiplier = base_multiplier;
    while max_depth > 0 && (x > 0 || y > 0) {
        ret = ret + (multiplier * BAYER_MATRIX[y % 2][x % 2].into());
        x /= 2;
        y /= 2;
        max_depth -= 1;
        multiplier = multiplier * base_multiplier;
    }
    ret
}
