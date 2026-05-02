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

/// Library-grade enum equivalent of the binary's `--noise` argument:
/// names a positional noise source. The factory layer (see
/// [`crate::factory`]) turns each variant into a concrete
/// `Fn(usize, usize) -> f32`. Variants that depend on a runtime image
/// asset (`File`, `Blue`) are gated behind the `image` feature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoiseSource {
    /// No noise; the strategy falls back to dominant-component selection.
    None,
    /// Deterministic Bayer matrix. `Some(n)` selects the 2^n × 2^n matrix;
    /// `None` is the infinite (recursively-extended) variant.
    Bayer(Option<usize>),
    InterleavedGradient,
    #[cfg(feature = "rand")]
    White,
    /// External noise image at the given path. Loaded by the factory.
    #[cfg(feature = "image")]
    File(alloc::string::String),
    /// Built-in blue-noise tile bundled with the crate.
    #[cfg(feature = "image")]
    Blue,
}

impl NoiseSource {
    pub const LONG_HELP: &'static str = concat!(
        "Noise source to use.\n\n",
        "Accepted values:\n",
        " none           No noise\n",
        " bayer:<N>      Bayer matrix of size 2^N\n",
        " bayer          Infinite Bayer pattern\n",
        " ign            Interleaved Gradient Noise\n",
        " white          White noise (requires `rand` feature)\n",
        " file:<PATH>    External noise image (requires `image` feature)\n",
        " blue           Built-in blue-noise tile (requires `image` feature)\n",
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidNoiseSource;

impl core::fmt::Display for InvalidNoiseSource {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("invalid noise-source name")
    }
}

impl core::error::Error for InvalidNoiseSource {}

impl core::str::FromStr for NoiseSource {
    type Err = InvalidNoiseSource;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "bayer" => Ok(Self::Bayer(None)),
            "ign" | "interleaved-gradient-noise" => Ok(Self::InterleavedGradient),
            #[cfg(feature = "rand")]
            "white" => Ok(Self::White),
            #[cfg(feature = "image")]
            "blue" => Ok(Self::Blue),
            _ if s.starts_with("bayer:") => {
                let n = s["bayer:".len()..]
                    .parse::<usize>()
                    .map_err(|_| InvalidNoiseSource)?;
                Ok(Self::Bayer(Some(n)))
            }
            #[cfg(feature = "image")]
            _ if s.starts_with("file:") => {
                Ok(Self::File(alloc::string::String::from(&s["file:".len()..])))
            }
            _ => Err(InvalidNoiseSource),
        }
    }
}
