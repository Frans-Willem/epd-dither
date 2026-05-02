pub trait DiffusionMatrix {
    fn divisor(&self) -> usize;
    fn targets(&self) -> &[(isize, usize, usize)];
}

#[cfg(feature = "alloc")]
impl DiffusionMatrix for alloc::boxed::Box<dyn DiffusionMatrix> {
    fn divisor(&self) -> usize {
        self.as_ref().divisor()
    }
    fn targets(&self) -> &[(isize, usize, usize)] {
        self.as_ref().targets()
    }
}

/// Borrowed-data diffusion matrix: pairs a divisor with a `'static` slice
/// of `(dx, dy, weight)` targets. Lets us hand a built-in matrix back as
/// a concrete value without allocating or committing to dyn dispatch.
#[derive(Clone, Copy, Debug)]
pub struct RefDiffusionMatrix(pub usize, pub &'static [(isize, usize, usize)]);

impl DiffusionMatrix for RefDiffusionMatrix {
    fn divisor(&self) -> usize {
        self.0
    }
    fn targets(&self) -> &[(isize, usize, usize)] {
        self.1
    }
}

// Built-in diffusion matrices. Each kernel is shown in its conventional
// raster-scan layout: `*` is the current pixel, weights to the right and
// below are diffused; the divisor below normalises them. The
// `(dx, dy, w)` triples are the same data, laid out so the column of
// each `dx` lines up across rows.

/// No diffusion at all.
pub const NO_DIFFUSE: RefDiffusionMatrix = RefDiffusionMatrix(1, &[]);

#[rustfmt::skip]
/// Floyd-Steinberg, divisor 16:
/// ```text
///       *  7
///    3  5  1
/// ```
pub const FLOYD_STEINBERG: RefDiffusionMatrix = RefDiffusionMatrix(16, &[
                            ( 1, 0, 7),
    (-1, 1, 3), ( 0, 1, 5), ( 1, 1, 1),
]);

#[rustfmt::skip]
/// Jarvis, Judice and Ninke, divisor 48:
/// ```text
///          *  7  5
///    3  5  7  5  3
///    1  3  5  3  1
/// ```
pub const JARVIS_JUDICE_AND_NINKE: RefDiffusionMatrix = RefDiffusionMatrix(48, &[
                                        ( 1, 0, 7), ( 2, 0, 5),
    (-2, 1, 3), (-1, 1, 5), ( 0, 1, 7), ( 1, 1, 5), ( 2, 1, 3),
    (-2, 2, 1), (-1, 2, 3), ( 0, 2, 5), ( 1, 2, 3), ( 2, 2, 1),
]);

#[rustfmt::skip]
/// Atkinson, divisor 8:
/// ```text
///       *  1  1
///    1  1  1
///       1
/// ```
pub const ATKINSON: RefDiffusionMatrix = RefDiffusionMatrix(8, &[
                            ( 1, 0, 1), ( 2, 0, 1),
    (-1, 1, 1), ( 0, 1, 1), ( 1, 1, 1),
                ( 0, 2, 1),
]);

#[rustfmt::skip]
/// Sierra, divisor 32:
/// ```text
///          *  5  3
///    2  4  5  4  2
///       2  3  2
/// ```
pub const SIERRA: RefDiffusionMatrix = RefDiffusionMatrix(32, &[
                                        ( 1, 0, 5), ( 2, 0, 3),
    (-2, 1, 2), (-1, 1, 4), ( 0, 1, 5), ( 1, 1, 4), ( 2, 1, 2),
                (-1, 2, 2), ( 0, 2, 3), ( 1, 2, 2),
]);

/// Library-grade enum equivalent of the binary's `--diffuse` argument:
/// names a built-in diffusion matrix. Use [`to_matrix`](Self::to_matrix)
/// to get an opaque `impl DiffusionMatrix` (currently a
/// [`RefDiffusionMatrix`]); the concrete representation is intentionally
/// hidden so we can swap it later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffuseMethod {
    None,
    FloydSteinberg,
    JarvisJudiceAndNinke,
    Atkinson,
    Sierra,
}

impl DiffuseMethod {
    pub const LONG_HELP: &'static str = concat!(
        "Diffusion matrix to use.\n\n",
        "Accepted values:\n",
        " none                    No diffusion\n",
        " floyd-steinberg         Floyd-Steinberg (default)\n",
        " jarvis-judice-and-ninke Jarvis, Judice, and Ninke\n",
        " atkinson                Atkinson\n",
        " sierra                  Sierra\n",
    );

    pub fn to_matrix(&self) -> impl DiffusionMatrix + use<> {
        match self {
            Self::None => NO_DIFFUSE,
            Self::FloydSteinberg => FLOYD_STEINBERG,
            Self::JarvisJudiceAndNinke => JARVIS_JUDICE_AND_NINKE,
            Self::Atkinson => ATKINSON,
            Self::Sierra => SIERRA,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidDiffuseMethod;

impl core::fmt::Display for InvalidDiffuseMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("invalid diffusion-method name")
    }
}

impl core::error::Error for InvalidDiffuseMethod {}

impl core::str::FromStr for DiffuseMethod {
    type Err = InvalidDiffuseMethod;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "floyd-steinberg" => Ok(Self::FloydSteinberg),
            "jarvis-judice-and-ninke" => Ok(Self::JarvisJudiceAndNinke),
            "atkinson" => Ok(Self::Atkinson),
            "sierra" => Ok(Self::Sierra),
            _ => Err(InvalidDiffuseMethod),
        }
    }
}
