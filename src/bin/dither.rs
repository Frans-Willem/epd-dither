use clap::{Parser, ValueEnum};
use epd_dither::Decomposer;
use epd_dither::decompose::gray::GrayDecomposer;
use epd_dither::decompose::naive::{NaiveDecomposer, NaiveDecomposerStrategy};
use epd_dither::decompose::octahedron::{OctahedronDecomposer, OctahedronDecomposerAxisStrategy};
use image::{ImageBuffer, ImageReader, Luma, Rgb};
use nalgebra::DVector;
use nalgebra::geometry::Point3;
use rand::distr::StandardUniform;
use rand::prelude::*;

#[derive(Clone, Debug)]
enum NoiseSource {
    None,
    Bayer(Option<usize>),
    InterleavedGradient,
    White,
    File(Box<ImageBuffer<Luma<f32>, Vec<f32>>>),
}

impl NoiseSource {
    const LONG_HELP: &'static str = concat!(
        "Noise source to use.\n\n",
        "Accepted values:\n",
        " none No noise\n",
        " bayer:<N> Bayer matrix of size 2**N (usize)\n",
        " bayer Infinite Bayer pattern\n",
        " ign Interleaved Gradient Noise\n",
        " interleaved-gradient-noise Same as `ign`\n",
        " white White noise\n\n",
        "Examples:\n",
        " --noise bayer:8\n",
        " --noise bayer\n",
        " --noise ign\n",
        " --noise white\n",
    );
}

impl std::str::FromStr for NoiseSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(NoiseSource::None),
            "bayer" => Ok(NoiseSource::Bayer(None)),

            "ign" | "interleaved-gradient-noise" => Ok(NoiseSource::InterleavedGradient),

            "white" => Ok(NoiseSource::White),

            _ if s.starts_with("bayer:") => {
                let n_str = &s["bayer:".len()..];
                let n = n_str.parse::<usize>().map_err(|_| {
                    format!(
                        "invalid value `{s}`: expected `bayer:<N>` where N is a positive integer"
                    )
                })?;
                Ok(NoiseSource::Bayer(Some(n)))
            }
            _ if s.starts_with("file:") => {
                let f_str = &s["file:".len()..];
                let input = ImageReader::open(f_str)
                    .unwrap()
                    .decode()
                    .unwrap()
                    .to_luma32f();
                Ok(NoiseSource::File(Box::new(input)))
            }
            _ => Err(format!(
                "invalid value `{s}` for `--noise`\n\n{}",
                NoiseSource::LONG_HELP
            )),
        }
    }
}

#[derive(Clone, Debug)]
enum DecomposeStrategy {
    OctahedronClosest,
    OctahedronFurthest,
    NaiveMix,
    NaiveDominant,
    /// 1-D / grayscale palette decomposition with the given spread ratio in [0, 1].
    Grayscale(f32),
}

impl DecomposeStrategy {
    const LONG_HELP: &'static str = concat!(
        "Decomposition strategy.\n\n",
        "Accepted values:\n",
        " octahedron-closest      Octahedron, pick closest axis (default)\n",
        " octahedron-furthest     Octahedron, pick furthest axis\n",
        " naive-mix               Naive, favour mixed weights\n",
        " naive-dominant          Naive, favour dominant component\n",
        " grayscale               1-D grayscale, spread = 0.25 (gradient-friendly default)\n",
        " grayscale:<spread>      1-D grayscale, spread in [0, 1]\n\n",
        "Examples:\n",
        " --strategy octahedron-closest\n",
        " --strategy grayscale\n",
        " --strategy grayscale:0.3\n",
    );
}

impl std::str::FromStr for DecomposeStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "octahedron-closest" => Ok(Self::OctahedronClosest),
            "octahedron-furthest" => Ok(Self::OctahedronFurthest),
            "naive-mix" => Ok(Self::NaiveMix),
            "naive-dominant" => Ok(Self::NaiveDominant),
            "grayscale" => Ok(Self::Grayscale(0.25)),
            _ if s.starts_with("grayscale:") => {
                let v_str = &s["grayscale:".len()..];
                let v = v_str.parse::<f32>().map_err(|_| {
                    format!(
                        "invalid value `{s}`: expected `grayscale:<spread>` with spread a number in [0, 1]"
                    )
                })?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(format!("invalid spread `{v}`: must be in [0, 1]"));
                }
                Ok(Self::Grayscale(v))
            }
            _ => Err(format!(
                "invalid value `{s}` for `--strategy`\n\n{}",
                Self::LONG_HELP
            )),
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum DiffuseMethod {
    None,
    FloydSteinberg,
    JarvisJudiceAndNinke,
    Atkinson,
    Sierra,
}

impl DiffuseMethod {
    pub fn to_boxed_matrix(
        &self,
    ) -> Box<dyn epd_dither::dither::diffusion_matrix::DiffusionMatrix> {
        match self {
            DiffuseMethod::None => Box::new(epd_dither::dither::diffusion_matrix::NoDiffuse),
            DiffuseMethod::Atkinson => Box::new(epd_dither::dither::diffusion_matrix::Atkinson),
            DiffuseMethod::FloydSteinberg => {
                Box::new(epd_dither::dither::diffusion_matrix::FloydSteinberg)
            }
            DiffuseMethod::JarvisJudiceAndNinke => {
                Box::new(epd_dither::dither::diffusion_matrix::JarvisJudiceAndNinke)
            }
            DiffuseMethod::Sierra => Box::new(epd_dither::dither::diffusion_matrix::Sierra),
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum Palette {
    Naive,
    Spectra6,
    Epdoptimize,
    Grayscale2,
    Grayscale4,
    Grayscale16,
}

#[derive(Parser)]
#[command(name = "dither")]
struct Args {
    #[arg()]
    input_file: String,
    #[arg()]
    output_file: String,
    #[arg(long, value_name="NOISE",long_help=NoiseSource::LONG_HELP,default_value = "ign")]
    noise: NoiseSource,
    #[arg(long, value_name = "STRATEGY", long_help = DecomposeStrategy::LONG_HELP, default_value = "octahedron-closest")]
    strategy: DecomposeStrategy,
    #[arg(long, value_name = "DIFFUSE", default_value = "floyd-steinberg")]
    diffuse: DiffuseMethod,
    #[arg(long, value_name = "DITHER_PALETTE", default_value = "spectra6")]
    dither_palette: Palette,
    #[arg(long, value_name = "OUTPUT_PALETTE", default_value = "spectra6")]
    output_palette: Palette,
}

impl Palette {
    fn as_slice(&self) -> &[Rgb<u8>] {
        /* Ordering as in the reterminal e1002 driver */
        match self {
            Palette::Naive => &[
                Rgb([0, 0, 0]),       // Black
                Rgb([255, 255, 255]), // White
                Rgb([255, 255, 0]),   // Yellow
                Rgb([255, 0, 0]),     // Red
                Rgb([0, 0, 255]),     // Blue
                Rgb([0, 255, 0]),     // Green
            ],
            Palette::Spectra6 => &[
                Rgb([58, 0, 66]),     // Black
                Rgb([179, 208, 200]), // White
                Rgb([215, 233, 0]),   // Yellow
                Rgb([151, 38, 44]),   // Red
                Rgb([61, 38, 152]),   // Blue
                Rgb([96, 104, 86]),   // Green
            ],
            Palette::Epdoptimize => &[
                Rgb([0x19, 0x1E, 0x21]), // Black
                Rgb([0xe8, 0xe8, 0xe8]), // White
                Rgb([0xef, 0xde, 0x44]), // Yellow
                Rgb([0xb2, 0x13, 0x18]), // Red
                Rgb([0x21, 0x57, 0xba]), // Blue
                Rgb([0x12, 0x5f, 0x20]), // Green
            ],
            Palette::Grayscale2 => &[Rgb([0, 0, 0]), Rgb([255, 255, 255])],
            Palette::Grayscale4 => &[
                Rgb([0, 0, 0]),
                Rgb([85, 85, 85]),
                Rgb([170, 170, 170]),
                Rgb([255, 255, 255]),
            ],
            Palette::Grayscale16 => &[
                Rgb([0x00, 0x00, 0x00]),
                Rgb([0x11, 0x11, 0x11]),
                Rgb([0x22, 0x22, 0x22]),
                Rgb([0x33, 0x33, 0x33]),
                Rgb([0x44, 0x44, 0x44]),
                Rgb([0x55, 0x55, 0x55]),
                Rgb([0x66, 0x66, 0x66]),
                Rgb([0x77, 0x77, 0x77]),
                Rgb([0x88, 0x88, 0x88]),
                Rgb([0x99, 0x99, 0x99]),
                Rgb([0xAA, 0xAA, 0xAA]),
                Rgb([0xBB, 0xBB, 0xBB]),
                Rgb([0xCC, 0xCC, 0xCC]),
                Rgb([0xDD, 0xDD, 0xDD]),
                Rgb([0xEE, 0xEE, 0xEE]),
                Rgb([0xFF, 0xFF, 0xFF]),
            ],
        }
    }
}

#[allow(dead_code)]
enum SpectraColors {
    Black = 0,
    White = 1,
    Blue = 2,
    Green = 3,
    Red = 4,
    Yellow = 5,
}

#[allow(dead_code)]
fn color_to_point(color: Rgb<f32>) -> Point3<f32> {
    let [r, g, b] = color.0;
    Point3::new(r, g, b)
}

/// BT.709 luma (perceptually-weighted brightness) applied directly in sRGB
/// space — no gamma round-trip, consistent with the rest of the pipeline
/// (see notes on Yule-Nielsen optical dot gain in reflective media).
fn rgb_to_brightness(color: Rgb<f32>) -> f32 {
    let [r, g, b] = color.0;
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

struct PaletteDitheringWithNoise<I: image::GenericImage, F: Fn(usize, usize) -> Option<f32>> {
    image: I,
    noise_fn: F,
    target: Vec<usize>,
}

impl<I: image::GenericImage, F: Fn(usize, usize) -> Option<f32>>
    epd_dither::dither::diffuse::ImageSize for PaletteDitheringWithNoise<I, F>
{
    fn width(&self) -> usize {
        self.image.width() as usize
    }
    fn height(&self) -> usize {
        self.image.height() as usize
    }
}

impl<I: image::GenericImage, F: Fn(usize, usize) -> Option<f32>>
    epd_dither::dither::diffuse::ImageReader<(I::Pixel, Option<f32>)>
    for PaletteDitheringWithNoise<I, F>
{
    fn get_pixel(&self, x: usize, y: usize) -> (I::Pixel, Option<f32>) {
        (
            self.image.get_pixel(x as u32, y as u32),
            (self.noise_fn)(x, y),
        )
    }
}

impl<I: image::GenericImage, F: Fn(usize, usize) -> Option<f32>>
    epd_dither::dither::diffuse::ImageWriter<usize> for PaletteDitheringWithNoise<I, F>
{
    fn put_pixel(&mut self, x: usize, y: usize, pixel: usize) {
        self.target.resize(
            self.image.width() as usize * self.image.height() as usize,
            0,
        );
        self.target[(y * self.image.width() as usize) + x] = pixel;
    }
}

struct DecomposingDitherStrategy<D, F> {
    decomposer: D,
    convert: F,
}

#[derive(Clone, Default)]
struct DecomposedQuantizationError(Option<DVector<f32>>);

impl core::ops::Mul<usize> for DecomposedQuantizationError {
    type Output = Self;
    fn mul(self, rhs: usize) -> Self {
        Self(self.0.map(|x| x * (rhs as f32)))
    }
}

impl core::ops::Div<usize> for DecomposedQuantizationError {
    type Output = Self;
    fn div(self, rhs: usize) -> Self {
        Self(self.0.map(|x| x / (rhs as f32)))
    }
}

impl core::ops::AddAssign<DecomposedQuantizationError> for DecomposedQuantizationError {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = match (core::mem::take(&mut self.0), rhs.0) {
            (a, None) => a,
            (None, b) => b,
            (Some(a), Some(b)) => Some(a + b),
        }
    }
}

impl<D, F> epd_dither::dither::diffuse::PixelStrategy for DecomposingDitherStrategy<D, F>
where
    D: Decomposer<f32>,
    F: Fn(Rgb<f32>) -> D::Input,
{
    type Source = (Rgb<f32>, Option<f32>); // Take both a pixel and an optional noise
    type Target = usize;
    type QuantizationError = DecomposedQuantizationError;

    fn quantize(
        &self,
        source: Self::Source,
        error: Self::QuantizationError,
    ) -> (Self::Target, Self::QuantizationError) {
        let (source, noise) = source;
        let mut decomposed = DVector::zeros(self.decomposer.palette_size());
        self.decomposer
            .decompose_into(&(self.convert)(source), decomposed.as_mut_slice());
        let decomposed = match error.0 {
            None => decomposed,
            Some(error) => decomposed + error,
        };
        let decomposed_clipped = decomposed.map(|x| if x < 0.0 { 0.0 } else { x });
        let decomposed_clipped_sum = decomposed_clipped.sum();
        let index = if let Some(noise) = noise
            && decomposed_clipped_sum > 0.0
        {
            let mut noise = noise * decomposed_clipped_sum;
            let mut index: usize = 0;
            while index + 1 < decomposed_clipped.nrows() && noise >= decomposed_clipped[index] {
                noise -= decomposed_clipped[index];
                index += 1;
            }
            index
        } else {
            decomposed.argmax().0
        };
        // Turn decomposed into
        let mut error = decomposed;
        error[index] -= 1.0;
        (index, DecomposedQuantizationError(Some(error)))
    }
}

fn main() {
    let args = Args::parse();
    println!("Opening image");
    let input = ImageReader::open(args.input_file)
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb32f();
    println!("Opened image");
    // TODO: Allow dither and output palette to be specified
    let dither_palette_u8 = args.dither_palette.as_slice();
    println!("Dither palette used:");
    for color in dither_palette_u8 {
        println!("  #{:02X}{:02X}{:02X},", color.0[0], color.0[1], color.0[2]);
    }
    let dither_palette_f32 = dither_palette_u8
        .iter()
        .map(|c| Rgb(c.0.map(|x| (x as f32) / 255.0)));
    let dither_palette_as_points: Vec<Point3<f32>> =
        dither_palette_f32.map(color_to_point).collect();
    let noise_fn = |x, y| match args.noise {
        NoiseSource::Bayer(Some(max_depth)) => Some(epd_dither::noise::bayer(x, y, max_depth)),
        NoiseSource::Bayer(None) => Some(epd_dither::noise::bayer_inf(x, y)),
        NoiseSource::InterleavedGradient => Some(epd_dither::noise::interleaved_gradient_noise(
            x as f32, y as f32,
        )),
        NoiseSource::White => Some(rand::rng().sample(StandardUniform)),
        NoiseSource::File(ref f) => {
            Some(f.get_pixel(x as u32 % f.width(), y as u32 % f.height()).0[0])
        }
        NoiseSource::None => None,
    };

    let mut inout = PaletteDitheringWithNoise {
        image: input,
        noise_fn,
        target: Vec::new(),
    };
    let matrix = args.diffuse.to_boxed_matrix();
    match args.strategy {
        DecomposeStrategy::OctahedronClosest => {
            let decomposer = OctahedronDecomposer::new(&dither_palette_as_points)
                .unwrap()
                .with_strategy(OctahedronDecomposerAxisStrategy::Closest);
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy {
                    decomposer,
                    convert: color_to_point,
                },
                matrix,
                &mut inout,
                true,
            );
        }
        DecomposeStrategy::OctahedronFurthest => {
            let decomposer = OctahedronDecomposer::new(&dither_palette_as_points)
                .unwrap()
                .with_strategy(OctahedronDecomposerAxisStrategy::Furthest);
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy {
                    decomposer,
                    convert: color_to_point,
                },
                matrix,
                &mut inout,
                true,
            );
        }
        DecomposeStrategy::NaiveMix => {
            let decomposer = NaiveDecomposer::new(dither_palette_as_points.as_slice())
                .unwrap()
                .with_strategy(NaiveDecomposerStrategy::FavorMix);
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy {
                    decomposer,
                    convert: color_to_point,
                },
                matrix,
                &mut inout,
                true,
            );
        }
        DecomposeStrategy::NaiveDominant => {
            let decomposer = NaiveDecomposer::new(dither_palette_as_points.as_slice())
                .unwrap()
                .with_strategy(NaiveDecomposerStrategy::FavorDominant);
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy {
                    decomposer,
                    convert: color_to_point,
                },
                matrix,
                &mut inout,
                true,
            );
        }
        DecomposeStrategy::Grayscale(spread_ratio) => {
            // Validate r == g == b for every dither-palette entry and that the
            // resulting brightness levels are strictly ascending. The output
            // palette index has to align with `dither_palette` order, so we
            // require sorted input rather than sorting internally.
            let levels: Vec<f32> = dither_palette_as_points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if !(p.x == p.y && p.y == p.z) {
                        panic!(
                            "grayscale strategy requires r == g == b for every dither-palette entry; entry {i} = {p:?} is not achromatic"
                        );
                    }
                    p.x
                })
                .collect();
            for w in levels.windows(2) {
                if w[0] >= w[1] {
                    panic!(
                        "grayscale dither-palette must be sorted strictly ascending by brightness"
                    );
                }
            }
            let decomposer = GrayDecomposer::new(levels)
                .unwrap()
                .with_spread_ratio(spread_ratio);
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy {
                    decomposer,
                    convert: rgb_to_brightness,
                },
                matrix,
                &mut inout,
                true,
            );
        }
    }
    let mut output = png::Encoder::new(
        std::io::BufWriter::new(std::fs::File::create(args.output_file).unwrap()),
        inout.image.width(),
        inout.image.height(),
    );
    output.set_color(png::ColorType::Indexed);
    output.set_depth(png::BitDepth::Eight);
    let palette: Vec<u8> = args
        .output_palette
        .as_slice()
        .iter()
        .flat_map(|rgb| rgb.0)
        .collect();
    output.set_palette(palette);
    let data: Vec<u8> = inout.target.iter().map(|x| *x as u8).collect();
    let mut output = output.write_header().unwrap();
    output.write_image_data(data.as_slice()).unwrap();
    println!("Done");
}
