use clap::{Parser, ValueEnum};
use epd_dither::decompose::gray::{
    GRAYSCALE2, GRAYSCALE4, GRAYSCALE16, OffsetBlendGrayDecomposer, PureSpreadGrayDecomposer,
};
use epd_dither::decompose::naive::{EPDOPTIMIZE, NaiveDecomposer, NaiveDecomposerStrategy};
use epd_dither::decompose::octahedron::{
    NAIVE_RGB6, OctahedronDecomposer, OctahedronDecomposerAxisStrategy, SPECTRA6,
};
use epd_dither::dither::{DecomposingDitherStrategy, diffuse::ImageWriter};
use epd_dither::image_adapter::PaletteDitheringWithNoise;
use image::{ImageBuffer, ImageReader, Luma, Rgb};
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
    /// No-spread 1-D grayscale; equivalent under any of the gray decomposers.
    /// Currently routed through `PureSpreadGrayDecomposer` with spread = 0.
    Grayscale,
    /// `PureSpreadGrayDecomposer` with the given spread ratio in [0, 1].
    GrayPureSpread(f32),
    /// `OffsetBlendGrayDecomposer` with the given offset in input-space units.
    GrayOffsetBlend(f32),
}

impl DecomposeStrategy {
    const LONG_HELP: &'static str = concat!(
        "Decomposition strategy.\n\n",
        "Accepted values:\n",
        " octahedron-closest        Octahedron, pick closest axis (default)\n",
        " octahedron-furthest       Octahedron, pick furthest axis\n",
        " naive-mix                 Naive, favour mixed weights\n",
        " naive-dominant            Naive, favour dominant component\n",
        " grayscale                 1-D grayscale, no spread\n",
        " gray-pure-spread:<r>      Pure-spread grayscale, r in [0, 1]\n",
        " gray-offset-blend:<r>     Offset-blend grayscale, r in [0, 1]\n\n",
        "Examples:\n",
        " --strategy octahedron-closest\n",
        " --strategy grayscale\n",
        " --strategy gray-pure-spread:0.25\n",
    );
}

fn parse_unit_interval(s: &str, prefix: &str) -> Result<f32, String> {
    let v_str = &s[prefix.len()..];
    let v = v_str.parse::<f32>().map_err(|_| {
        format!("invalid value `{s}`: expected `{prefix}<r>` with r a number in [0, 1]")
    })?;
    if !(0.0..=1.0).contains(&v) {
        return Err(format!(
            "invalid value `{v}` for `{prefix}<r>`: must be in [0, 1]"
        ));
    }
    Ok(v)
}

impl std::str::FromStr for DecomposeStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "octahedron-closest" => Ok(Self::OctahedronClosest),
            "octahedron-furthest" => Ok(Self::OctahedronFurthest),
            "naive-mix" => Ok(Self::NaiveMix),
            "naive-dominant" => Ok(Self::NaiveDominant),
            "grayscale" => Ok(Self::Grayscale),
            _ if s.starts_with("gray-pure-spread:") => Ok(Self::GrayPureSpread(
                parse_unit_interval(s, "gray-pure-spread:")?,
            )),
            _ if s.starts_with("gray-offset-blend:") => Ok(Self::GrayOffsetBlend(
                parse_unit_interval(s, "gray-offset-blend:")?,
            )),
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
    fn as_rgb_vec(&self) -> Vec<[u8; 3]> {
        match self {
            Palette::Naive => NAIVE_RGB6.to_vec(),
            Palette::Spectra6 => SPECTRA6.to_vec(),
            Palette::Epdoptimize => EPDOPTIMIZE.to_vec(),
            Palette::Grayscale2 => GRAYSCALE2.iter().map(|&v| [v, v, v]).collect(),
            Palette::Grayscale4 => GRAYSCALE4.iter().map(|&v| [v, v, v]).collect(),
            Palette::Grayscale16 => GRAYSCALE16.iter().map(|&v| [v, v, v]).collect(),
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

/// Validate that every entry of a dither-palette is achromatic (r == g == b)
/// and that the resulting brightness levels are strictly ascending, then
/// return the levels. The output palette index has to align with the
/// `dither_palette` order, so we require sorted input rather than sorting
/// internally.
fn grayscale_levels(palette: &[Point3<f32>]) -> Vec<f32> {
    let levels: Vec<f32> = palette
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
            panic!("grayscale dither-palette must be sorted strictly ascending by brightness");
        }
    }
    levels
}

/// BT.709 luma (perceptually-weighted brightness) applied directly in sRGB
/// space — no gamma round-trip, consistent with the rest of the pipeline
/// (see notes on Yule-Nielsen optical dot gain in reflective media).
fn rgb_to_brightness(color: Rgb<f32>) -> f32 {
    let [r, g, b] = color.0;
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Flat `Vec<usize>`-backed paletted-image sink for the dither pipeline.
struct IndexedBuffer {
    data: Vec<usize>,
    width: usize,
}

impl IndexedBuffer {
    fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0; width * height],
            width,
        }
    }
}

impl ImageWriter<usize> for IndexedBuffer {
    fn put_pixel(&mut self, x: usize, y: usize, pixel: usize) {
        self.data[(y * self.width) + x] = pixel;
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
    let dither_palette_u8 = args.dither_palette.as_rgb_vec();
    println!("Dither palette used:");
    for color in &dither_palette_u8 {
        println!("  #{:02X}{:02X}{:02X},", color[0], color[1], color[2]);
    }
    let dither_palette_as_points: Vec<Point3<f32>> = dither_palette_u8
        .iter()
        .map(|c| {
            let [r, g, b] = c.map(|x| (x as f32) / 255.0);
            Point3::new(r, g, b)
        })
        .collect();
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

    let (width, height) = (input.width() as usize, input.height() as usize);
    let mut inout = PaletteDitheringWithNoise {
        image: input,
        noise_fn,
        writer: IndexedBuffer::new(width, height),
    };
    let matrix = args.diffuse.to_boxed_matrix();
    match args.strategy {
        DecomposeStrategy::OctahedronClosest => {
            let decomposer = OctahedronDecomposer::new(&dither_palette_as_points)
                .unwrap()
                .with_strategy(OctahedronDecomposerAxisStrategy::Closest);
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy::new(decomposer, color_to_point),
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
                DecomposingDitherStrategy::new(decomposer, color_to_point),
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
                DecomposingDitherStrategy::new(decomposer, color_to_point),
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
                DecomposingDitherStrategy::new(decomposer, color_to_point),
                matrix,
                &mut inout,
                true,
            );
        }
        DecomposeStrategy::Grayscale => {
            // Both gray decomposers produce identical output at parameter = 0
            // (plain bracket decomposition), but OffsetBlend takes its
            // early-out path while PureSpread still runs the full
            // asymmetric-spread arithmetic. Route through OffsetBlend.
            let levels = grayscale_levels(&dither_palette_as_points);
            let decomposer = OffsetBlendGrayDecomposer::new(levels).unwrap();
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy::new(decomposer, rgb_to_brightness),
                matrix,
                &mut inout,
                true,
            );
        }
        DecomposeStrategy::GrayPureSpread(spread_ratio) => {
            let levels = grayscale_levels(&dither_palette_as_points);
            let decomposer = PureSpreadGrayDecomposer::new(levels)
                .unwrap()
                .with_spread_ratio(spread_ratio);
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy::new(decomposer, rgb_to_brightness),
                matrix,
                &mut inout,
                true,
            );
        }
        DecomposeStrategy::GrayOffsetBlend(distance) => {
            let levels = grayscale_levels(&dither_palette_as_points);
            let decomposer = OffsetBlendGrayDecomposer::new(levels)
                .unwrap()
                .with_distance(distance);
            epd_dither::dither::diffuse::diffuse_dither(
                DecomposingDitherStrategy::new(decomposer, rgb_to_brightness),
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
        .as_rgb_vec()
        .into_iter()
        .flatten()
        .collect();
    output.set_palette(palette);
    let data: Vec<u8> = inout.writer.data.iter().map(|x| *x as u8).collect();
    let mut output = output.write_header().unwrap();
    output.write_image_data(data.as_slice()).unwrap();
    println!("Done");
}
