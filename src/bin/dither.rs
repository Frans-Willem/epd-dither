use clap::{Parser, ValueEnum};
use epd_dither::decompose::gray::{
    GRAYSCALE2, GRAYSCALE4, GRAYSCALE16, OffsetBlendGrayDecomposer, PureSpreadGrayDecomposer,
};
use epd_dither::decompose::naive::{EPDOPTIMIZE, NaiveDecomposer, NaiveDecomposerStrategy};
use epd_dither::decompose::octahedron::{
    NAIVE_RGB6, OctahedronDecomposer, OctahedronDecomposerAxisStrategy,
};
use epd_dither::dither::{DecomposingDitherStrategy, ImageSize, ImageSplit, ImageWriter};
use epd_dither::spectra6::{
    SPECTRA6, SPECTRA6_D50, SPECTRA6_D50_ADJUSTED, SPECTRA6_D50_BPC50_ADJUSTED,
    SPECTRA6_D50_BPC75_ADJUSTED, SPECTRA6_D50_BPC80_ADJUSTED, SPECTRA6_D50_BPC90_ADJUSTED,
    SPECTRA6_D50_BPC100_ADJUSTED, SPECTRA6_D65, SPECTRA6_D65_ADJUSTED, SPECTRA6_D65_BPC50_ADJUSTED,
    SPECTRA6_D65_BPC75_ADJUSTED, SPECTRA6_D65_BPC80_ADJUSTED, SPECTRA6_D65_BPC90_ADJUSTED,
    SPECTRA6_D65_BPC100_ADJUSTED,
};
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
    /// `NaiveDecomposer` with `TetraBlend(p)` — smooth blend over containing
    /// tetrahedra weighted by `(∏_j w_j)^p`.
    NaiveBlend(u32),
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
        " naive-blend[:<p>]         Naive, smooth blend (default p=1)\n",
        " grayscale                 1-D grayscale, no spread\n",
        " gray-pure-spread:<r>      Pure-spread grayscale, r in [0, 1]\n",
        " gray-offset-blend:<r>     Offset-blend grayscale, r in [0, 1]\n\n",
        "Examples:\n",
        " --strategy octahedron-closest\n",
        " --strategy grayscale\n",
        " --strategy gray-pure-spread:0.25\n",
        " --strategy naive-blend:2\n",
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
            "naive-blend" => Ok(Self::NaiveBlend(1)),
            _ if s.starts_with("naive-blend:") => {
                let p_str = &s["naive-blend:".len()..];
                let p = p_str.parse::<u32>().map_err(|_| {
                    format!(
                        "invalid value `{s}`: expected `naive-blend:<p>` where p is a non-negative integer"
                    )
                })?;
                Ok(Self::NaiveBlend(p))
            }
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
    Spectra6D50,
    Spectra6D50Adjusted,
    Spectra6D50Bpc50Adjusted,
    Spectra6D50Bpc75Adjusted,
    Spectra6D50Bpc80Adjusted,
    Spectra6D50Bpc90Adjusted,
    Spectra6D50Bpc100Adjusted,
    Spectra6D65,
    Spectra6D65Adjusted,
    Spectra6D65Bpc50Adjusted,
    Spectra6D65Bpc75Adjusted,
    Spectra6D65Bpc80Adjusted,
    Spectra6D65Bpc90Adjusted,
    Spectra6D65Bpc100Adjusted,
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
            Palette::Spectra6D50 => SPECTRA6_D50.to_vec(),
            Palette::Spectra6D50Adjusted => SPECTRA6_D50_ADJUSTED.to_vec(),
            Palette::Spectra6D50Bpc50Adjusted => SPECTRA6_D50_BPC50_ADJUSTED.to_vec(),
            Palette::Spectra6D50Bpc75Adjusted => SPECTRA6_D50_BPC75_ADJUSTED.to_vec(),
            Palette::Spectra6D50Bpc80Adjusted => SPECTRA6_D50_BPC80_ADJUSTED.to_vec(),
            Palette::Spectra6D50Bpc90Adjusted => SPECTRA6_D50_BPC90_ADJUSTED.to_vec(),
            Palette::Spectra6D50Bpc100Adjusted => SPECTRA6_D50_BPC100_ADJUSTED.to_vec(),
            Palette::Spectra6D65 => SPECTRA6_D65.to_vec(),
            Palette::Spectra6D65Adjusted => SPECTRA6_D65_ADJUSTED.to_vec(),
            Palette::Spectra6D65Bpc50Adjusted => SPECTRA6_D65_BPC50_ADJUSTED.to_vec(),
            Palette::Spectra6D65Bpc75Adjusted => SPECTRA6_D65_BPC75_ADJUSTED.to_vec(),
            Palette::Spectra6D65Bpc80Adjusted => SPECTRA6_D65_BPC80_ADJUSTED.to_vec(),
            Palette::Spectra6D65Bpc90Adjusted => SPECTRA6_D65_BPC90_ADJUSTED.to_vec(),
            Palette::Spectra6D65Bpc100Adjusted => SPECTRA6_D65_BPC100_ADJUSTED.to_vec(),
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
    height: usize,
}

impl IndexedBuffer {
    fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0; width * height],
            width,
            height,
        }
    }
}

impl ImageSize for IndexedBuffer {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
}

impl ImageWriter<usize> for IndexedBuffer {
    fn put_pixel(&mut self, x: usize, y: usize, pixel: usize) {
        self.data[(y * self.width) + x] = pixel;
    }
}

/// Run `diffuse_dither` for one `DecomposingDitherStrategy`, branching on
/// whether a noise function is configured. The `with_noise` / no-noise
/// branches yield different concrete types for the strategy's `N`
/// parameter, so the split is unavoidable here — pulling it into a helper
/// keeps the eight strategy arms below from doubling.
fn run_decomposing<D, F, Src, N, M, IO>(
    decomposer: D,
    convert: F,
    noise_fn: Option<N>,
    matrix: &M,
    inout: &mut IO,
) where
    D: epd_dither::Decomposer<f32>,
    F: Fn(Src) -> D::Input,
    N: Fn(usize, usize) -> f32,
    M: epd_dither::dither::diffusion_matrix::DiffusionMatrix + ?Sized,
    IO: epd_dither::dither::ImageSize
        + epd_dither::dither::ImageReader<Src>
        + epd_dither::dither::ImageWriter<usize>,
{
    let strategy = DecomposingDitherStrategy::new(decomposer, convert);
    match noise_fn {
        Some(n) => epd_dither::dither::diffuse::diffuse_dither(
            &strategy.with_noise(n),
            matrix,
            inout,
            true,
        ),
        None => epd_dither::dither::diffuse::diffuse_dither(&strategy, matrix, inout, true),
    }
}

/// Match over `DecomposeStrategy` and dispatch to `run_decomposing` with
/// the appropriate decomposer + convert function for each variant. Lifted
/// out of `main` so the noise selection above can supply a single generic
/// `N` to the entire eight-arm strategy match in one place.
fn run_strategy<N, M, IO>(
    strategy_choice: &DecomposeStrategy,
    palette: &[Point3<f32>],
    noise_fn: Option<N>,
    matrix: &M,
    inout: &mut IO,
) where
    N: Fn(usize, usize) -> f32,
    M: epd_dither::dither::diffusion_matrix::DiffusionMatrix + ?Sized,
    IO: epd_dither::dither::ImageSize
        + epd_dither::dither::ImageReader<Rgb<f32>>
        + epd_dither::dither::ImageWriter<usize>,
{
    match *strategy_choice {
        DecomposeStrategy::OctahedronClosest => {
            let decomposer = OctahedronDecomposer::new(palette)
                .unwrap()
                .with_strategy(OctahedronDecomposerAxisStrategy::Closest);
            run_decomposing(decomposer, color_to_point, noise_fn, matrix, inout);
        }
        DecomposeStrategy::OctahedronFurthest => {
            let decomposer = OctahedronDecomposer::new(palette)
                .unwrap()
                .with_strategy(OctahedronDecomposerAxisStrategy::Furthest);
            run_decomposing(decomposer, color_to_point, noise_fn, matrix, inout);
        }
        DecomposeStrategy::NaiveMix => {
            let decomposer = NaiveDecomposer::new(palette)
                .unwrap()
                .with_strategy(NaiveDecomposerStrategy::FavorMix);
            run_decomposing(decomposer, color_to_point, noise_fn, matrix, inout);
        }
        DecomposeStrategy::NaiveDominant => {
            let decomposer = NaiveDecomposer::new(palette)
                .unwrap()
                .with_strategy(NaiveDecomposerStrategy::FavorDominant);
            run_decomposing(decomposer, color_to_point, noise_fn, matrix, inout);
        }
        DecomposeStrategy::NaiveBlend(power) => {
            let decomposer = NaiveDecomposer::new(palette)
                .unwrap()
                .with_strategy(NaiveDecomposerStrategy::TetraBlend(power));
            run_decomposing(decomposer, color_to_point, noise_fn, matrix, inout);
        }
        DecomposeStrategy::Grayscale => {
            // Both gray decomposers produce identical output at parameter = 0
            // (plain bracket decomposition), but OffsetBlend takes its
            // early-out path while PureSpread still runs the full
            // asymmetric-spread arithmetic. Route through OffsetBlend.
            let levels = grayscale_levels(palette);
            let decomposer = OffsetBlendGrayDecomposer::new(levels).unwrap();
            run_decomposing(decomposer, rgb_to_brightness, noise_fn, matrix, inout);
        }
        DecomposeStrategy::GrayPureSpread(spread_ratio) => {
            let levels = grayscale_levels(palette);
            let decomposer = PureSpreadGrayDecomposer::new(levels)
                .unwrap()
                .with_spread_ratio(spread_ratio);
            run_decomposing(decomposer, rgb_to_brightness, noise_fn, matrix, inout);
        }
        DecomposeStrategy::GrayOffsetBlend(distance) => {
            let levels = grayscale_levels(palette);
            let decomposer = OffsetBlendGrayDecomposer::new(levels)
                .unwrap()
                .with_distance(distance);
            run_decomposing(decomposer, rgb_to_brightness, noise_fn, matrix, inout);
        }
    }
}

fn main() {
    let args = Args::parse();
    println!("Opening image");
    let input = image::ImageReader::open(args.input_file)
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
    let (width, height) = (input.width() as usize, input.height() as usize);
    let output_width = input.width();
    let output_height = input.height();
    let mut inout = ImageSplit::new(input, IndexedBuffer::new(width, height)).unwrap();
    let matrix = args.diffuse.to_boxed_matrix();
    match args.noise {
        NoiseSource::Bayer(Some(max_depth)) => run_strategy(
            &args.strategy,
            &dither_palette_as_points,
            Some(move |x: usize, y: usize| epd_dither::noise::bayer(x, y, max_depth)),
            &matrix,
            &mut inout,
        ),
        NoiseSource::Bayer(None) => run_strategy(
            &args.strategy,
            &dither_palette_as_points,
            Some(|x: usize, y: usize| epd_dither::noise::bayer_inf(x, y)),
            &matrix,
            &mut inout,
        ),
        NoiseSource::InterleavedGradient => run_strategy(
            &args.strategy,
            &dither_palette_as_points,
            Some(|x: usize, y: usize| {
                epd_dither::noise::interleaved_gradient_noise(x as f32, y as f32)
            }),
            &matrix,
            &mut inout,
        ),
        NoiseSource::White => run_strategy(
            &args.strategy,
            &dither_palette_as_points,
            Some(|_x: usize, _y: usize| rand::rng().sample::<f32, _>(StandardUniform)),
            &matrix,
            &mut inout,
        ),
        NoiseSource::File(f) => run_strategy(
            &args.strategy,
            &dither_palette_as_points,
            Some(move |x: usize, y: usize| {
                f.get_pixel(x as u32 % f.width(), y as u32 % f.height()).0[0]
            }),
            &matrix,
            &mut inout,
        ),
        NoiseSource::None => run_strategy::<fn(usize, usize) -> f32, _, _>(
            &args.strategy,
            &dither_palette_as_points,
            None,
            &matrix,
            &mut inout,
        ),
    }
    let mut output = png::Encoder::new(
        std::io::BufWriter::new(std::fs::File::create(args.output_file).unwrap()),
        output_width,
        output_height,
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
