use clap::{Parser, ValueEnum};
use epd_dither::decompose::gray::{
    GRAYSCALE2, GRAYSCALE4, GRAYSCALE16, OffsetBlendGrayDecomposer, PureSpreadGrayDecomposer,
};
use epd_dither::decompose::naive::{EPDOPTIMIZE, NaiveDecomposer, NaiveDecomposerStrategy};
use epd_dither::decompose::octahedron::{
    NAIVE_RGB6, OctahedronDecomposer, OctahedronDecomposerAxisStrategy,
};
use epd_dither::dither::{
    BundledDitherer, DecomposingDitherStrategy, DynDitherer, ImageSize, ImageSplit, ImageWriter,
};
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
    pub fn to_dynamic_matrix(&self) -> epd_dither::dither::diffusion_matrix::DynamicDiffusionMatrix {
        use epd_dither::dither::diffusion_matrix::{
            Atkinson, DynamicDiffusionMatrix, FloydSteinberg, JarvisJudiceAndNinke, NoDiffuse,
            Sierra,
        };
        match self {
            DiffuseMethod::None => DynamicDiffusionMatrix::new(NoDiffuse),
            DiffuseMethod::Atkinson => DynamicDiffusionMatrix::new(Atkinson),
            DiffuseMethod::FloydSteinberg => DynamicDiffusionMatrix::new(FloydSteinberg),
            DiffuseMethod::JarvisJudiceAndNinke => DynamicDiffusionMatrix::new(JarvisJudiceAndNinke),
            DiffuseMethod::Sierra => DynamicDiffusionMatrix::new(Sierra),
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

/// BT.709 luma (perceptually-weighted brightness) applied directly in sRGB
/// space — no gamma round-trip, consistent with the rest of the pipeline
/// (see notes on Yule-Nielsen optical dot gain in reflective media).
fn rgb_to_brightness(color: Rgb<f32>) -> f32 {
    let [r, g, b] = color.0;
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// True iff every entry is achromatic (r == g == b) and the entries are
/// strictly ascending in brightness. Required for any grayscale dither
/// strategy: the output palette index has to align with the input order,
/// so we want the caller to pass a sorted achromatic palette rather than
/// silently project chromatic colours via Rec.709 luma or sort behind
/// their back.
fn verify_grayscale_palette(p: &[Rgb<u8>]) -> bool {
    p.iter().all(|c| c.0[0] == c.0[1] && c.0[1] == c.0[2])
        && p.windows(2).all(|w| w[0].0[0] < w[1].0[0])
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

/// Concrete reader/writer pair the binary actually uses. `BundledDitherer`
/// is type-erased to `Box<dyn DynDitherer<InOutType>>` against this alias
/// so the build phase reduces to a nested match and the dither call itself
/// is one virtual dispatch.
type InOutType = ImageSplit<ImageBuffer<Rgb<f32>, Vec<f32>>, IndexedBuffer>;
use epd_dither::dither::diffusion_matrix::DynamicDiffusionMatrix;

/// Box a `DecomposingDitherStrategy` + matrix as a `DynDitherer`,
/// branching once on whether noise is configured. The `with_noise` /
/// no-noise branches produce different concrete `N` parameters; the
/// `Box<dyn DynDitherer<_>>` return unifies them.
fn build_decomposing<D, F, Src, N>(
    decomposer: D,
    convert: F,
    noise_fn: Option<N>,
    matrix: DynamicDiffusionMatrix,
) -> Box<dyn DynDitherer<InOutType>>
where
    D: epd_dither::Decomposer<f32> + 'static,
    F: Fn(Src) -> D::Input + 'static,
    Src: 'static,
    N: Fn(usize, usize) -> f32 + 'static,
    InOutType: epd_dither::dither::ImageReader<Src>,
{
    let strategy = DecomposingDitherStrategy::new(decomposer, convert);
    match noise_fn {
        Some(n) => Box::new(BundledDitherer::new(strategy.with_noise(n), matrix)),
        None => Box::new(BundledDitherer::new(strategy, matrix)),
    }
}

/// Match over `DecomposeStrategy` and build the corresponding
/// `Box<dyn DynDitherer<InOutType>>`. Generic over the noise closure type
/// so the eight strategy arms only appear once regardless of which noise
/// source is selected.
fn build_with_noise<N>(
    strategy_choice: &DecomposeStrategy,
    palette: &[Rgb<u8>],
    noise_fn: Option<N>,
    matrix: DynamicDiffusionMatrix,
) -> Box<dyn DynDitherer<InOutType>>
where
    N: Fn(usize, usize) -> f32 + 'static,
{
    let palette_rgb_f32: Vec<Rgb<f32>> = palette
        .iter()
        .map(|c| {
            let [r, g, b] = c.0;
            Rgb([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0])
        })
        .collect();
    match *strategy_choice {
        DecomposeStrategy::OctahedronClosest => {
            let palette_points: Vec<Point3<f32>> =
                palette_rgb_f32.iter().map(|p| color_to_point(*p)).collect();
            let decomposer = OctahedronDecomposer::new(&palette_points)
                .unwrap()
                .with_strategy(OctahedronDecomposerAxisStrategy::Closest);
            build_decomposing(decomposer, color_to_point, noise_fn, matrix)
        }
        DecomposeStrategy::OctahedronFurthest => {
            let palette_points: Vec<Point3<f32>> =
                palette_rgb_f32.iter().map(|p| color_to_point(*p)).collect();
            let decomposer = OctahedronDecomposer::new(&palette_points)
                .unwrap()
                .with_strategy(OctahedronDecomposerAxisStrategy::Furthest);
            build_decomposing(decomposer, color_to_point, noise_fn, matrix)
        }
        DecomposeStrategy::NaiveMix => {
            let palette_points: Vec<Point3<f32>> =
                palette_rgb_f32.iter().map(|p| color_to_point(*p)).collect();
            let decomposer = NaiveDecomposer::new(&palette_points)
                .unwrap()
                .with_strategy(NaiveDecomposerStrategy::FavorMix);
            build_decomposing(decomposer, color_to_point, noise_fn, matrix)
        }
        DecomposeStrategy::NaiveDominant => {
            let palette_points: Vec<Point3<f32>> =
                palette_rgb_f32.iter().map(|p| color_to_point(*p)).collect();
            let decomposer = NaiveDecomposer::new(&palette_points)
                .unwrap()
                .with_strategy(NaiveDecomposerStrategy::FavorDominant);
            build_decomposing(decomposer, color_to_point, noise_fn, matrix)
        }
        DecomposeStrategy::NaiveBlend(power) => {
            let palette_points: Vec<Point3<f32>> =
                palette_rgb_f32.iter().map(|p| color_to_point(*p)).collect();
            let decomposer = NaiveDecomposer::new(&palette_points)
                .unwrap()
                .with_strategy(NaiveDecomposerStrategy::TetraBlend(power));
            build_decomposing(decomposer, color_to_point, noise_fn, matrix)
        }
        DecomposeStrategy::Grayscale => {
            assert!(
                verify_grayscale_palette(palette),
                "grayscale strategy requires an achromatic, strictly-ascending palette"
            );
            // Both gray decomposers produce identical output at parameter = 0
            // (plain bracket decomposition), but OffsetBlend takes its
            // early-out path while PureSpread still runs the full
            // asymmetric-spread arithmetic. Route through OffsetBlend.
            let levels: Vec<f32> = palette_rgb_f32.iter().map(|p| rgb_to_brightness(*p)).collect();
            let decomposer = OffsetBlendGrayDecomposer::new(levels).unwrap();
            build_decomposing(decomposer, rgb_to_brightness, noise_fn, matrix)
        }
        DecomposeStrategy::GrayPureSpread(spread_ratio) => {
            assert!(
                verify_grayscale_palette(palette),
                "grayscale strategy requires an achromatic, strictly-ascending palette"
            );
            let levels: Vec<f32> = palette_rgb_f32.iter().map(|p| rgb_to_brightness(*p)).collect();
            let decomposer = PureSpreadGrayDecomposer::new(levels)
                .unwrap()
                .with_spread_ratio(spread_ratio);
            build_decomposing(decomposer, rgb_to_brightness, noise_fn, matrix)
        }
        DecomposeStrategy::GrayOffsetBlend(distance) => {
            assert!(
                verify_grayscale_palette(palette),
                "grayscale strategy requires an achromatic, strictly-ascending palette"
            );
            let levels: Vec<f32> = palette_rgb_f32.iter().map(|p| rgb_to_brightness(*p)).collect();
            let decomposer = OffsetBlendGrayDecomposer::new(levels)
                .unwrap()
                .with_distance(distance);
            build_decomposing(decomposer, rgb_to_brightness, noise_fn, matrix)
        }
    }
}

/// Top-level builder: select a noise closure type, then dispatch to
/// `build_with_noise`. Each `NoiseSource` arm pins a concrete `N`; the
/// `None` variant supplies a `fn`-pointer placeholder via turbofish.
fn build_ditherer(
    strategy_choice: &DecomposeStrategy,
    palette: &[Rgb<u8>],
    noise_choice: NoiseSource,
    matrix: DynamicDiffusionMatrix,
) -> Box<dyn DynDitherer<InOutType>> {
    match noise_choice {
        NoiseSource::Bayer(Some(max_depth)) => build_with_noise(
            strategy_choice,
            palette,
            Some(move |x: usize, y: usize| epd_dither::noise::bayer(x, y, max_depth)),
            matrix,
        ),
        NoiseSource::Bayer(None) => build_with_noise(
            strategy_choice,
            palette,
            Some(|x: usize, y: usize| epd_dither::noise::bayer_inf(x, y)),
            matrix,
        ),
        NoiseSource::InterleavedGradient => build_with_noise(
            strategy_choice,
            palette,
            Some(|x: usize, y: usize| {
                epd_dither::noise::interleaved_gradient_noise(x as f32, y as f32)
            }),
            matrix,
        ),
        NoiseSource::White => build_with_noise(
            strategy_choice,
            palette,
            Some(|_x: usize, _y: usize| rand::rng().sample::<f32, _>(StandardUniform)),
            matrix,
        ),
        NoiseSource::File(f) => build_with_noise(
            strategy_choice,
            palette,
            Some(move |x: usize, y: usize| {
                f.get_pixel(x as u32 % f.width(), y as u32 % f.height()).0[0]
            }),
            matrix,
        ),
        NoiseSource::None => build_with_noise::<fn(usize, usize) -> f32>(
            strategy_choice,
            palette,
            None,
            matrix,
        ),
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
    let dither_palette_rgb: Vec<Rgb<u8>> = dither_palette_u8.iter().map(|&c| Rgb(c)).collect();
    let (width, height) = (input.width() as usize, input.height() as usize);
    let output_width = input.width();
    let output_height = input.height();
    let mut inout = ImageSplit::new(input, IndexedBuffer::new(width, height)).unwrap();
    let matrix = args.diffuse.to_dynamic_matrix();
    let ditherer = build_ditherer(&args.strategy, &dither_palette_rgb, args.noise, matrix);
    ditherer.dyn_dither_into(&mut inout);
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
