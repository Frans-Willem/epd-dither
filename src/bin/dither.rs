use epd_dither::decomposer_bruteforce::{DecomposerBruteforce, DecomposerBruteforceStrategy};
use epd_dither::decomposer6c::{Decomposer6C, Decomposer6CAxisStrategy};
use image::{DynamicImage, ImageBuffer, ImageReader, Luma, Rgb};
use nalgebra::geometry::Point3;
use nalgebra::{DVector, Vector6};

use clap::{Parser, ValueEnum};
use rand::distr::StandardUniform;
use rand::prelude::*;

#[derive(Clone, Debug)]
enum NoiseSource {
    Bayer(Option<usize>),
    InterleavedGradient,
    White,
    File(Box<ImageBuffer<Luma<f32>, Vec<f32>>>),
}

impl NoiseSource {
    const LONG_HELP: &'static str = concat!(
        "Noise source to use.\n\n",
        "Accepted values:\n",
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

#[derive(Clone, Debug, ValueEnum)]
enum DecomposeStrategy {
    OctahedronClosest,
    OctahedronFurthest,
    BruteforceMix,
    BruteforceDominant,
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
    #[arg(long, value_name = "STRATEGY", default_value = "octahedron-closest")]
    strategy: DecomposeStrategy,
}

#[allow(dead_code)]
const PALETTE_MINE: [Rgb<f32>; 6] = [
    Rgb([
        0.226_762_65,
        0.0, //-0.0055970314385675474,
        0.259_709_48,
    ]),
    Rgb([0.700_009_5, 0.816_166_4, 0.786_138_5]),
    Rgb([0.239_182_67, 0.148_258_42, 0.596_627_6]),
    Rgb([0.378_047_26, 0.408_988_65, 0.338_833_5]),
    Rgb([0.590_308_67, 0.147_103_09, 0.172_003_87]),
    Rgb([
        0.841_725_77,
        0.912_686_1,
        0.0, //-0.053016650312371474,
    ]),
];

const PALETTE_EPDOPTIMIZE: [Rgb<u8>; 6] = [
    Rgb([0x19, 0x1E, 0x21]),
    Rgb([0xe8, 0xe8, 0xe8]),
    Rgb([0x21, 0x57, 0xba]),
    Rgb([0x12, 0x5f, 0x20]),
    Rgb([0xb2, 0x13, 0x18]),
    Rgb([0xef, 0xde, 0x44]),
];

const PALETTE_MEASURED: [Rgb<u8>; 6] = [
    Rgb([179, 208, 200]),
    Rgb([61, 38, 152]),
    Rgb([96, 104, 86]),
    Rgb([151, 38, 44]),
    Rgb([215, 233, 0]),
    Rgb([58, 0, 66]),
];

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

// TODO: Move to library
fn pick_from_barycentric_weights(weights: DVector<f32>, offset: f32) -> usize {
    let mut index = 0;
    let mut offset = offset;
    while index + 1 < 6 && weights[index] <= offset {
        offset -= weights[index];
        index += 1;
    }
    index
}

fn owned_to_dynamic_vector<T: nalgebra::Scalar, const N: usize>(
    vec: nalgebra::SVector<T, N>,
) -> DVector<T> {
    DVector::from_column_slice(vec.as_slice())
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
    println!("Palette used:");
    // TODO: Allow dither and output palette to be specified
    let palette_u8 = PALETTE_EPDOPTIMIZE;
    for color in palette_u8 {
        println!("  #{:02X}{:02X}{:02X},", color.0[0], color.0[1], color.0[2]);
    }
    let palette_f32 = palette_u8.map(|c| Rgb(c.0.map(|x| (x as f32) / 255.0)));
    let palette_as_points = palette_f32.map(color_to_point);
    let decompose: Box<dyn Fn(Point3<f32>) -> DVector<f32>> = match args.strategy {
        DecomposeStrategy::OctahedronClosest => {
            let decomposer = Decomposer6C::new(&palette_as_points).unwrap();
            Box::new(move |x| {
                owned_to_dynamic_vector(decomposer.decompose(&x, Decomposer6CAxisStrategy::Closest))
            })
        }
        DecomposeStrategy::OctahedronFurthest => {
            let decomposer = Decomposer6C::new(&palette_as_points).unwrap();
            Box::new(move |x| {
                owned_to_dynamic_vector(
                    decomposer.decompose(&x, Decomposer6CAxisStrategy::Furthest),
                )
            })
        }
        DecomposeStrategy::BruteforceMix => {
            let decomposer = DecomposerBruteforce::new(&palette_as_points).unwrap();
            Box::new(move |x| decomposer.decompose(&x, DecomposerBruteforceStrategy::FavorMix))
        }
        DecomposeStrategy::BruteforceDominant => {
            let decomposer = DecomposerBruteforce::new(&palette_as_points).unwrap();
            Box::new(move |x| decomposer.decompose(&x, DecomposerBruteforceStrategy::FavorDominant))
        }
    };

    let mut input = input;
    println!("Iterating over pixels");
    for (x, y, pixel) in input.enumerate_pixels_mut() {
        let value: Rgb<f32> = *pixel;

        let value = color_to_point(value);
        let barycentric: DVector<f32> = decompose(value);
        let noise = match args.noise {
            NoiseSource::Bayer(Some(max_depth)) => {
                epd_dither::noise::bayer(x as usize, y as usize, max_depth)
            }
            NoiseSource::Bayer(None) => epd_dither::noise::bayer_inf(x as usize, y as usize),
            NoiseSource::InterleavedGradient => {
                epd_dither::noise::interleaved_gradient_noise(x as f32, y as f32)
            }
            NoiseSource::White => rand::rng().sample(StandardUniform),
            NoiseSource::File(ref f) => {
                f.get_pixel(x as u32 % f.width(), y as u32 % f.height()).0[0].clone()
            }
        };
        let index = pick_from_barycentric_weights(barycentric, noise);
        let value = palette_f32[index];
        *pixel = value;
    }
    println!("Converting back to U8");
    let input: DynamicImage = input.into();
    let input = input.into_rgb8();
    input.save(args.output_file).unwrap();
    println!("Done");
}
