use clap::Parser;
use epd_dither::Palette;
use epd_dither::dither::diffusion_matrix::DiffuseMethod;
use epd_dither::dither::{DecomposeStrategy, DynDitherer, ImageSplit};
use epd_dither::factory::decompose_ditherer;
use epd_dither::noise::NoiseSource;
use epd_dither::palette_image::{PaletteImage, VerifiedPalette};
use image::Rgb;

#[derive(Parser)]
#[command(name = "dither")]
struct Args {
    #[arg()]
    input_file: String,
    #[arg()]
    output_file: String,
    #[arg(long, value_name="NOISE", long_help=NoiseSource::LONG_HELP, default_value = "ign")]
    noise: NoiseSource,
    #[arg(long, value_name = "STRATEGY", long_help = DecomposeStrategy::LONG_HELP, default_value = "octahedron-closest")]
    strategy: DecomposeStrategy,
    #[arg(long, value_name = "DIFFUSE", long_help = DiffuseMethod::LONG_HELP, default_value = "floyd-steinberg")]
    diffuse: DiffuseMethod,
    #[arg(long, value_name = "DITHER_PALETTE", long_help = Palette::LONG_HELP, default_value = "spectra6")]
    dither_palette: Palette,
    #[arg(long, value_name = "OUTPUT_PALETTE", long_help = Palette::LONG_HELP, default_value = "spectra6")]
    output_palette: Palette,
}

fn main() {
    let args = Args::parse();
    println!("Opening image");
    let input = image::ImageReader::open(&args.input_file)
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb32f();
    println!("Opened image");

    let dither_palette = args.dither_palette.as_rgb_slice();
    println!("Dither palette used:");
    for color in dither_palette {
        println!("  #{:02X}{:02X}{:02X},", color[0], color[1], color[2]);
    }

    let output_width = input.width();
    let output_height = input.height();
    let output_palette: Vec<Rgb<u8>> = args
        .output_palette
        .as_rgb_slice()
        .iter()
        .map(|&c| Rgb(c))
        .collect();
    let writer = PaletteImage::new(
        output_width,
        output_height,
        VerifiedPalette::new(output_palette).unwrap(),
    );
    let mut inout = ImageSplit::new(input, writer).unwrap();

    let palette_rgb: Vec<Rgb<u8>> = dither_palette.iter().map(|&c| Rgb(c)).collect();
    let ditherer: Box<dyn DynDitherer<_>> = decompose_ditherer::<Rgb<f32>, Rgb<u8>, _>(
        args.strategy,
        args.noise,
        &palette_rgb,
        args.diffuse.to_matrix(),
    )
    .unwrap();
    ditherer.dyn_dither_into(&mut inout);

    let png_bytes = inout.writer.to_png().unwrap();
    std::fs::write(&args.output_file, png_bytes).unwrap();
    println!("Done");
}
