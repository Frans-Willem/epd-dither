use clap::Parser;
use epd_dither::Palette;
use epd_dither::dither::diffusion_matrix::DiffuseMethod;
use epd_dither::dither::{DecomposeStrategy, DynDitherer, ImageSize, ImageSplit, ImageWriter};
use epd_dither::factory::decompose_ditherer;
use epd_dither::noise::NoiseSource;
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

    let (width, height) = (input.width() as usize, input.height() as usize);
    let output_width = input.width();
    let output_height = input.height();
    let mut inout = ImageSplit::new(input, IndexedBuffer::new(width, height)).unwrap();

    let palette_rgb: Vec<Rgb<u8>> = dither_palette.iter().map(|&c| Rgb(c)).collect();
    let ditherer: Box<dyn DynDitherer<_>> = decompose_ditherer::<Rgb<f32>, Rgb<u8>, _>(
        args.strategy,
        args.noise,
        &palette_rgb,
        args.diffuse.to_matrix(),
    )
    .unwrap();
    ditherer.dyn_dither_into(&mut inout);

    let mut output = png::Encoder::new(
        std::io::BufWriter::new(std::fs::File::create(&args.output_file).unwrap()),
        output_width,
        output_height,
    );
    output.set_color(png::ColorType::Indexed);
    output.set_depth(png::BitDepth::Eight);
    let palette: Vec<u8> = args
        .output_palette
        .as_rgb_slice()
        .iter()
        .flat_map(|c| *c)
        .collect();
    output.set_palette(palette);
    let data: Vec<u8> = inout.writer.data.iter().map(|x| *x as u8).collect();
    let mut output = output.write_header().unwrap();
    output.write_image_data(data.as_slice()).unwrap();
    println!("Done");
}
