use image::{DynamicImage, ImageReader, Rgb};
use nalgebra::geometry::Point3;
use nalgebra::Vector6;
use epd_dither::barycentric::octahedron::OctahedronProjector;

#[allow(dead_code)]
const PALETTE: [Rgb<f32>; 6] = [
    Rgb([
        0.22676264646610847,
        0.0, //-0.0055970314385675474,
        0.2597094644681131,
    ]),
    Rgb([0.7000095212021477, 0.8161663966432444, 0.7861384978213591]),
    Rgb([0.2391826586300411, 0.1482584219935382, 0.596627604515917]),
    Rgb([0.37804726446284537, 0.40898865257247025, 0.3388335156024263]),
    Rgb([0.5903086439496402, 0.14710309178681208, 0.17200386219219121]),
    Rgb([
        0.8417257856614314,
        0.9126861145185275,
        0.0, //-0.053016650312371474,
    ]),
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

fn pick_from_barycentric_weights(weights: Vector6<f32>, offset: f32) -> usize {
    let mut index = 0;
    let mut offset = offset;
    while index + 1 < 6 && weights[index] < offset {
        offset -= weights[index];
        index += 1;
    }
    index
}

fn interleaved_gradient_noise(x: u32, y: u32) -> f32 {
    // InterleavedGradientNoise[x_, y_] := FractionalPart[52.9829189*FractionalPart[0.06711056*x + 0.00583715*y]]
    let inner1 : f32 = (0.06711056 * (x as f32)) + (0.00583715 * (y as f32));
    let inner2 : f32 = 52.9839189 * inner1.fract();
    return inner2.fract();
}

fn main() {
    println!("Opening image");
    let input = ImageReader::open("test.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb32f();
    println!("Opened image");
    let projector = OctahedronProjector::new([
        color_to_point(PALETTE[SpectraColors::Black as usize]),
        color_to_point(PALETTE[SpectraColors::White as usize]),
        color_to_point(PALETTE[SpectraColors::Blue as usize]),
        color_to_point(PALETTE[SpectraColors::Green as usize]),
        color_to_point(PALETTE[SpectraColors::Yellow as usize]),
        color_to_point(PALETTE[SpectraColors::Red as usize]),
    ]);

    // NOTE: Maybe the octahedron isn't convex at all, maybe blue-green crosses "behind" the north
    // to south pole, and maybe we can just ignore that one. Would that even affect both
    // pole-barycentric-coordinates being <0
    // TODO: Check why there are faces towards white being checked, that shouldn't happen at all :/
    println!(
        "Full black: {:?}",
        projector.project(&Point3::<f32>::new(0.0, 0.0, 0.0))
    );

    let mut input = input;
    println!("Iterating over pixels");
    for (x,y,pixel) in input.enumerate_pixels_mut() {
        let value: Rgb<f32> = *pixel;

        let value = color_to_point(value);
        let barycentric: Vector6<f32> = projector.project(&value);
        let barycentric: Vector6<f32> = Vector6::new(
            barycentric[0],
            barycentric[1],
            barycentric[2],
            barycentric[3],
            barycentric[5],
            barycentric[4],
        );
        let noise = interleaved_gradient_noise(x,y);
        let index = pick_from_barycentric_weights(barycentric, noise);
        let value = PALETTE[index].clone();
        /*
        let value: Vector3<f32> = matrix * barycentric;
        let value: Point3<f32> = Point3::from(value);
        let value: Rgb<f32> = point_to_color(value);
        */

        *pixel = value;
    }
    println!("Converting back to U8");
    let input: DynamicImage = input.into();
    let input = input.into_rgb8();
    input.save("changed.png").unwrap();
    print!("Done");
}
