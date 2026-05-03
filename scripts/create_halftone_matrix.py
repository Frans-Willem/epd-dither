import numpy as np
from PIL import Image

def main(size, filename):
    # Create a matrix with the x position in each cell
    xs = np.array([[x for x in range(0,size)]] * size)
    # Distance to middle
    distance_to_mid_x = xs - ((size-1) / 2)
    distance_sq_to_mid_x = distance_to_mid_x ** 2
    # Note that ys and distance_to_mid_y is just the transpose of the x ones
    distance_sq_to_mid = distance_sq_to_mid_x + distance_sq_to_mid_x.transpose()
    distance_to_mid = np.sqrt(distance_sq_to_mid)
    # Remap from 0 to 0.5
    dot_a = distance_to_mid - np.min(distance_to_mid)
    dot_a /= np.max(dot_a) * 2
    dot_b = 1 - dot_a
    # Convert to [AB, BA]
    dots = np.hstack([np.vstack([dot_a,dot_b]), np.vstack([dot_b, dot_a])])
    #print("Dots", dots)

    coordinates = [(x,y) for x in range(0, dots.shape[1]) for y in range(0, dots.shape[0])]
    coordinates = sorted(coordinates, key=lambda c: dots[c[1], c[0]])
    #print("Coordinates:", coordinates)

    if dots.shape[0] * dots.shape[1] <= 256:
        dtype = np.uint8
        dtype_range = 2**8
    else:
        dtype = np.uint16
        dtype_range = 2**16

    output = np.zeros(dots.shape, dtype=dtype)
    for index, coordinate in enumerate(coordinates):
        index = (index * dtype_range) // (dots.shape[0] * dots.shape[1])
        #print("Coord", coordinate, "=", index)
        output[coordinate[1], coordinate[0]] = index

    output = Image.fromarray(output)
    output.save(filename)
    print("Done")
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Creates a noise/threshold matrix that mimics newspaper halftoning")
    parser.add_argument("size", help="Size (in pixels) of one dot. Full image will be twice this in both width and height", type=int)
    parser.add_argument("filename", help="Filename to save to")
    args = parser.parse_args()
    main(**vars(args))
