#!/usr/bin/env bash
# Regression harness: dither documentation/lena_original.png over every
# (strategy, deterministic-noise) combination and write the outputs to <out_dir>.
# Re-run on two different revisions and `diff -r` (or sha256sum) the directories
# to verify a refactor hasn't changed pixel output.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_dir>" >&2
    exit 1
fi

OUT_DIR="$1"
mkdir -p "$OUT_DIR"

cd "$(dirname "$0")/.."

INPUT="documentation/lena_original.png"
DITHER="target/release/dither"

cargo build --release --bin dither >&2

NOISES=(none bayer:8 bayer ign file:assets/HDR_L_0.png)

run_dither() {
    local strategy="$1"
    local noise="$2"
    shift 2
    # Sanitize for filename: replace : / . with _, but strip any directory
    # prefix from `file:<path>` first so the slug stays anchored to the asset
    # basename (otherwise moving the asset reshuffles all baseline filenames).
    local slug_strat="${strategy//:/_}"
    slug_strat="${slug_strat//./_}"
    local slug_noise="$noise"
    if [[ "$slug_noise" == file:* ]]; then
        slug_noise="file:$(basename "${slug_noise#file:}")"
    fi
    slug_noise="${slug_noise//:/_}"
    slug_noise="${slug_noise//\//_}"
    slug_noise="${slug_noise//./_}"
    local out="$OUT_DIR/${slug_strat}__${slug_noise}.png"
    echo "  $strategy / $noise -> $(basename "$out")" >&2
    "$DITHER" --strategy "$strategy" --noise "$noise" "$@" "$INPUT" "$out"
}

# Spectra-6-target strategies (default palette = spectra6).
for strategy in octahedron-closest octahedron-furthest naive-mix naive-dominant; do
    for noise in "${NOISES[@]}"; do
        run_dither "$strategy" "$noise"
    done
done

# Grayscale strategies need a grayscale palette; using grayscale4 here.
for strategy in grayscale gray-pure-spread:0.25 gray-pure-spread:0.5 gray-pure-spread:1 gray-offset-blend:0.25 gray-offset-blend:0.5 gray-offset-blend:1; do
    for noise in "${NOISES[@]}"; do
        run_dither "$strategy" "$noise" --dither-palette grayscale4 --output-palette grayscale4
    done
done

echo "Done. Outputs in $OUT_DIR" >&2
