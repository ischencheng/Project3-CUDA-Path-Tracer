#!/bin/bash
# Renders the README images into img/ (run from the repository root after a
# Release build). Showcase scenes use ACES + gamma; Cornell scenes keep the
# linear look of the reference render.
set -e
EXE="$(pwd)/build/bin/Release/cis565_path_tracer.exe"
OUT="$(pwd)/build/gallery"
mkdir -p "$OUT"
cd "$OUT"
FILMIC="--tonemap aces --gamma 1"

$EXE ../../scenes/cover.json --headless --spp 3000 $FILMIC --denoise 1 --out cover
$EXE ../../scenes/cornell.json --headless --spp 5000 --out cornell
$EXE ../../scenes/materials.json --headless --spp 4000 $FILMIC --denoise 1 --out materials
$EXE ../../scenes/textures.json --headless --spp 3000 $FILMIC --denoise 1 --out textures
$EXE ../../scenes/motion_blur.json --headless --spp 3000 $FILMIC --out motion_blur
$EXE ../../scenes/motion_blur.json --headless --spp 3000 $FILMIC --motion 0 --out motion_still
$EXE ../../scenes/sunset.json --headless --spp 3000 $FILMIC --denoise 1 --out sunset
$EXE ../../scenes/dragon.json --headless --spp 3000 $FILMIC --denoise 1 --out dragon

# strategy comparisons at equal sample counts
for m in "--nee 0" "--nee 1 --mis 0" "--nee 1 --mis 1"; do
    n=$(echo $m | tr -d ' -')
    $EXE ../../scenes/veach_mis.json --headless --spp 64 $FILMIC --out veach_$n $m
done
$EXE ../../scenes/veach_mis.json --headless --spp 8192 $FILMIC --out veach_reference

# antialiasing and depth of field
$EXE ../../scenes/cornell.json --headless --spp 1000 --aa 0 --out cornell_noaa
$EXE ../../scenes/cornell.json --headless --spp 1000 --aa 1 --out cornell_aa

# denoiser inputs at low sample counts
for spp in 4 16 64; do
    $EXE ../../scenes/cover.json --headless --spp $spp --res 960x540 $FILMIC --denoise 1 --save-features --out cover_dn
done

# random vs Sobol at 4 spp
$EXE ../../scenes/textures.json --headless --spp 4 --res 480x270 $FILMIC --sampler random --out sampler_random
$EXE ../../scenes/textures.json --headless --spp 4 --res 480x270 $FILMIC --sampler sobol --out sampler_sobol

# depth of field on/off, BVH cost heatmap
$EXE ../../scenes/cover.json --headless --spp 1024 --res 960x540 $FILMIC --denoise 1 --aperture 0 --out cover_pinhole
$EXE ../../scenes/cover.json --headless --spp 1024 --res 960x540 $FILMIC --denoise 1 --out cover_lens
$EXE ../../scenes/cover.json --headless --spp 1 --res 960x540 --save-bvh-cost --out cover_cost
