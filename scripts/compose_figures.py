"""Assembles the README comparison figures from the renders in build/gallery.

usage: python compose_figures.py   (after scripts/render_gallery.sh)
"""
import os
import shutil

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "build", "gallery")
DST = os.path.join(ROOT, "img", "gallery")
BG = (252, 252, 251)
INK = (11, 11, 11)


def font(size):
    for name in ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load(name):
    return Image.open(os.path.join(SRC, name)).convert("RGB")


def row(images, labels=None, gap=8, label_h=34):
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    h = max(im.height for im in images) + (label_h if labels else 0)
    out = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(out)
    f = font(22)
    x = 0
    for i, im in enumerate(images):
        out.paste(im, (x, label_h if labels else 0))
        if labels:
            draw.text((x + 4, 4), labels[i], fill=INK, font=f)
        x += im.width + gap
    return out


def grid(images, labels, cols, gap=8, label_h=34):
    rows = [row(images[i:i + cols], labels[i:i + cols], gap, label_h) for i in range(0, len(images), cols)]
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, sum(r.height for r in rows) + gap * (len(rows) - 1)), BG)
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + gap
    return out


def zoom(im, box, factor):
    crop = im.crop(box)
    return crop.resize((crop.width * factor, crop.height * factor), Image.NEAREST)


def save(im, name, width=None):
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(os.path.join(DST, name), optimize=True)
    print("saved", name, im.size)


def main():
    os.makedirs(DST, exist_ok=True)
    shutil.copy(os.path.join(SRC, "cover.3000samp.denoised.png"), os.path.join(ROOT, "img", "cover.png"))
    save(load("cornell.5000samp.png"), "cornell.png")
    save(load("materials.4000samp.denoised.png"), "materials.png")
    save(load("textures.3000samp.denoised.png"), "textures.png")
    save(load("sunset.3000samp.denoised.png"), "sunset.png")
    save(load("dragon.3000samp.denoised.png"), "dragon.png")

    # antialiasing: sphere edge and back corner, 4x
    a, b = load("cornell_noaa.1000samp.png"), load("cornell_aa.1000samp.png")
    sphere, light = (300, 385, 380, 465), (305, 160, 385, 240)
    save(row([zoom(a, sphere, 4), zoom(b, sphere, 4), zoom(a, light, 4), zoom(b, light, 4)],
             ["no AA", "stochastic AA", "no AA", "stochastic AA"]), "aa_compare.png", 1400)

    # depth of field
    save(row([load("cover_pinhole.1024samp.denoised.png"), load("cover_lens.1024samp.denoised.png")],
             ["pinhole", "thin lens, radius 0.09"]), "dof_compare.png", 1800)

    # light sampling strategies at 64 spp
    save(row([load("veach_nee0.64samp.png"), load("veach_nee1mis0.64samp.png"), load("veach_nee1mis1.64samp.png")],
             ["BSDF sampling", "light sampling", "MIS"]), "veach_compare.png", 1800)

    # denoiser inputs and outputs at 16 spp
    save(grid([load("cover_dn.16samp.png"), load("cover_dn.16samp.denoised.png"),
               load("cover_dn.16samp.albedo.png"), load("cover_dn.16samp.normal.png")],
              ["16 spp", "16 spp denoised", "albedo guide", "normal guide"], 2), "denoise_grid.png", 1800)
    ref = load("cover.3000samp.png").resize((960, 540), Image.LANCZOS)
    save(grid([load("cover_dn.4samp.denoised.png"), load("cover_dn.16samp.denoised.png"),
               load("cover_dn.64samp.denoised.png"), ref],
              ["4 spp denoised", "16 spp denoised", "64 spp denoised", "3000 spp, no denoiser"], 2),
         "denoise_spp.png", 1800)

    # motion blur
    save(row([load("motion_still.3000samp.png"), load("motion_blur.3000samp.png")],
             ["motion blur off", "motion blur on"]), "motion_compare.png", 1600)

    # samplers at 4 spp
    save(row([load("sampler_random.4samp.png"), load("sampler_sobol.4samp.png")],
             ["random, 4 spp", "Sobol, 4 spp"]), "sampler_compare.png", 1400)

    # BVH cost heatmap
    shutil.copy(os.path.join(SRC, "cover_cost.1samp.bvhcost.png"), os.path.join(DST, "bvh_cost.png"))


if __name__ == "__main__":
    main()
