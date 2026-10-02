"""Draws feature callouts on the cover image for the README.

usage: python annotate_cover.py [img/cover.png] [img/cover_annotated.jpg]
"""
import sys

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "C:/Windows/Fonts/"
TITLE = ImageFont.truetype(FONT_DIR + "seguisb.ttf", 25)
BODY = ImageFont.truetype(FONT_DIR + "segoeui.ttf", 21)
PAD_X, PAD_Y, GAP = 14, 9, 3

# (box top-left, target point on the image or None, title, detail line)
CALLOUTS = [
    ((470, 22), (790, 330), "Glass dragon, 135k triangles",
     "glTF mesh in a binned-SAH BVH; exact Fresnel + Beer-Lambert absorption"),
    ((24, 160), None, "HDR environment map",
     "importance-sampled, next event estimation + MIS"),
    ((24, 318), (395, 500), "DamagedHelmet (glTF)",
     "base color, metallic-roughness, normal, emissive maps"),
    ((1222, 150), (1110, 230), "Cloth backdrop",
     "glTF image texture"),
    ((1205, 430), (1150, 575), "Gold sphere",
     "GGX conductor, VNDF sampling"),
    ((24, 812), (575, 655), "Rough teal glass",
     "GGX transmission + volume absorption"),
    ((690, 812), (1005, 655), "Smooth glass sphere",
     "refracts an inverted image of the dragon"),
    ((1218, 812), (1330, 755), "Thin-lens depth of field",
     "aperture 0.09, focused on the dragon"),
]


def box_size(draw, title, body):
    tw = draw.textbbox((0, 0), title, font=TITLE)
    bw = draw.textbbox((0, 0), body, font=BODY)
    w = max(tw[2], bw[2]) + 2 * PAD_X
    h = (tw[3] - tw[1]) + (bw[3] - bw[1]) + GAP + 2 * PAD_Y + 8
    return w, h


def closest_point(rect, p):
    x0, y0, x1, y1 = rect
    return (min(max(p[0], x0), x1), min(max(p[1], y0), y1))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "img/cover.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else "img/cover_annotated.jpg"
    base = Image.open(src).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    boxes = []
    for (x, y), target, title, body in CALLOUTS:
        w, h = box_size(draw, title, body)
        x = min(x, base.width - w - 24)  # keep right-hand boxes inside the frame
        y = min(y, base.height - h - 12)
        boxes.append(((x, y, x + w, y + h), target, title, body))

    # leader lines first so the boxes sit on top of them
    for rect, target, _, _ in boxes:
        if target is None:
            continue
        start = closest_point(rect, target)
        draw.line([start, target], fill=(0, 0, 0, 150), width=5)
        draw.line([start, target], fill=(255, 255, 255, 255), width=2)
        r = 6
        draw.ellipse([target[0] - r - 2, target[1] - r - 2, target[0] + r + 2, target[1] + r + 2], fill=(0, 0, 0, 150))
        draw.ellipse([target[0] - r, target[1] - r, target[0] + r, target[1] + r], fill=(255, 255, 255, 255))

    for rect, _, title, body in boxes:
        draw.rounded_rectangle(rect, radius=9, fill=(18, 20, 26, 205))
        x, y = rect[0] + PAD_X, rect[1] + PAD_Y
        draw.text((x, y), title, font=TITLE, fill=(255, 255, 255, 255))
        th = draw.textbbox((0, 0), title, font=TITLE)[3]
        draw.text((x, y + th + GAP), body, font=BODY, fill=(214, 218, 226, 255))

    Image.alpha_composite(base, overlay).convert("RGB").save(dst, quality=90, optimize=True)
    print("wrote", dst)


if __name__ == "__main__":
    main()
