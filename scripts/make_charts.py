"""Builds the README charts from the CSV files in analysis/.

usage: python make_charts.py
Colors follow a fixed categorical order (validated for color vision
deficiencies); every chart has a legend for two or more series and the
README carries the numbers as tables.
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "analysis")
IMG = os.path.join(ROOT, "img", "charts")

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
})


def read(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def clean(ax, xgrid=False):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="x", visible=xgrid)
    ax.tick_params(length=0)


def save(fig, name):
    os.makedirs(IMG, exist_ok=True)
    fig.savefig(os.path.join(IMG, name))
    plt.close(fig)
    print("saved", name)


def plain_log_axis(ax, axis, ticks=None):
    """Log axis with plain-number tick labels (1, 4, 16 ... or 0.01, 0.1)."""
    fmt = FuncFormatter(lambda v, _: ("%g" % v))
    target = ax.xaxis if axis == "x" else ax.yaxis
    if ticks is not None:
        target.set_ticks(ticks)
    target.set_major_formatter(fmt)
    target.set_minor_formatter(NullFormatter())


def line(ax, x, y, color, label, marker=True):
    ax.plot(x, y, color=color, label=label, marker="o" if marker else None, markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5)


def end_label(ax, x, y, text, dx=6, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", color=INK2, fontsize=9)


# ---------------------------------------------------------------------------

def alive_paths():
    rows = read("alive_paths.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = {"cornell_open": "Open box (front is open)", "cornell_closed": "Closed box"}
    for i, sc in enumerate(["cornell_open", "cornell_closed"]):
        pts = [(int(r["bounce"]), int(r["alive"]) / 1000) for r in rows if r["scene"] == sc]
        x, y = zip(*pts)
        line(ax, x, y, SERIES[i], names[sc])
        end_label(ax, x[-1], y[-1], "%.0fk" % y[-1])
    ax.set_xlabel("bounce")
    ax.set_ylabel("live paths (thousands)")
    ax.set_title("Live paths at the start of each bounce (800x800, no russian roulette)", loc="left")
    ax.set_ylim(0, 700)
    clean(ax)
    ax.legend(loc="lower left")
    save(fig, "alive_paths.png")


def compaction():
    rows = read("compaction.csv")
    if not rows:
        return
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for ax, sc, title in zip(axes, ["cornell_open", "cornell_closed"], ["Open Cornell box", "Closed Cornell box"]):
        vals = [float(next(r["ms"] for r in rows if r["scene"] == sc and r["mode"] == m)) for m in modes]
        ypos = list(range(len(modes)))[::-1]
        ax.barh(ypos, vals, height=0.42, color=SERIES[0])
        for y, v in zip(ypos, vals):
            ax.annotate("%.1f ms" % v, (v, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels(modes if ax is axes[0] else [""] * len(modes))
        ax.set_title(title, loc="left")
        ax.set_xlabel("ms per iteration (depth 16)")
        clean(ax, xgrid=True)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0, max(float(r["ms"]) for r in rows) * 1.25)
    save(fig, "compaction.png")

    sweep = read("compaction_threshold.csv")
    if sweep:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for i, sc in enumerate(["cornell_open", "cornell_closed"]):
            pts = sorted((float(r["threshold"]), float(r["ms"])) for r in sweep if r["scene"] == sc)
            x, y = zip(*pts)
            line(ax, x, y, SERIES[i], "Open box" if i == 0 else "Closed box")
        off = {r["scene"]: float(r["ms"]) for r in rows if r["mode"] == "off"}
        for i, sc in enumerate(["cornell_open", "cornell_closed"]):
            ax.axhline(off[sc], color=SERIES[i], linewidth=1, alpha=0.45)
            ax.annotate("no compaction", (0.4, off[sc]), xytext=(2, 4), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_xlabel("compact when at most this fraction of the active paths survived")
        ax.set_ylabel("ms per iteration")
        ax.set_title("Adaptive compaction threshold", loc="left")
        ax.invert_xaxis()
        clean(ax)
        ax.legend(loc="center right")
        save(fig, "compaction_threshold.png")


def roulette():
    rows = read("roulette_alive.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    i = 0
    for sc, scn in [("cornell_open", "open"), ("cornell_closed", "closed")]:
        for rr, rrn in [("0", "without RR"), ("1", "with RR")]:
            pts = [(int(r["bounce"]), int(r["alive"]) / 1000) for r in rows if r["scene"] == sc and r["rr"] == rr]
            x, y = zip(*pts)
            line(ax, x, y, SERIES[i], "%s box, %s" % (scn, rrn))
            i += 1
    ax.set_xlabel("bounce")
    ax.set_ylabel("live paths (thousands)")
    ax.set_title("Russian roulette (from bounce 3)", loc="left")
    ax.set_ylim(0, 700)
    clean(ax)
    ax.legend(loc="upper right")
    save(fig, "roulette_alive.png")


STAGES = ["Generate", "Intersect", "Sort", "Shade", "Compact", "Shadow"]


def stacked(ax, labels, breakdowns, totals):
    ypos = list(range(len(labels)))[::-1]
    left = [0.0] * len(labels)
    for s, stage in enumerate(STAGES):
        vals = [b.get(stage, 0.0) for b in breakdowns]
        ax.barh(ypos, vals, left=left, height=0.45, color=SERIES[s], label=stage,
                edgecolor=SURFACE, linewidth=1.5)
        left = [l + v for l, v in zip(left, vals)]
    for y, l, t in zip(ypos, left, totals):
        ax.annotate("%.1f ms" % t, (l, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    clean(ax, xgrid=True)
    ax.grid(axis="y", visible=False)


def sorting():
    rows = read("sorting.csv")
    if not rows:
        return
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    titles = {"cornell": "Cornell box (5 materials)", "materials": "Materials box (10 materials, 5 BSDF types)",
              "textures": "Textured spheres (image + procedural)", "cover": "Cover scene (150k triangles, glass)"}
    fig, axes = plt.subplots(len(scenes), 1, figsize=(8, 2.3 * len(scenes)))
    for ax, sc in zip(axes, scenes):
        rs = [r for r in rows if r["scene"] == sc]
        labels = [r["mode"] for r in rs]
        bds = [{s: float(r[s]) for s in STAGES} for r in rs]
        stacked(ax, labels, bds, [float(r["ms"]) for r in rs])
        ax.set_title(titles.get(sc, sc), loc="left")
        ax.set_xlim(0, max(sum(b.values()) for b in bds) * 1.18)
    axes[-1].set_xlabel("ms per iteration, per-stage breakdown from synchronized CUDA events (label: unsynchronized total)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "sorting.png")


def optimization():
    # Cornell box 800x800, depth 8: measured with --profile right before and
    # after the kernel restructuring commit (see the git history).
    before = {"Generate": 1.033, "Intersect": 51.443, "Sort": 0.0, "Shade": 6.585, "Compact": 55.100, "Shadow": 0.754}
    after = {"Generate": 0.742, "Intersect": 1.560, "Sort": 0.0, "Shade": 2.928, "Compact": 2.524, "Shadow": 0.0}
    fig, ax = plt.subplots(figsize=(8, 2.0))
    stacked(ax, ["base-code structure\n(thrust::partition)", "inlined headers +\nCUB compaction"], [before, after], [115.3, 8.0])
    ax.set_title("Cornell box iteration time before/after restructuring", loc="left")
    ax.set_xlabel("ms per iteration")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in [0, 1, 3, 4, 5]], ["Generate", "Intersect", "Shade", "Compact", "Final gather"],
              loc="lower right", ncol=5, fontsize=8)
    ax.set_xlim(0, 135)
    save(fig, "optimization.png")


def bvh():
    rows = read("bvh_scaling.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    for i, m in enumerate(modes):
        pts = sorted((int(r["triangles"]), float(r["ms"])) for r in rows if r["mode"] == m)
        x, y = zip(*pts)
        line(ax, x, y, SERIES[i], m)
        if m != "BVH + AABB culling":   # overlaps the plain BVH line
            end_label(ax, x[-1], y[-1], "%.1f ms" % y[-1] if y[-1] < 100 else "%.0f ms" % y[-1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax, "y", [1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xlabel("triangles in the mesh")
    ax.set_ylabel("ms per iteration (log)")
    ax.set_title("Ray-mesh intersection: BVH vs brute force (320x320, depth 4)", loc="left")
    clean(ax, xgrid=True)
    ax.legend(loc="upper left")
    save(fig, "bvh_scaling.png")

    leaf = read("bvh_leaf.csv")
    if leaf:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
        x = [r["leaf_size"] for r in leaf]
        for ax, key, title, fmt in [(axes[0], "ms", "Render time (ms / iteration)", "%.1f"),
                                    (axes[1], "build_ms", "BVH build time (ms, CPU)", "%.0f")]:
            vals = [float(r[key]) for r in leaf]
            ax.bar(x, vals, width=0.45, color=SERIES[0])
            for xi, v in zip(x, vals):
                ax.annotate(fmt % v, (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", color=INK2, fontsize=9)
            ax.set_title(title, loc="left")
            ax.set_xlabel("maximum leaf size")
            ax.set_ylim(0, max(vals) * 1.18)
            clean(ax)
        save(fig, "bvh_leaf.png")


def convergence(name, title_map, out, group="strategy"):
    rows = read(name)
    if not rows:
        return
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    fig, axes = plt.subplots(1, len(scenes), figsize=(4.4 * len(scenes), 3.4))
    if len(scenes) == 1:
        axes = [axes]
    for ax, sc in zip(axes, scenes):
        keys = list(dict.fromkeys(r[group] for r in rows if r["scene"] == sc))
        for i, k in enumerate(keys):
            pts = sorted((int(r["spp"]), float(r["rmse"])) for r in rows if r["scene"] == sc and r[group] == k)
            x, y = zip(*pts)
            line(ax, x, y, SERIES[i], k)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        plain_log_axis(ax, "x", sorted({int(r["spp"]) for r in rows if r["scene"] == sc}))
        plain_log_axis(ax, "y")
        ax.set_xlabel("samples per pixel")
        ax.set_title(title_map.get(sc, sc), loc="left")
        clean(ax, xgrid=True)
    axes[0].set_ylabel("RMSE vs reference (log)")
    axes[0].legend(loc="lower left")
    save(fig, out)


def textures():
    rows = read("textures.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [r["material"] for r in rows]
    vals = [float(r["shade_ms"]) for r in rows]
    ypos = list(range(len(labels)))[::-1]
    colors = [SERIES[0] if "image" in l else SERIES[1] if "procedural" in l else MUTED for l in labels]
    ax.barh(ypos, vals, height=0.42, color=colors)
    for y, v in zip(ypos, vals):
        ax.annotate("%.2f ms" % v, (v, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("shading kernel time per iteration (ms, 800x800, depth 4)")
    ax.set_title("Image vs procedural textures", loc="left")
    ax.set_xlim(0, max(vals) * 1.2)
    clean(ax, xgrid=True)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=MUTED, label="untextured"), Patch(color=SERIES[0], label="image texture"),
                       Patch(color=SERIES[1], label="procedural")], loc="lower right")
    save(fig, "textures.png")


def denoiser():
    rows = read("denoiser.csv")
    if not rows:
        return
    key = "display_rmse" if "display_rmse_raw" in rows[0] else "rmse"
    fig, ax = plt.subplots(figsize=(7, 3.6))
    raw = sorted({(int(r["spp"]), float(r[key + "_raw"])) for r in rows})
    x, y = zip(*raw)
    line(ax, x, y, MUTED, "path traced (no denoiser)")
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    for i, m in enumerate(modes):
        pts = sorted((int(r["spp"]), float(r[key + "_denoised"])) for r in rows if r["mode"] == m)
        x, y = zip(*pts)
        line(ax, x, y, SERIES[i], "denoised: " + m)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    plain_log_axis(ax, "x", sorted({int(r["spp"]) for r in rows}))
    plain_log_axis(ax, "y", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xlabel("samples per pixel")
    ax.set_ylabel("RMSE after tone mapping (log)" if key == "display_rmse" else "RMSE vs reference (log)")
    ax.set_title("Open Image Denoise on the cover scene (640x360, vs 8192 spp reference)", loc="left")
    clean(ax, xgrid=True)
    ax.legend(loc="upper right")
    save(fig, "denoiser.png")


def main():
    alive_paths()
    compaction()
    roulette()
    sorting()
    optimization()
    bvh()
    convergence("light_sampling.csv", {"veach_mis": "Glossy plates and four lights", "cornell": "Cornell box"},
                "light_sampling.png")
    convergence("samplers.csv", {"cornell": "Cornell box", "textures": "Textured spheres", "veach_mis": "Glossy plates"},
                "samplers.png", group="sampler")
    textures()
    denoiser()


if __name__ == "__main__":
    main()
