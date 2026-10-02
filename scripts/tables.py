"""Prints the README data tables (markdown) from the CSVs in analysis/."""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "analysis")


def read(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pivot(rows, row_key, col_key, val_key, fmt):
    cols = list(dict.fromkeys(r[col_key] for r in rows))
    keys = list(dict.fromkeys(r[row_key] for r in rows))
    out = []
    for k in keys:
        line = [k]
        for c in cols:
            v = [r[val_key] for r in rows if r[row_key] == k and r[col_key] == c]
            line.append(fmt % float(v[0]) if v else "-")
        out.append(line)
    return cols, out


def main():
    which = sys.argv[1:]
    if not which or "compaction" in which:
        rows = read("compaction.csv")
        cols, body = pivot(rows, "mode", "scene", "ms", "%.1f")
        print(table(["compaction"] + [c.replace("cornell_", "") + " box (ms)" for c in cols], body), "\n")
    if not which or "roulette" in which:
        rows = read("roulette.csv")
        cols, body = pivot(rows, "scene", "rr", "ms", "%.1f")
        print(table(["scene", "RR off (ms)", "RR on (ms)"], [[b[0].replace("cornell_", "") + " box"] + b[1:] for b in body]), "\n")
    if not which or "sorting" in which:
        rows = read("sorting.csv")
        cols, body = pivot(rows, "mode", "scene", "ms", "%.1f")
        print(table(["shading strategy"] + [c + " (ms)" for c in cols], body), "\n")
        cols, body = pivot(rows, "mode", "scene", "Shade", "%.2f")
        print(table(["shading kernel time"] + [c + " (ms)" for c in cols], body), "\n")
    if not which or "bvh" in which:
        rows = read("bvh_scaling.csv")
        cols, body = pivot(rows, "triangles", "mode", "ms", "%.2f")
        print(table(["triangles"] + [c + " (ms)" for c in cols], body), "\n")
        rows = read("bvh_leaf.csv")
        print(table(["max leaf size", "ms / iteration", "build (ms)", "nodes", "depth"],
                    [[r["leaf_size"], "%.1f" % float(r["ms"]), "%.0f" % float(r["build_ms"]), r["nodes"], r["depth"]] for r in rows]), "\n")
    if not which or "light" in which:
        rows = read("light_sampling.csv")
        for sc in dict.fromkeys(r["scene"] for r in rows):
            rs = [r for r in rows if r["scene"] == sc]
            cols, body = pivot(rs, "spp", "strategy", "rmse", "%.4f")
            print(sc)
            print(table(["spp"] + [c + " RMSE" for c in cols], body), "\n")
        rows = read("light_sampling_time.csv")
        cols, body = pivot(rows, "strategy", "scene", "ms", "%.2f")
        print(table(["strategy"] + [c + " (ms)" for c in cols], body), "\n")
    if not which or "samplers" in which:
        rows = read("samplers.csv")
        for sc in dict.fromkeys(r["scene"] for r in rows):
            rs = [r for r in rows if r["scene"] == sc]
            cols, body = pivot(rs, "spp", "sampler", "rmse", "%.4f")
            print(sc)
            print(table(["spp"] + [c + " RMSE" for c in cols], body), "\n")
    if not which or "textures" in which:
        rows = read("textures.csv")
        print(table(["material", "iteration (ms)", "shading kernel (ms)"],
                    [[r["material"], "%.2f" % float(r["ms"]), "%.2f" % float(r["shade_ms"])] for r in rows]), "\n")
    if not which or "denoiser" in which:
        rows = read("denoiser.csv")
        key = "display_rmse" if "display_rmse_raw" in rows[0] else "rmse"
        modes = list(dict.fromkeys(r["mode"] for r in rows))
        body = []
        for spp in dict.fromkeys(r["spp"] for r in rows):
            rs = [r for r in rows if r["spp"] == spp]
            body.append([spp, "%.4f" % float(rs[0][key + "_raw"])]
                        + ["%.4f" % float(next(r for r in rs if r["mode"] == m)[key + "_denoised"]) for m in modes])
        print(table(["spp", "raw RMSE"] + ["denoised: " + m for m in modes], body), "\n")
        print("denoise ms:", ", ".join("%s %.0f" % (r["mode"], float(r["denoise_ms"])) for r in rows if r["spp"] == "64"))
    if not which or "features" in which:
        rows = read("features.csv")
        print(table(["config", "ms"], [[r["config"], "%.2f" % float(r["ms"])] for r in rows]), "\n")


if __name__ == "__main__":
    main()
