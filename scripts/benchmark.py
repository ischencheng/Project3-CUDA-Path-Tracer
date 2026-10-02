"""Runs the performance experiments used in the README and writes CSV files.

usage: python benchmark.py path/to/cis565_path_tracer.exe [experiment ...]

Every measurement renders headless with warm-up iterations excluded. Total
iteration times come from runs without per-stage synchronization; stage
breakdowns and alive-path counts come from separate --profile runs.
"""
import csv
import math
import os
import re
import subprocess
import sys

import numpy as np

from pfm import read_pfm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(ROOT, "scenes")
OUT = os.path.join(ROOT, "analysis")
WORK = os.path.join(ROOT, "build", "bench")
STAGES = ["Generate", "Intersect", "Sort", "Shade", "Compact", "Shadow"]


def run(exe, scene, args, spp=120, warmup=20, profile=False):
    cmd = [exe, scene, "--headless", "--spp", str(spp), "--warmup", str(warmup), "--out", "bench"] + args
    if profile:
        cmd.append("--profile")
    text = subprocess.run(cmd, cwd=WORK, check=True, capture_output=True, text=True).stdout
    result = {}
    m = re.search(r"Rendered \d+ iterations, ([\d.]+) ms/iteration", text)
    result["ms"] = float(m.group(1))
    for stage in STAGES:
        m = re.search(r"^\s+%s\s+([\d.]+) ms" % stage, text, re.M)
        if m:
            result[stage] = float(m.group(1))
    m = re.search(r"alive paths per bounce: ([\d ]+)", text)
    if m:
        result["alive"] = [int(v) for v in m.group(1).split()]
    m = re.search(r"Scene has (\d+) triangles", text)
    result["triangles"] = int(m.group(1)) if m else 0
    m = re.search(r"Denoised in ([\d.]+) ms", text)
    if m:
        result["denoise_ms"] = float(m.group(1))
    return result


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", name)


def scene(name):
    return os.path.join(SCENES, name + ".json")


# ---------------------------------------------------------------------------

def compaction(exe):
    """Stream compaction modes in an open and a closed Cornell box."""
    rows, alive_rows = [], []
    modes = {"off": ["--compact", "off"],
             "thrust::remove_if (every bounce)": ["--compact", "thrust", "--compact-threshold", "1"],
             "CUB select (every bounce)": ["--compact", "cub", "--compact-threshold", "1"],
             "CUB select (adaptive, <= 60% alive)": ["--compact", "cub", "--compact-threshold", "0.6"]}
    for sc in ["cornell_open", "cornell_closed"]:
        for mode, args in modes.items():
            base = args + ["--rr", "0"]
            t = run(exe, scene(sc), base)["ms"]
            p = run(exe, scene(sc), base, spp=60, warmup=10, profile=True)
            rows.append([sc, mode, t] + [p.get(s, 0.0) for s in STAGES])
            if mode == "off":
                alive_rows += [[sc, d, n] for d, n in enumerate(p["alive"])]
            print(sc, mode, t)
    write_csv("compaction.csv", ["scene", "mode", "ms"] + STAGES, rows)
    write_csv("alive_paths.csv", ["scene", "bounce", "alive"], alive_rows)
    sweep = []
    for sc in ["cornell_open", "cornell_closed"]:
        for th in [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
            t = run(exe, scene(sc), ["--compact", "cub", "--compact-threshold", str(th), "--rr", "0"])["ms"]
            sweep.append([sc, th, t])
            print(sc, th, t)
    write_csv("compaction_threshold.csv", ["scene", "threshold", "ms"], sweep)


def roulette(exe):
    """Russian roulette on/off, open vs closed."""
    rows, alive_rows = [], []
    for sc in ["cornell_open", "cornell_closed"]:
        for rr in ["0", "1"]:
            t = run(exe, scene(sc), ["--rr", rr])["ms"]
            p = run(exe, scene(sc), ["--rr", rr], spp=60, warmup=10, profile=True)
            rows.append([sc, rr, t])
            alive_rows += [[sc, rr, d, n] for d, n in enumerate(p["alive"])]
            print(sc, "rr", rr, t)
    write_csv("roulette.csv", ["scene", "rr", "ms"], rows)
    write_csv("roulette_alive.csv", ["scene", "rr", "bounce", "alive"], alive_rows)


SHADING_MODES = {
    "megakernel": [],
    "thrust sort": ["--sort", "thrust"],
    "CUB sort + gather": ["--sort", "cub"],
    "CUB sort, indirect": ["--sort", "indirect"],
    "wavefront queues": ["--wavefront", "1"],
}


def sorting(exe):
    """Material sorting strategies across scenes."""
    rows = []
    for sc, res in [("cornell", "800x800"), ("materials", "800x800"), ("textures", "960x540"), ("cover", "960x540")]:
        for mode, args in SHADING_MODES.items():
            a = args + ["--res", res]
            t = run(exe, scene(sc), a, spp=80, warmup=10)["ms"]
            p = run(exe, scene(sc), a, spp=40, warmup=5, profile=True)
            rows.append([sc, mode, t] + [p.get(s, 0.0) for s in STAGES])
            print(sc, mode, t)
    write_csv("sorting.csv", ["scene", "mode", "ms"] + STAGES, rows)


def make_sphere_obj(path, rings, segments):
    with open(path, "w") as o:
        for i in range(rings + 1):
            th = math.pi * i / rings
            for j in range(segments + 1):
                ph = 2 * math.pi * j / segments
                x, y, z = math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)
                o.write("v %f %f %f\nvn %f %f %f\n" % (x, y, z, x, y, z))
        for i in range(rings):
            for j in range(segments):
                a = i * (segments + 1) + j + 1
                b = a + segments + 1
                o.write("f %d//%d %d//%d %d//%d\n" % (a, a, b, b, a + 1, a + 1))
                o.write("f %d//%d %d//%d %d//%d\n" % (a + 1, a + 1, b, b, b + 1, b + 1))
    return 2 * rings * segments


def bvh(exe):
    """Triangle count scaling with and without the BVH / bounding box culling."""
    import json
    rows = []
    for rings in [8, 16, 32, 64, 128, 256, 512]:
        obj = os.path.join(WORK, "sphere_%d.obj" % rings)
        tris = make_sphere_obj(obj, rings, rings * 2)
        sc = {
            "Materials": {"light": {"TYPE": "Emitting", "RGB": [1, 1, 1], "EMITTANCE": 5},
                          "white": {"TYPE": "Diffuse", "RGB": [0.8, 0.8, 0.8]},
                          "red": {"TYPE": "Diffuse", "RGB": [0.8, 0.2, 0.2]}},
            "Camera": {"RES": [320, 320], "FOVY": 40.0, "ITERATIONS": 10, "DEPTH": 4, "FILE": "bvh",
                       "EYE": [0, 2, 9], "LOOKAT": [0, 1.5, 0], "UP": [0, 1, 0]},
            "Objects": [
                {"TYPE": "cube", "MATERIAL": "white", "TRANS": [0, -0.05, 0], "SCALE": [10, 0.1, 10]},
                {"TYPE": "sphere", "MATERIAL": "light", "TRANS": [3, 6, 3], "SCALE": [2, 2, 2]},
                {"TYPE": "mesh", "FILE": os.path.basename(obj), "MATERIAL": "red", "TRANS": [0, 1.5, 0], "SCALE": [1.5, 1.5, 1.5]},
            ],
        }
        path = os.path.join(WORK, "bvh_%d.json" % rings)
        json.dump(sc, open(path, "w"))
        for name, args in [("BVH + AABB culling", ["--bvh", "1", "--cull", "1"]),
                           ("BVH", ["--bvh", "1", "--cull", "0"]),
                           ("AABB culling only", ["--bvh", "0", "--cull", "1"]),
                           ("brute force", ["--bvh", "0", "--cull", "0"])]:
            if args[1] == "0" and tris > 20000:
                continue    # minutes per iteration (and the Windows GPU watchdog)
            spp = 40 if args[1] == "1" else 4
            r = run(exe, path, args, spp=spp, warmup=2 if args[1] == "0" else 10)
            rows.append([tris, name, r["ms"]])
            print(tris, name, r["ms"])
    write_csv("bvh_scaling.csv", ["triangles", "mode", "ms"], rows)


def bvh_leaf(exe):
    """BVH leaf size on the cover scene (135k + 15k triangles)."""
    rows = []
    for leaf in [1, 2, 4, 8, 16, 32]:
        cmd = [exe, scene("cover"), "--headless", "--spp", "50", "--warmup", "10", "--res", "960x540",
               "--bvh-leaf", str(leaf), "--out", "bench"]
        text = subprocess.run(cmd, cwd=WORK, check=True, capture_output=True, text=True).stdout
        ms = float(re.search(r"([\d.]+) ms/iteration", text).group(1))
        build = sum(float(v) for v in re.findall(r"built in ([\d.]+) ms", text))
        nodes = sum(int(v) for v in re.findall(r"(\d+) nodes", text))
        depth = max(int(v) for v in re.findall(r"depth (\d+)", text))
        rows.append([leaf, ms, build, nodes, depth])
        print("leaf", leaf, ms, build, nodes)
    write_csv("bvh_leaf.csv", ["leaf_size", "ms", "build_ms", "nodes", "depth"], rows)


def rmse_vs_reference(exe, scene_path, variants, spps, res, ref_args, ref_spp):
    ref_name = "ref_" + os.path.splitext(os.path.basename(scene_path))[0]
    subprocess.run([exe, scene_path, "--headless", "--spp", str(ref_spp), "--res", res, "--out", ref_name, "--pfm"]
                   + ref_args, cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    ref = read_pfm(os.path.join(WORK, "%s.%dsamp.pfm" % (ref_name, ref_spp)))
    rows = []
    for name, args in variants.items():
        for spp in spps:
            out = "v"
            subprocess.run([exe, scene_path, "--headless", "--spp", str(spp), "--res", res, "--out", out, "--pfm"]
                           + args, cwd=WORK, check=True, stdout=subprocess.DEVNULL)
            img = read_pfm(os.path.join(WORK, "%s.%dsamp.pfm" % (out, spp)))
            rmse = float(np.sqrt(np.mean((img - ref) ** 2)))
            rows.append([name, spp, rmse])
            print(name, spp, rmse)
    return rows


def light_sampling(exe):
    """BSDF sampling vs next event estimation vs MIS."""
    variants = {"BSDF sampling": ["--nee", "0"], "light sampling (NEE)": ["--nee", "1", "--mis", "0"],
                "MIS": ["--nee", "1", "--mis", "1"]}
    rows = []
    for sc in ["veach_mis", "cornell"]:
        r = rmse_vs_reference(exe, scene(sc), variants, [4, 16, 64, 256], "300x200" if sc == "veach_mis" else "240x240",
                              ["--nee", "1", "--mis", "1"], 16384)
        rows += [[sc] + row for row in r]
    times = []
    for sc in ["veach_mis", "cornell"]:
        for name, args in variants.items():
            times.append([sc, name, run(exe, scene(sc), args)["ms"]])
    write_csv("light_sampling.csv", ["scene", "strategy", "spp", "rmse"], rows)
    write_csv("light_sampling_time.csv", ["scene", "strategy", "ms"], times)


def samplers(exe):
    """Random vs Owen-scrambled Sobol."""
    variants = {"random": ["--sampler", "random"], "Sobol": ["--sampler", "sobol"]}
    rows = []
    for sc, res in [("cornell", "240x240"), ("textures", "320x180"), ("veach_mis", "300x200")]:
        r = rmse_vs_reference(exe, scene(sc), variants, [1, 4, 16, 64, 256, 1024], res, ["--sampler", "random"], 32768)
        rows += [[sc] + row for row in r]
    write_csv("samplers.csv", ["scene", "sampler", "spp", "rmse"], rows)


def textures(exe):
    """Shading cost of image textures vs procedural textures."""
    import json
    variants = {
        "flat color": {"TYPE": "PBR", "RGB": [0.8, 0.8, 0.8], "ROUGHNESS": 0.4},
        "image texture": {"TYPE": "PBR", "RGB": [1, 1, 1], "ROUGHNESS": 0.4, "TEXTURE": "textures/marble_01_diff_1k.jpg", "UV_SCALE": [4, 4]},
        "image + normal map": {"TYPE": "PBR", "RGB": [1, 1, 1], "ROUGHNESS": 0.4, "TEXTURE": "textures/marble_01_diff_1k.jpg",
                               "NORMAL_MAP": "textures/marble_01_nor_gl_1k.jpg", "UV_SCALE": [4, 4]},
        "procedural checker": {"TYPE": "PBR", "RGB": [0.8, 0.8, 0.8], "ROUGHNESS": 0.4, "PROCEDURAL": {"TYPE": "checker", "COLOR2": [0.1, 0.1, 0.1], "SCALE": 16}},
        "procedural wood": {"TYPE": "PBR", "RGB": [0.6, 0.4, 0.2], "ROUGHNESS": 0.4, "PROCEDURAL": {"TYPE": "wood", "COLOR2": [0.3, 0.15, 0.05], "SCALE": 3}},
        "procedural marble": {"TYPE": "PBR", "RGB": [0.9, 0.9, 0.9], "ROUGHNESS": 0.4, "PROCEDURAL": {"TYPE": "marble", "COLOR2": [0.2, 0.2, 0.3], "SCALE": 3}},
        "procedural bump": {"TYPE": "PBR", "RGB": [0.8, 0.8, 0.8], "ROUGHNESS": 0.4, "BUMP": {"STRENGTH": 0.05, "SCALE": 4}},
    }
    rows = []
    for name, mat in variants.items():
        sc = {
            "Materials": {"m": mat},
            "Camera": {"RES": [800, 800], "FOVY": 30.0, "ITERATIONS": 10, "DEPTH": 4, "FILE": "tex",
                       "EYE": [0, 0, 3.2], "LOOKAT": [0, 0, 0], "UP": [0, 1, 0]},
            "Environment": {"COLOR": [0.8, 0.85, 0.9]},
            "Objects": [{"TYPE": "cube", "MATERIAL": "m", "TRANS": [0, 0, 0], "SCALE": [6, 6, 0.1]}],
        }
        path = os.path.join(SCENES, "_bench_texture.json")
        json.dump(sc, open(path, "w"))
        t = run(exe, path, [])["ms"]
        p = run(exe, path, [], spp=60, warmup=10, profile=True)
        rows.append([name, t, p["Shade"]])
        print(name, t, p["Shade"])
    os.remove(os.path.join(SCENES, "_bench_texture.json"))
    write_csv("textures.csv", ["material", "ms", "shade_ms"], rows)


def display(img):
    """ACES filmic + gamma 2.2, clamped: the transform the showcase images use."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.clip((img * (a * img + b)) / (img * (c * img + d) + e), 0.0, 1.0)
    return x ** (1.0 / 2.2)


def denoiser(exe):
    """OIDN quality (RMSE vs reference, linear and display space) and time."""
    res = "640x360"
    sc = scene("cover")
    ref_spp = 8192
    subprocess.run([exe, sc, "--headless", "--spp", str(ref_spp), "--res", res, "--out", "dn_ref", "--pfm"],
                   cwd=WORK, check=True, stdout=subprocess.DEVNULL)
    ref = read_pfm(os.path.join(WORK, "dn_ref.%dsamp.pfm" % ref_spp))
    rows = []
    for spp in [1, 4, 16, 64, 256]:
        for name, args in [("color only", ["--denoise-aux", "0"]), ("albedo + normal", ["--denoise-prefilter", "0"]),
                           ("albedo + normal, prefiltered", [])]:
            text = subprocess.run([exe, sc, "--headless", "--spp", str(spp), "--res", res, "--out", "dn", "--pfm",
                                   "--denoise", "1"] + args, cwd=WORK, check=True, capture_output=True, text=True).stdout
            ms = float(re.search(r"Denoised in ([\d.]+) ms", text).group(1))
            raw = read_pfm(os.path.join(WORK, "dn.%dsamp.pfm" % spp))
            den = read_pfm(os.path.join(WORK, "dn.%dsamp.denoised.pfm" % spp))
            rows.append([spp, name, float(np.sqrt(np.mean((raw - ref) ** 2))),
                         float(np.sqrt(np.mean((den - ref) ** 2))), ms,
                         float(np.sqrt(np.mean((display(raw) - display(ref)) ** 2))),
                         float(np.sqrt(np.mean((display(den) - display(ref)) ** 2)))])
            print(spp, name, rows[-1])
    write_csv("denoiser.csv", ["spp", "mode", "rmse_raw", "rmse_denoised", "denoise_ms",
                               "display_rmse_raw", "display_rmse_denoised"], rows)


def features(exe):
    """Cost of motion blur, depth of field and environment lighting."""
    rows = []
    rows.append(["motion blur off", run(exe, scene("motion_blur"), ["--motion", "0"])["ms"]])
    rows.append(["motion blur on", run(exe, scene("motion_blur"), ["--motion", "1"])["ms"]])
    write_csv("features.csv", ["config", "ms"], rows)


EXPERIMENTS = {
    "compaction": compaction, "roulette": roulette, "sorting": sorting, "bvh": bvh, "bvh_leaf": bvh_leaf,
    "light_sampling": light_sampling, "samplers": samplers, "textures": textures, "denoiser": denoiser,
    "features": features,
}


def main():
    exe = os.path.abspath(sys.argv[1])
    names = sys.argv[2:] or list(EXPERIMENTS)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    for name in names:
        print("==", name)
        EXPERIMENTS[name](exe)


if __name__ == "__main__":
    main()
