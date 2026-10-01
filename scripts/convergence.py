"""RMSE versus samples per pixel for the random and Sobol samplers.

usage: python convergence.py exe scene.json out.csv [res]
Renders a high sample count reference with the random sampler, then both
samplers at power-of-two sample counts, and writes rows of
(sampler, spp, rmse).
"""
import csv
import os
import subprocess
import sys

import numpy as np

from pfm import read_pfm


def render(exe, scene, out, spp, res, extra, cwd):
    subprocess.run([exe, scene, "--headless", "--spp", str(spp), "--res", res, "--out", out, "--pfm"] + extra,
                   cwd=cwd, check=True, stdout=subprocess.DEVNULL)
    return read_pfm(os.path.join(cwd, f"{out}.{spp}samp.pfm"))


def main():
    exe = os.path.abspath(sys.argv[1])
    scene = os.path.abspath(sys.argv[2])
    out_csv = sys.argv[3]
    res = sys.argv[4] if len(sys.argv) > 4 else "200x200"
    cwd = os.path.join(os.path.dirname(exe), "convergence")
    os.makedirs(cwd, exist_ok=True)
    name = os.path.splitext(os.path.basename(scene))[0]

    ref = render(exe, scene, f"{name}_ref", 16384, res, ["--sampler", "random"], cwd)
    rows = []
    for sampler in ["random", "sobol"]:
        for spp in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            img = render(exe, scene, f"{name}_{sampler}", spp, res, ["--sampler", sampler], cwd)
            rmse = float(np.sqrt(np.mean((img - ref) ** 2)))
            rows.append((sampler, spp, rmse))
            print(f"{sampler:7s} {spp:5d} spp  rmse {rmse:.5f}")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sampler", "spp", "rmse"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
