"""White furnace test: a sphere inside a uniform white environment.

A BSDF that conserves energy and is sampled with correct weights makes the
sphere indistinguishable from the background (radiance 1). Single-scattering
microfacet models lose some energy at high roughness, so they are expected to
read slightly below 1.

usage: python furnace.py path/to/cis565_path_tracer.exe [spp]
"""
import json
import os
import subprocess
import sys

import numpy as np

from pfm import read_pfm

MATERIALS = {
    "diffuse": {"TYPE": "Diffuse", "RGB": [1, 1, 1]},
    "mirror": {"TYPE": "Specular", "RGB": [1, 1, 1], "ROUGHNESS": 0.0},
    "metal_r0.3": {"TYPE": "Specular", "RGB": [1, 1, 1], "ROUGHNESS": 0.3},
    "metal_r0.8": {"TYPE": "Specular", "RGB": [1, 1, 1], "ROUGHNESS": 0.8},
    "glass": {"TYPE": "Refractive", "RGB": [1, 1, 1], "IOR": 1.5},
    "glass_r0.3": {"TYPE": "Refractive", "RGB": [1, 1, 1], "IOR": 1.5, "ROUGHNESS": 0.3},
    "pbr_dielectric_r0.4": {"TYPE": "PBR", "RGB": [1, 1, 1], "ROUGHNESS": 0.4, "METALLIC": 0.0},
    "pbr_metal_r0.2": {"TYPE": "PBR", "RGB": [1, 1, 1], "ROUGHNESS": 0.2, "METALLIC": 1.0},
}


def scene(material):
    return {
        "Materials": {"m": material},
        "Camera": {"RES": [128, 128], "FOVY": 20.0, "ITERATIONS": 256, "DEPTH": 64,
                   "FILE": "furnace", "EYE": [0, 0, 6], "LOOKAT": [0, 0, 0], "UP": [0, 1, 0]},
        "Environment": {"COLOR": [1, 1, 1]},
        "Objects": [{"TYPE": "sphere", "MATERIAL": "m", "TRANS": [0, 0, 0],
                     "ROTAT": [0, 0, 0], "SCALE": [2, 2, 2]}],
    }


def main():
    exe = os.path.abspath(sys.argv[1])
    spp = sys.argv[2] if len(sys.argv) > 2 else "256"
    out = os.path.join(os.path.dirname(exe), "furnace")
    os.makedirs(out, exist_ok=True)
    for name, mat in MATERIALS.items():
        path = os.path.join(out, name + ".json")
        with open(path, "w") as f:
            json.dump(scene(mat), f)
        subprocess.run([exe, path, "--headless", "--spp", spp, "--out", name, "--pfm"],
                       cwd=out, check=True, stdout=subprocess.DEVNULL)
        img = read_pfm(os.path.join(out, f"{name}.{spp}samp.pfm"))
        h, w, _ = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        inside = (xx - w / 2 + 0.5) ** 2 + (yy - h / 2 + 0.5) ** 2 < (0.3 * w) ** 2
        sphere = img[inside].mean()
        print(f"{name:22s} sphere mean = {sphere:.4f}  (background {img[~inside].mean():.4f})")


if __name__ == "__main__":
    main()
