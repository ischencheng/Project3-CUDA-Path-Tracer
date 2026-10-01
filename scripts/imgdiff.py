"""Compare two rendered PNGs: prints RMSE, max abs difference and identical-pixel ratio."""
import sys

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    d = a - b
    rmse = np.sqrt(np.mean(d * d))
    same = np.mean(np.all(np.abs(d) < 1e-9, axis=2))
    print(f"rmse={rmse:.6f} maxabs={np.abs(d).max():.4f} identical_pixels={same * 100:.2f}%")


if __name__ == "__main__":
    main()
