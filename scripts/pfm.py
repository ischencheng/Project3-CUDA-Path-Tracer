"""Minimal reader for the .pfm files written with --pfm."""
import numpy as np


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().strip()
        channels = 3 if header == b"PF" else 1
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=endian + "f4")
    # rows are stored bottom to top
    return np.flipud(data.reshape(height, width, channels))
