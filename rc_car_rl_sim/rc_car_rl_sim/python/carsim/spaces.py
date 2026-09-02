"""Minimal stand-ins for gymnasium.spaces.Box / Dict. No network access was available to
install gymnasium in this environment, so this replicates just the surface area env.py needs
(shape/dtype/low/high/sample/contains). If gymnasium is available in your environment, you can
ignore this module and wrap RCCarEnv's spaces with the real thing -- the shapes match exactly."""
import numpy as np


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.asarray(low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.asarray(high, dtype=dtype)
        self.shape = shape if shape is not None else self.low.shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)

    def __repr__(self):
        return f"Box(shape={self.shape}, dtype={self.dtype})"


class Dict:
    def __init__(self, spaces: dict):
        self.spaces = spaces

    def sample(self):
        return {k: v.sample() for k, v in self.spaces.items()}

    def __repr__(self):
        return f"Dict({self.spaces})"
