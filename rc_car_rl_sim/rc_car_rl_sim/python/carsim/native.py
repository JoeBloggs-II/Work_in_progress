"""Thin ctypes bridge to libcarsim.so. No pybind11/pybullet available in this environment,
so the C++ engine exposes a flat extern "C" API and we bind it directly -- zero extra
Python dependencies beyond the standard library + numpy."""
import ctypes
import os
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_here, "libcarsim.so")
_lib = ctypes.CDLL(_lib_path)

_lib.cs_create.restype = ctypes.c_void_p
_lib.cs_create.argtypes = []
_lib.cs_destroy.argtypes = [ctypes.c_void_p]
_lib.cs_reset.argtypes = [ctypes.c_void_p]
_lib.cs_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_int]
_lib.cs_get_observation.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
_lib.cs_render_onboard.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
_lib.cs_render_custom.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double), ctypes.c_double,
                                   ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
_lib.cs_set_pose.argtypes = [ctypes.c_void_p] + [ctypes.c_double]*7

OBS_RAW_LEN = 44


class NativeSim:
    """One RC car simulation instance. Not thread-safe; create one per environment."""

    def __init__(self):
        self._h = _lib.cs_create()
        self._onboard_buf_cache = {}
        self._custom_buf_cache = {}

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            _lib.cs_destroy(h)

    def reset(self):
        _lib.cs_reset(self._h)

    def step(self, action7, dt=1.0/240.0, n_substeps=8):
        arr = (ctypes.c_double*7)(*[float(x) for x in action7])
        _lib.cs_step(self._h, arr, dt, n_substeps)

    def get_observation_raw(self):
        buf = (ctypes.c_double*OBS_RAW_LEN)()
        _lib.cs_get_observation(self._h, buf)
        return np.array(buf, dtype=np.float64)

    def _get_buf(self, cache, width, height):
        key = (width, height)
        buf = cache.get(key)
        if buf is None:
            buf = (ctypes.c_ubyte*(width*height*3))()
            cache[key] = buf
        return buf

    def render_onboard(self, width=128, height=128):
        # Reusing one ctypes buffer per (width,height) avoids reallocating+zeroing a fresh
        # buffer every single frame -- measurably cheaper over a multi-hundred-frame rollout.
        buf = self._get_buf(self._onboard_buf_cache, width, height)
        _lib.cs_render_onboard(self._h, width, height, buf)
        return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()

    def render_custom(self, cam_pos, cam_fwd, cam_up, fov_y_rad, width, height):
        cp = (ctypes.c_double*3)(*cam_pos)
        cf = (ctypes.c_double*3)(*cam_fwd)
        cu = (ctypes.c_double*3)(*cam_up)
        buf = self._get_buf(self._custom_buf_cache, width, height)
        _lib.cs_render_custom(self._h, cp, cf, cu, float(fov_y_rad), width, height, buf)
        return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3).copy()

    def set_pose(self, pos, quat_xyzw):
        _lib.cs_set_pose(self._h, *[float(v) for v in pos], *[float(v) for v in quat_xyzw])
