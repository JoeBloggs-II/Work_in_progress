# WebGPU Raytraced Audio Demo

A self-contained browser experiment that couples:

1. **Manual ray tracing in a WebGPU compute shader**
   - Six room planes.
   - One spherical obstacle.
   - Multiple specular reflections.
   - Per-surface reflection/absorption.
   - A finite-radius listener.
   - Direct + reflected source-to-listener paths.

2. **GPU impulse-response estimation**
   - Thousands of stochastic rays.
   - Reflection-path energy is accumulated into an atomic fixed-point delay histogram.
   - The histogram is copied back to the CPU and normalized.

3. **Web Audio rendering**
   - A live oscillator is convolved with the GPU-estimated room impulse response using `ConvolverNode`.
   - The impulse response is regenerated whenever the room/tracing settings change.

4. **Visualization**
   - Top-down room view.
   - GPU-produced first/second reflection points for debug rays.
   - Source, listener and obstacle.
   - Live impulse-response graph.

## Run it

From this directory:

```bash
python3 -m http.server 8000
```

Then open:

http://localhost:8000/

WebGPU requires a secure context, and `localhost` is treated as secure by browsers.

## Notes

This is intentionally "raytraced" without relying on a browser-specific hardware ray-tracing extension. The compute shader performs the geometric intersections itself. That makes the demo more portable, though performance is lower than a dedicated RT pipeline.

The acoustic model is an educational approximation rather than a production room-acoustics solver. It does not model diffraction, frequency-dependent materials, HRTF/spatialization, air absorption, wave interference, or phase. The useful result is a live, physically-inspired reflection tail that is recomputed from the 3D room geometry.

The room view is top-down; the z coordinate still participates in the GPU tracing.
