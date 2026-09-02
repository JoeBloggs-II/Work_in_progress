# RC Car RL Simulation

A from-scratch, headless RC car physics simulation with a WGS gym-style RL environment: real
per-wheel torque-curve motors, suspension, tire slip, aerodynamics, gyroscopic coupling, a
pitch-axis reaction-wheel flywheel, a panning onboard camera, and dual IMUs -- built in C++
(no Bullet/PyBullet/Box2D; none were installable here, see "Why no physics engine library"
below) and driven from Python via `ctypes`.

No training is implemented. The environment currently runs a scripted test suite and then hands
control to a random policy, per the brief.

## Quick start

```
./build_and_run.sh
```

That's it — one command builds the C++ library, runs the automated physics regression suite
(25 checks, including a 50-second adversarial random-action stress test), smoke-tests the
Python environment, and regenerates `output/rc_car_test_video.mp4`. Takes about 15-20 seconds
total on a single CPU core.

Requires: g++ with C++17, Python 3 with `numpy`, `imageio` (+ `imageio-ffmpeg` or system
`ffmpeg`), and `Pillow`. Nothing else -- no pip installs beyond those, no GPU, no display
server. The video is already included in this zip if you just want to look at it first.

Everything renders through a software rasterizer (see "Why no GPU rendering" below), so this
runs identically on a CI box with no display and no GPU.

### Performance

The whole pipeline (physics + two camera renders + frame compositing + H.264 encoding, all on
one CPU core, no GPU) runs faster than real-time: it generates the 27-second, 810-frame test
video in about 15-18 seconds. Per-call costs, profiled directly:

| operation                     | cost       |
|--------------------------------|-----------|
| physics step (8 substeps)      | ~0.1 ms   |
| onboard camera render (128x128)| ~0.3 ms   |
| chase-cam render (640x360)     | ~4 ms     |
| frame compositing (PIL overlay)| ~3 ms     |
| video frame encode (H.264)     | ~3 ms     |

The chase-cam render is the single largest cost, dominated by fill rate (pixels touched), not
triangle count -- the ground is tiled coarsely (12m tiles) since the checker pattern is computed
per-pixel from world position rather than per-tile, so tile size only affects culling
granularity, not visual detail. The rasterizer also reuses its color/depth buffers and the
ctypes-side image buffers across calls rather than reallocating every frame.

### Re-running just the regression tests

```
cd cpp/tests && g++ -O2 -std=c++17 -I.. -o test_physics test_physics.cpp && ./test_physics
```

## What's in the video

Seven back-to-back phases, each with an on-screen label and a live telemetry readout, plus a
picture-in-picture of the agent's own onboard camera feed:

1. **Settle** -- no input, confirms suspension/gravity/rest state
2. **Forward drive**
3. **Brake + reverse**
4. **Controlled turn**
5. **Ramp jump** -- demonstrates the wheel gyroscopic coupling and flywheel attitude control:
   the flywheel spins up once airborne, visibly affecting pitch
6. **Camera pan sweep**
7. **Random policy** (untrained) -- the RL environment's action space exercised by i.i.d.
   sampling with a short zero-order hold (see `carsim/random_agent.py`)

## Architecture

```
cpp/
  vecmath.hpp   Vec3 / Quat / Mat4 -- no external math library
  physics.hpp   the car: chassis rigid body + 4 wheels + flywheel + aero (see below)
  render.hpp    software triangle rasterizer + scene geometry builders
  capi.cpp      flat extern "C" API over the above, for ctypes
  tests/
    test_physics.cpp   25-check automated regression suite (see "Performance" for how to run)
python/carsim/
  native.py     ctypes bindings to libcarsim.so
  spaces.py     minimal Box/Dict (gymnasium wasn't installable here -- see below)
  env.py        RCCarEnv: the actual Gym-API environment (reset/step/render, spaces)
  random_agent.py
python/tests/
  run_full_demo.py   the scripted test suite + video export
output/
  rc_car_test_video.mp4   pre-rendered output of the command above
build_and_run.sh   one-command build + test + demo
```

## The car

- **Chassis**: 2.8 kg, 0.42 x 0.22 x 0.12 m box. Local axes: X=forward, Y=up, Z=right.
- **Wheels**: 4, independently driven (hub motors, not a shared differential), radius 6 cm,
  wheelbase 0.28 m, track 0.25 m (wheels sit slightly outside the body -- see "Tuning" below).
  Front two steer.
- **Flywheel** (the "single flywheel at a point along the body" from the brief): mounted at
  chassis center, spin axis = lateral (pitch axis). **Why pitch, and why there**: the wheels'
  own gyroscopic coupling already acts mostly on pitch during a jump (that was the original,
  separate ask this project grew out of), so a pitch-axis reaction wheel is the natural
  complementary actuator -- it gives the agent authority to control the car's attitude
  specifically during the part of a jump where the wheels have no ground reaction to work with.
  This is the same principle as a reaction-wheel unicycle or a CubeSat attitude control wheel.
  Dimensions: 4 cm radius x 2 cm thick, steel-density (7000 kg/m^3) -> ~0.70 kg, chosen dense
  and compact the way a real reaction wheel would be, sized to store meaningful angular
  momentum (~0.7 kg*m^2/s at its ~1600 rad/s free-spin speed) without dominating the car's mass.
- **Camera**: front-mounted (near the nose), single-axis pan servo, +/-94 degrees range,
  128x128 RGB by default.
- **Two IMUs**: one at the front, one at the rear of the chassis (`physics.hpp::readIMU`),
  each reporting real 6-axis specific-force + angular-rate as a physical accelerometer/gyro
  would (i.e. it reads ~9.81 m/s^2 "up" at rest, not zero -- this is specific force, not
  coordinate acceleration). Two IMUs (rather than one at the CoM) let the agent see effects a
  single central IMU would average away, and is closer to how a real chassis-twist-sensitive
  setup would actually be instrumented.

## Sensors & actions (see `env.py` docstring for the full rationale)

Action space is `Box(7,)` in [-1,1]: steering target, four wheel-motor duty cycles, flywheel
duty cycle, camera pan rate. Observation space is a `Dict` of: the onboard camera image, both
IMUs, per-wheel encoder speed, steering angle, flywheel speed, camera pan angle, per-wheel
suspension compression, and per-wheel binary ground contact. Deliberately **not** included:
ground-truth global position/velocity/orientation -- a real car doesn't have GPS-grade pose for
free, and training against it would make the resulting policy less representative of a real
deployment. True pose is still available via `env.get_debug_state()`, just not fed to the
policy -- that's what the video renderer and reward function use.

## Why no physics engine library, and no GPU rendering

This environment has no outbound network access (pip installs of `pybullet`, `pybind11`, and
`gymnasium` were all rejected -- `403 Host not in allowlist`, even though some of those hosts
are nominally allowlisted; `apt` was equally unreachable), and no GPU (`/dev/dri` doesn't
exist). So: the physics is a from-scratch spring-damper "raycast vehicle" style model (chassis
as the one true 6-DOF rigid body; wheels and flywheel are scalar rotational DOFs coupled to it
through forces/torques -- no generalized multibody solver needed), and rendering is a from-
scratch software triangle rasterizer with a Z-buffer and flat shading. Both are plain C++17
with no dependencies beyond the standard library, which also means this environment has no
external dependencies to break on someone else's machine beyond a C++ compiler.

If you *do* have network access and want to swap in PyBullet, MuJoCo, or a real renderer later,
the C API in `capi.cpp` is the seam to replace -- `env.py` doesn't know or care that the
backend is a hand-rolled rasterizer.

## Known characteristics / honest limitations

- **Pitch response under sustained hard throttle.** Each wheel is an independent hub motor, and
  its motor reaction torque genuinely acts on the chassis (Newton's third law at the axle
  mount) -- this is real physics, not an artifact. At full throttle with all 4 wheels, this
  produces a noticeable nose-up pitch tendency, tuned down (peak torque + chassis inertia) to
  stay recoverable rather than eliminated outright, since a light 4-hub-motor buggy plausibly
  really would be wheelie-prone. If you want a firmly planted car regardless of throttle input,
  lower `WHEEL_PEAK_TORQUE` in `physics.hpp` further.
- **The rasterizer does not clip triangles at the near plane**, it only trivially rejects a
  triangle when all three vertices fail; a triangle straddling the plane gets a distorted edge
  for one frame rather than a clean cut. Cosmetic only, not physics-affecting.
- **Rollover is possible** under aggressive steering + throttle combinations at speed (a light,
  fairly narrow-track car cornering near its tire-grip limit will roll, same as a real one
  would) -- `env.py` terminates the episode when this happens (`up_dot_world < 0.15`) rather
  than trying to make it physically impossible.
- **Reward function is provisional** -- forward-progress shaping with a control-effort penalty
  and a flip penalty. It has not been tuned against any training run, since no training is
  implemented yet.
