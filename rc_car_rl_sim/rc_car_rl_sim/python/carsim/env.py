"""RCCarEnv -- a Gymnasium-API-compatible environment (reset/step/render, .observation_space,
.action_space) wrapping the C++ physics+rendering engine via ctypes. No training is implemented
here; see tests/run_full_demo.py for a random-action rollout.

=== Design notes (why the observation/action space looks like this) ===

Action space: Box(7,) in [-1, 1] --
  [0]      steer            -> scaled to +/-STEER_MAX radians, front-wheel servo target
  [1:5]    duty_FL,FR,RL,RR -> per-wheel hub-motor duty cycle (not raw torque -- see physics.hpp's
                                motorTorqueCurve; this matches how a real ESC is actually commanded)
  [5]      duty_flywheel    -> pitch-axis reaction-wheel motor duty cycle
  [6]      cam_pan_rate     -> onboard camera pan servo angular rate command

Observation space: Dict --
  camera(128,128,3) uint8         front-facing onboard camera, panned by the agent (action[6])
  imu_front(6,) / imu_rear(6,)    [ax,ay,az,gx,gy,gz] body-frame accelerometer+gyro at the front
                                   and rear of the chassis (two IMUs, as requested -- lets the
                                   agent infer angular effects like body twist that a single IMU
                                   at the CoM would average away)
  wheel_speeds(4,)                per-wheel encoder reading (rad/s) -- realistic, real ESCs/robots
                                   have this
  steering_angle(1,), flywheel_speed(1,), camera_pan_angle(1,)   -- proprioceptive servo/motor state
  suspension_compression(4,)      linear travel sensor per wheel -- realistic (potentiometer/hall
                                   sensor on the suspension arm)
  wheel_contact(4,)               binary ground-contact per wheel -- realistic (many RC/robotics
                                   platforms have a simple contact or current-spike based estimate)

Deliberately NOT included: ground-truth global position/velocity/orientation. A real car doesn't
have GPS-grade pose without extra sensors we haven't modeled, and training directly against
ground-truth pose would make the policy less representative of a real deployment. True pose is
still available via env.get_debug_state() for logging, reward shaping, and the video renderer --
just not fed to the policy.

Reward (provisional -- not tuned, no training implemented yet): forward progress along the car's
initial heading, minus a small control-effort penalty, minus a termination penalty for flipping.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native import NativeSim
import spaces

PHYSICS_DT = 1.0/240.0
SUBSTEPS_PER_ENV_STEP = 8  # -> ~30 Hz control rate


class RCCarEnv:
    metadata = {"control_hz": 1.0/(PHYSICS_DT*SUBSTEPS_PER_ENV_STEP)}

    def __init__(self, image_size=128, max_episode_seconds=20.0):
        self.image_size = image_size
        self.max_episode_steps = int(max_episode_seconds * self.metadata["control_hz"])
        self._sim = NativeSim()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "camera": spaces.Box(0, 255, shape=(image_size, image_size, 3), dtype=np.uint8),
            "imu_front": spaces.Box(-500.0, 500.0, shape=(6,), dtype=np.float32),
            "imu_rear": spaces.Box(-500.0, 500.0, shape=(6,), dtype=np.float32),
            "wheel_speeds": spaces.Box(-700.0, 700.0, shape=(4,), dtype=np.float32),
            "steering_angle": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "flywheel_speed": spaces.Box(-1700.0, 1700.0, shape=(1,), dtype=np.float32),
            "camera_pan_angle": spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32),
            "suspension_compression": spaces.Box(-0.1, 0.1, shape=(4,), dtype=np.float32),
            "wheel_contact": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
        })

        self._step_count = 0
        self._initial_fwd = np.array([1.0, 0.0, 0.0])
        self._last_progress_pos = np.zeros(3)

    def _raw(self):
        return self._sim.get_observation_raw()

    def _pack_obs(self, raw, camera_img):
        return {
            "camera": camera_img,
            "imu_front": raw[23:29].astype(np.float32),
            "imu_rear": raw[29:35].astype(np.float32),
            "wheel_speeds": raw[0:4].astype(np.float32),
            "steering_angle": np.array([raw[4]], dtype=np.float32),
            "flywheel_speed": np.array([raw[5]], dtype=np.float32),
            "camera_pan_angle": np.array([raw[6]], dtype=np.float32),
            "suspension_compression": raw[7:11].astype(np.float32),
            "wheel_contact": raw[11:15].astype(np.float32),
        }

    def get_debug_state(self):
        """Privileged ground-truth state: NOT part of the observation given to a policy.
        For logging, reward shaping, and rendering the demo video only."""
        raw = self._raw()
        return {
            "pos": raw[37:40], "quat_xyzw": raw[40:44],
            "up_dot_world": raw[35], "sim_time": raw[36],
            "wheel_slip": raw[15:19], "normal_load": raw[19:23],
        }

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._sim.reset()
        self._step_count = 0
        dbg = self.get_debug_state()
        self._last_progress_pos = dbg["pos"].copy()
        raw = self._raw()
        cam = self._sim.render_onboard(self.image_size, self.image_size)
        obs = self._pack_obs(raw, cam)
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._sim.step(action, dt=PHYSICS_DT, n_substeps=SUBSTEPS_PER_ENV_STEP)
        self._step_count += 1

        raw = self._raw()
        cam = self._sim.render_onboard(self.image_size, self.image_size)
        obs = self._pack_obs(raw, cam)
        dbg = self.get_debug_state()

        progress = float(np.dot(dbg["pos"] - self._last_progress_pos, self._initial_fwd))
        self._last_progress_pos = dbg["pos"].copy()
        control_penalty = 0.01 * float(np.sum(action[1:5]**2))
        reward = progress - control_penalty

        flipped = dbg["up_dot_world"] < 0.15
        out_of_bounds = abs(dbg["pos"][0]) > 150 or abs(dbg["pos"][2]) > 150
        terminated = bool(flipped or out_of_bounds)
        if flipped:
            reward -= 5.0
        truncated = self._step_count >= self.max_episode_steps

        info = {"debug_state": dbg}
        return obs, reward, terminated, truncated, info

    def render_chase_cam(self, width=640, height=360, distance=2.6, height_offset=1.1):
        """External cinematic camera for video export -- not part of the agent's observation."""
        dbg = self.get_debug_state()
        pos = dbg["pos"]
        qx, qy, qz, qw = dbg["quat_xyzw"]
        fwd = _quat_rotate((qx, qy, qz, qw), (1, 0, 0))
        cam_pos = pos - np.array(fwd)*distance + np.array([0, height_offset, 0])
        look_at = pos + np.array([0, 0.05, 0])
        cam_fwd = look_at - cam_pos
        cam_fwd = cam_fwd / (np.linalg.norm(cam_fwd)+1e-9)
        return self._sim.render_custom(cam_pos, cam_fwd, (0, 1, 0), np.radians(58), width, height)

    def close(self):
        pass


def _quat_rotate(q_xyzw, v):
    x, y, z, w = q_xyzw
    vx, vy, vz = v
    uvx = y*vz - z*vy
    uvy = z*vx - x*vz
    uvz = x*vy - y*vx
    uuvx = y*uvz - z*uvy
    uuvy = z*uvx - x*uvz
    uuvz = x*uvy - y*uvx
    return (vx + 2*(w*uvx+uuvx), vy + 2*(w*uvy+uuvy), vz + 2*(w*uvz+uuvz))
