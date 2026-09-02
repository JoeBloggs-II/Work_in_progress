"""Runs a scripted test suite exercising every subsystem (settle, forward, reverse, turning,
ramp jump with gyroscopic + flywheel attitude control, camera pan), then hands control to a
random policy, recording the whole thing to output/rc_car_test_video.mp4.

Run: python3 tests/run_full_demo.py
"""
import os
import sys
import time
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from carsim.env import RCCarEnv
from carsim.random_agent import RandomAgent

W, H = 640, 360
FPS = 30
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output", "rc_car_test_video.mp4")


def zero_action():
    return np.zeros(7, dtype=np.float64)


def compose_frame(chase_rgb, onboard_rgb, phase_label, telemetry_lines):
    img = Image.fromarray(chase_rgb)
    draw = ImageDraw.Draw(img)

    # picture-in-picture onboard camera feed, top-right
    pip = Image.fromarray(onboard_rgb).resize((150, 150))
    img.paste(pip, (W-160, 10))
    draw.rectangle([W-160, 10, W-10, 160], outline=(255, 255, 255), width=2)
    draw.text((W-158, 162), "onboard cam", fill=(255, 255, 255))

    # phase label + telemetry, top-left
    draw.rectangle([8, 8, 300, 30+16*len(telemetry_lines)], fill=(10, 12, 14))
    draw.text((14, 12), phase_label, fill=(255, 170, 40))
    for i, line in enumerate(telemetry_lines):
        draw.text((14, 32+16*i), line, fill=(220, 220, 220))

    return np.array(img)


def telemetry_for(env, extra=None):
    dbg = env.get_debug_state()
    lines = [
        f"pos=({dbg['pos'][0]:.2f},{dbg['pos'][1]:.2f},{dbg['pos'][2]:.2f})",
        f"upDotWorld={dbg['up_dot_world']:.2f}",
    ]
    if extra:
        lines.extend(extra)
    return lines


def run_phase(env, writer, label, seconds, action_fn, extra_telemetry_fn=None):
    n_steps = int(seconds * FPS)
    for i in range(n_steps):
        action = action_fn(i, n_steps)
        obs, reward, terminated, truncated, info = env.step(action)
        chase = env.render_chase_cam(W, H)
        extra = extra_telemetry_fn(env, obs, i) if extra_telemetry_fn else None
        frame = compose_frame(chase, obs["camera"], label, telemetry_for(env, extra))
        writer.append_data(frame)
        if terminated:
            env.reset()


def main():
    t_start = time.time()
    env = RCCarEnv(image_size=128, max_episode_seconds=999)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264", quality=8, macro_block_size=1)

    # ---- Phase 1: settle ----
    env.reset()
    run_phase(env, writer, "TEST 1/7: SETTLE (no input)", 1.5, lambda i, n: zero_action())

    # ---- Phase 2: forward ----
    env.reset()
    def fwd_action(i, n):
        a = zero_action(); a[1]=a[2]=a[3]=a[4]=0.35; return a
    run_phase(env, writer, "TEST 2/7: FORWARD DRIVE", 3.0, fwd_action,
              lambda env, obs, i: [f"wheel speeds={obs['wheel_speeds'].round(1)}"])

    # ---- Phase 3: reverse (continues from forward, matching a real brake-then-reverse) ----
    def rev_action(i, n):
        a = zero_action(); a[1]=a[2]=a[3]=a[4]=-0.35; return a
    run_phase(env, writer, "TEST 3/7: BRAKE + REVERSE", 3.0, rev_action)

    # ---- Phase 4: controlled turning ----
    env.reset()
    def turn_action(i, n):
        a = zero_action(); a[0]=0.3; a[1]=a[2]=a[3]=a[4]=0.32; return a
    run_phase(env, writer, "TEST 4/7: CONTROLLED TURN", 4.0, turn_action,
              lambda env, obs, i: [f"steer={obs['steering_angle'][0]:.2f} rad"])

    # ---- Phase 5: ramp jump w/ gyroscopic wheel effect + flywheel attitude control ----
    env.reset()
    liftoff_frame = {"seen": False}
    def jump_action(i, n):
        a = zero_action(); a[1]=a[2]=a[3]=a[4]=0.75
        dbg = env.get_debug_state()
        # spin up the flywheel once airborne, to visibly demonstrate reaction-wheel pitch control
        if env._raw()[11:15].sum() == 0 and dbg["pos"][0] > 2.0:
            a[5] = 1.0
        return a
    run_phase(env, writer, "TEST 5/7: RAMP JUMP (gyroscopic + flywheel attitude control)", 4.5, jump_action,
              lambda env, obs, i: [f"flywheel={obs['flywheel_speed'][0]:.0f} rad/s",
                                    f"contact={obs['wheel_contact'].astype(int)}"])

    # ---- Phase 6: camera pan sweep ----
    env.reset()
    def pan_action(i, n):
        a = zero_action(); a[1]=a[2]=a[3]=a[4]=0.1
        a[6] = np.sin(2*np.pi*i/n * 2)
        return a
    run_phase(env, writer, "TEST 6/7: CAMERA PAN SWEEP", 3.0, pan_action,
              lambda env, obs, i: [f"pan={obs['camera_pan_angle'][0]:.2f} rad"])

    # ---- Phase 7: random policy ----
    env.reset()
    agent = RandomAgent(env.action_space, seed=0)
    def rand_action(i, n):
        return agent.act()
    run_phase(env, writer, "TEST 7/7: RANDOM POLICY (untrained)", 8.0, rand_action)

    writer.close()
    print(f"Wrote {OUT_PATH}")
    print(f"Total wall-clock time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
