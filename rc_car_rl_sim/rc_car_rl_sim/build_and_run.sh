#!/usr/bin/env bash
# One-command build + test + demo. Requires: g++ (C++17), python3 with numpy/imageio/Pillow.
# No pip installs, no GPU, no display needed -- everything here is self-contained.
set -e
cd "$(dirname "$0")"

echo "== [1/4] Building libcarsim.so =="
cd cpp
g++ -O3 -shared -fPIC -std=c++17 -Wall -Wextra -o ../python/carsim/libcarsim.so capi.cpp
echo "   built python/carsim/libcarsim.so"

echo "== [2/4] Running C++ physics regression suite =="
cd tests
g++ -O2 -std=c++17 -I.. -o /tmp/test_physics test_physics.cpp
/tmp/test_physics
cd ../..

echo "== [3/4] Running Python environment smoke test =="
python3 - << 'EOF'
import sys
sys.path.insert(0, "python")
from carsim.env import RCCarEnv
env = RCCarEnv()
obs, info = env.reset()
for _ in range(60):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    if term:
        env.reset()
assert obs["camera"].shape == (128,128,3)
print("   environment smoke test OK")
EOF

echo "== [4/4] Generating test video (scripted maneuvers + random policy) =="
cd python/tests
python3 run_full_demo.py
cd ../..

echo
echo "Done. Video at output/rc_car_test_video.mp4"
