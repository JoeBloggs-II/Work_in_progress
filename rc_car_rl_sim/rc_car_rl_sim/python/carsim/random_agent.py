"""Random policy: samples a new random action every ~0.3s and holds it (zero-order hold) in
between, rather than resampling every control step. Pure i.i.d. noise every ~33ms looks like
violent twitching and isn't representative of any real exploration policy; a short hold is the
simplest reasonable stand-in. No learning happens here -- this is only "move around randomly",
per the brief. Training is a separate, not-yet-implemented task."""
import numpy as np


class RandomAgent:
    def __init__(self, action_space, hold_seconds=0.3, control_hz=30.0, seed=None):
        self.action_space = action_space
        self.hold_steps = max(1, int(hold_seconds*control_hz))
        self._counter = 0
        self._current = None
        self._rng = np.random.default_rng(seed)

    def act(self, obs=None):
        if self._current is None or self._counter <= 0:
            self._current = self._rng.uniform(self.action_space.low, self.action_space.high)
            self._counter = self.hold_steps
        self._counter -= 1
        return self._current
