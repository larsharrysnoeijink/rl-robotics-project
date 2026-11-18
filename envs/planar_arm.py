"""The course project task: a torque-limited planar 2-DOF arm reaching past an obstacle.

The arm moves in a vertical plane under gravity and starts hanging down. Its torque limits
are small enough that it cannot lift itself straight to the target, which is why a controller
linearised at the target fails from the start state. It is fully actuated: one motor per joint.

State (and observation):  x = (q1, q2, dq1, dq2)
    q1 is the shoulder angle from hanging straight down, q2 the elbow angle relative to link 1.
Action:  u in [-1, 1]^2, the fraction of the torque limit applied at each joint.
Reward:  r = -||p - p_target|| - ACTION_PENALTY * ||u||^2
Cost:    info["cost"] = 1 when the fingertip is inside the obstacle, else 0.
         The obstacle does not collide and the reward ignores it: that is the point of the cost.
Episode: 200 steps at 50 Hz, no early termination.

Start:   start="hanging" (the default, and always used for evaluation), or start="random", any pose
         at rest, used for training episodes. To reach the target the arm must first move away from
         it, so from hanging alone a learner rarely sees the states near the target. Random starts are
         a privilege of the simulator: on a real robot, someone would have to put the arm there.
"""

from pathlib import Path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

# Physical parameters. assets/planar_arm.xml holds the same numbers.
LINK_LENGTH = 0.5
LINK_MASS = 1.0
LINK_INERTIA = LINK_MASS * LINK_LENGTH**2 / 12.0
JOINT_DAMPING = 0.1
GRAVITY = 9.81
TORQUE_LIMITS = np.array([6.0, 5.0])
JOINT_LOW = np.array([-4.0, -2.8])      # the elbow bends one way, like a real one
JOINT_HIGH = np.array([4.0, 0.2])
SUBSTEP = 0.01
FRAME_SKIP = 2

# Task parameters.
TARGET = np.array([0.35, 0.6])      # (x, z) of the target
OBSTACLE_CENTRE = np.array([0.22, 0.2])
OBSTACLE_RADIUS = 0.25
ACTION_PENALTY = 0.05
START_STATE = np.zeros(4)           # hanging at rest
START_NOISE = 0.05                  # half-width of the uniform noise on the start state
RANDOM_START_LOW = np.array([-np.pi, -2.5])     # range of the joint angles when start="random"
RANDOM_START_HIGH = np.array([np.pi, 0.0])
REACHED_RADIUS = 0.05               # an episode counts as a success if it ends this close


def fingertip_position(q):
    """Fingertip (x, z) for joint angles q = (q1, q2)."""
    q1, q2 = q[0], q[1]
    x = LINK_LENGTH * (np.sin(q1) + np.sin(q1 + q2))
    z = -LINK_LENGTH * (np.cos(q1) + np.cos(q1 + q2))
    return np.array([x, z])


def target_configuration():
    """Joint angles that put the fingertip on the target, with the elbow bent the way it can bend."""
    x, z = TARGET
    cos_q2 = (x**2 + z**2 - 2 * LINK_LENGTH**2) / (2 * LINK_LENGTH**2)
    q2 = -np.arccos(np.clip(cos_q2, -1.0, 1.0))
    # With angles from hanging down: q1 = atan2(x, -z) - atan2(sin(q2), 1 + cos(q2)).
    q1 = np.arctan2(x, -z) - np.arctan2(np.sin(q2), 1.0 + np.cos(q2))
    return np.array([q1, q2])


def gravity_torque(q):
    """Joint torques (in N m) that hold the arm still at configuration q."""
    lc = LINK_LENGTH / 2
    g2 = LINK_MASS * GRAVITY * lc * np.sin(q[0] + q[1])
    g1 = LINK_MASS * GRAVITY * (lc + LINK_LENGTH) * np.sin(q[0]) + g2
    return np.array([g1, g2])


class PlanarArmEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(self, render_mode=None, start="hanging", **kwargs):
        utils.EzPickle.__init__(self, render_mode=render_mode, start=start, **kwargs)
        if start not in ("hanging", "random"):
            raise ValueError(f"start must be 'hanging' or 'random', got '{start}'")
        self.start = start
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            str(Path(__file__).parent / "assets" / "planar_arm.xml"),
            FRAME_SKIP,
            observation_space=observation_space,
            default_camera_config={"azimuth": 90.0, "elevation": 0.0, "distance": 3.2,
                                   "lookat": np.array([0.0, 0.0, 0.0])},
            render_mode=render_mode,
            **kwargs,
        )

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.do_simulation(action, self.frame_skip)
        # From the joint angles: after a step, MuJoCo's site positions still lag by one substep.
        p = fingertip_position(self.data.qpos)
        distance = float(np.linalg.norm(p - TARGET))
        reward = -distance - ACTION_PENALTY * float(np.sum(action**2))
        cost = float(np.linalg.norm(p - OBSTACLE_CENTRE) < OBSTACLE_RADIUS)
        info = {"cost": cost, "distance": distance, "fingertip": p}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, False, False, info

    def reset_model(self):
        noise = self.np_random.uniform(-START_NOISE, START_NOISE, size=4)
        state = START_STATE + noise
        if self.start == "random":
            while True:     # any pose at rest, except one already inside the obstacle
                state[:2] = self.np_random.uniform(RANDOM_START_LOW, RANDOM_START_HIGH)
                if np.linalg.norm(fingertip_position(state[:2]) - OBSTACLE_CENTRE) >= OBSTACLE_RADIUS:
                    break
        self.set_state(state[:2], state[2:])
        return self._get_obs()
