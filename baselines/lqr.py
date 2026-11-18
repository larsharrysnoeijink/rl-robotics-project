"""LQR on the planar arm, linearised at the target configuration.

The arm is linearised at (q*, 0) with the holding torque u* that cancels gravity there, and the
discrete Riccati equation gives the gain:

    u = u* - K (x - x*),    clipped to the torque limits.

This controller is good near the target and fails from the hanging start: the linear model does
not know about the torque limit or about how gravity changes along the way. No learning, so it
is evaluated once; `python train.py --algo LQR` writes a run that compares with the others.
"""

import numpy as np
from scipy.linalg import solve_discrete_are

from envs.planar_arm import TORQUE_LIMITS, gravity_torque, target_configuration
from envs.planar_arm_dynamics import step

ALGO = "LQR"
TRAINABLE = False
DEFAULTS = dict(
    q_position=20.0,    # state weight on joint angles
    q_velocity=1.0,     # state weight on joint velocities
    r=0.5,              # control weight
)


def linearise(x_eq, u_eq, eps=1e-6):
    """Discrete-time (A, B) of the one-step model around (x_eq, u_eq), by central differences."""
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    for i in range(4):
        dx = eps * np.eye(4)[i]
        A[:, i] = (step(x_eq + dx, u_eq) - step(x_eq - dx, u_eq)) / (2 * eps)
    for i in range(2):
        du = eps * np.eye(2)[i]
        B[:, i] = (step(x_eq, u_eq + du) - step(x_eq, u_eq - du)) / (2 * eps)
    return A, B


class LQRController:
    def __init__(self, obs_dim, act_dim, act_low, act_high, device, *, q_position, q_velocity, r):
        q_star = target_configuration()
        self.x_star = np.concatenate([q_star, np.zeros(2)])
        self.u_star = gravity_torque(q_star) / TORQUE_LIMITS
        self.act_low, self.act_high = act_low, act_high
        A, B = linearise(self.x_star, self.u_star)
        Q = np.diag([q_position, q_position, q_velocity, q_velocity])
        R = r * np.eye(2)
        P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)

    def select_action(self, obs, deterministic=True):
        u = self.u_star - self.K @ (obs - self.x_star)
        return np.clip(u, self.act_low, self.act_high).astype(np.float32)

    def update(self, obs, action, reward, next_obs, terminated, truncated):
        pass

    def save(self, path):
        np.savez(path, K=self.K, x_star=self.x_star, u_star=self.u_star)

    def load(self, path):
        pass


AGENT_CLASS = LQRController
