"""Analytic model of the planar arm, batched in numpy, for the model-based baselines.

    M(q) ddq + c(q, dq) + g(q) + b dq = tau,    tau = TORQUE_LIMITS * u

`step` integrates with RK4 on the simulator's substep, so one call predicts one env.step
(tests/test_env.py checks this against MuJoCo). `fast_step` takes one semi-implicit Euler
step of the same length: less accurate and about eight times cheaper, for planning.
"""

import numpy as np

from envs.planar_arm import (ACTION_PENALTY, FRAME_SKIP, GRAVITY, JOINT_DAMPING, LINK_INERTIA, LINK_LENGTH, LINK_MASS,
                             OBSTACLE_CENTRE, OBSTACLE_RADIUS, SUBSTEP, TARGET, TORQUE_LIMITS)


def acceleration(q, dq, tau):
    """ddq for arrays q, dq, tau of shape (..., 2)."""
    m, l, lc, inertia = LINK_MASS, LINK_LENGTH, LINK_LENGTH / 2, LINK_INERTIA
    c2, s2 = np.cos(q[..., 1]), np.sin(q[..., 1])
    m11 = 2 * inertia + m * lc**2 + m * (l**2 + lc**2 + 2 * l * lc * c2)
    m12 = inertia + m * (lc**2 + l * lc * c2)
    m22 = inertia + m * lc**2
    h = m * l * lc * s2
    g2 = m * GRAVITY * lc * np.sin(q[..., 0] + q[..., 1])
    g1 = m * GRAVITY * (lc + l) * np.sin(q[..., 0]) + g2
    rhs1 = tau[..., 0] - JOINT_DAMPING * dq[..., 0] + h * dq[..., 1] * (2 * dq[..., 0] + dq[..., 1]) - g1
    rhs2 = tau[..., 1] - JOINT_DAMPING * dq[..., 1] - h * dq[..., 0] ** 2 - g2
    det = m11 * m22 - m12**2
    return np.stack([(m22 * rhs1 - m12 * rhs2) / det, (m11 * rhs2 - m12 * rhs1) / det], axis=-1)


def step(state, u):
    """One env.step: state (..., 4), u (..., 2) in [-1, 1] -> next state (..., 4)."""
    tau = TORQUE_LIMITS * u

    def f(x):
        return np.concatenate([x[..., 2:], acceleration(x[..., :2], x[..., 2:], tau)], axis=-1)

    x = state
    for _ in range(FRAME_SKIP):
        k1 = f(x)
        k2 = f(x + 0.5 * SUBSTEP * k1)
        k3 = f(x + 0.5 * SUBSTEP * k2)
        k4 = f(x + SUBSTEP * k3)
        x = x + SUBSTEP / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def fast_step(state, u):
    """One env.step by semi-implicit Euler: same signature as `step`."""
    dt = SUBSTEP * FRAME_SKIP
    dq = state[..., 2:] + dt * acceleration(state[..., :2], state[..., 2:], TORQUE_LIMITS * u)
    return np.concatenate([state[..., :2] + dt * dq, dq], axis=-1)


def fingertip(state):
    """Fingertip (x, z) for state (..., 4) -> (..., 2)."""
    q1, q2 = state[..., 0], state[..., 1]
    x = LINK_LENGTH * (np.sin(q1) + np.sin(q1 + q2))
    z = -LINK_LENGTH * (np.cos(q1) + np.cos(q1 + q2))
    return np.stack([x, z], axis=-1)


def task_reward(next_x, u, obstacle_penalty=0.0):
    """The environment's reward for reaching state next_x with input u, batched. The problem gives this."""
    p = fingertip(next_x)
    reward = -np.linalg.norm(p - TARGET, axis=-1) - ACTION_PENALTY * np.sum(u**2, axis=-1)
    if obstacle_penalty > 0:
        reward = reward - obstacle_penalty * (np.linalg.norm(p - OBSTACLE_CENTRE, axis=-1) < OBSTACLE_RADIUS)
    return reward
