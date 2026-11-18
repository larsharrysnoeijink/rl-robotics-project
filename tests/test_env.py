"""The project environments do what the syllabus and the analytic models say."""

import unittest

import gymnasium as gym
import numpy as np
from scipy.linalg import solve_discrete_are

import envs  # noqa: F401
from envs import planar_arm as arm
from envs import scalar_lqr
from envs.planar_arm_dynamics import fast_step, fingertip, step, task_reward


class TestPlanarArm(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("PlanarArm-v0")

    def tearDown(self):
        self.env.close()

    def test_specification(self):
        self.assertEqual(self.env.observation_space.shape, (4,))
        self.assertEqual(self.env.action_space.shape, (2,))
        self.assertAlmostEqual(self.env.unwrapped.dt, 0.02)      # 50 Hz
        self.assertEqual(self.env.spec.max_episode_steps, 200)   # 4 seconds

    def test_xml_matches_constants(self):
        model = self.env.unwrapped.model
        np.testing.assert_allclose(model.actuator_gear[:, 0], arm.TORQUE_LIMITS)
        np.testing.assert_allclose(model.body_mass[1:], arm.LINK_MASS)
        np.testing.assert_allclose(model.dof_damping, arm.JOINT_DAMPING)
        np.testing.assert_allclose(model.jnt_range[:, 0], arm.JOINT_LOW)
        np.testing.assert_allclose(model.jnt_range[:, 1], arm.JOINT_HIGH)
        np.testing.assert_allclose(model.body_inertia[1:, 1], arm.LINK_INERTIA, atol=1e-5)
        np.testing.assert_allclose(model.opt.timestep, arm.SUBSTEP)
        np.testing.assert_allclose(model.site("target").pos[[0, 2]], arm.TARGET)
        np.testing.assert_allclose(model.geom("obstacle").pos[[0, 2]], arm.OBSTACLE_CENTRE)
        np.testing.assert_allclose(model.geom("obstacle").size[0], arm.OBSTACLE_RADIUS)

    def test_episode_has_no_early_termination(self):
        self.env.reset(seed=0)
        for t in range(200):
            _, _, terminated, truncated, _ = self.env.step(self.env.action_space.sample())
            self.assertFalse(terminated)
            self.assertEqual(truncated, t == 199)

    def test_reset_is_reproducible(self):
        first, _ = self.env.reset(seed=7)
        second, _ = self.env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertLess(np.abs(first).max(), arm.START_NOISE + 1e-9)

    def test_random_starts_are_at_rest_inside_the_limits_and_outside_the_obstacle(self):
        env = gym.make("PlanarArm-v0", start="random")
        far_from_hanging = 0
        for seed in range(200):
            obs, _ = env.reset(seed=seed)
            self.assertTrue(np.all(obs[:2] >= arm.JOINT_LOW) and np.all(obs[:2] <= arm.JOINT_HIGH))
            self.assertLessEqual(np.abs(obs[2:]).max(), arm.START_NOISE)
            tip = arm.fingertip_position(obs[:2])
            self.assertGreaterEqual(np.linalg.norm(tip - arm.OBSTACLE_CENTRE), arm.OBSTACLE_RADIUS)
            far_from_hanging += np.abs(obs[:2]).max() > 0.5
        env.close()
        self.assertGreater(far_from_hanging, 100)

    def test_training_uses_random_starts_and_the_registered_task_hangs(self):
        self.assertEqual(envs.TRAINING_KWARGS["PlanarArm-v0"], {"start": "random"})
        self.assertEqual(self.env.unwrapped.start, "hanging")

    def test_analytic_step_matches_simulator(self):
        unwrapped = self.env.unwrapped
        unwrapped.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = rng.uniform([-2, -2, -5, -5], [2, -0.1, 5, 5])    # inside the joint stops, which the model lacks
            u = rng.uniform(-1, 1, 2)
            unwrapped.set_state(x[:2], x[2:])
            simulated, *_ = unwrapped.step(u)
            np.testing.assert_allclose(step(x, u), simulated, atol=1e-3)

    def test_task_reward_matches_the_environment(self):
        unwrapped = self.env.unwrapped
        unwrapped.reset(seed=0)
        rng = np.random.default_rng(1)
        for _ in range(20):
            x = rng.uniform([-2, -2, -5, -5], [2, -0.1, 5, 5])
            u = rng.uniform(-1, 1, 2)
            unwrapped.set_state(x[:2], x[2:])
            next_x, reward, *_ = unwrapped.step(u)
            self.assertAlmostEqual(task_reward(next_x, u), reward, places=6)

    def test_planning_model_is_close(self):
        x = np.array([1.0, -0.5, 2.0, -1.0])
        u = np.array([0.5, -0.5])
        error = np.abs(fast_step(x, u) - step(x, u))
        self.assertLess(error[:2].max(), 0.03)     # radians: first-order in dt by construction
        self.assertLess(error[2:].max(), 0.2)      # radians per second; the planner replans every step

    def test_target_configuration_can_be_held(self):
        q = arm.target_configuration()
        np.testing.assert_allclose(arm.fingertip_position(q), arm.TARGET, atol=1e-9)
        np.testing.assert_allclose(fingertip(np.concatenate([q, np.zeros(2)])), arm.TARGET, atol=1e-9)
        u = arm.gravity_torque(q) / arm.TORQUE_LIMITS
        self.assertLess(np.abs(u).max(), 1.0)
        self.assertTrue(np.all((q > arm.JOINT_LOW) & (q < arm.JOINT_HIGH)))
        unwrapped = self.env.unwrapped
        unwrapped.reset(seed=0)
        unwrapped.set_state(q, np.zeros(2))
        for _ in range(25):
            _, _, _, _, info = unwrapped.step(u)
        self.assertLess(info["distance"], 1e-2)

    def test_torque_limit_stops_a_direct_lift(self):
        """Holding the arm stretched out sideways takes more than the shoulder can give."""
        self.assertGreater(arm.gravity_torque(np.array([np.pi / 2, 0.0]))[0], arm.TORQUE_LIMITS[0])

    def test_cost_marks_the_obstacle_and_reward_ignores_it(self):
        unwrapped = self.env.unwrapped
        unwrapped.reset(seed=0)
        # Inverse kinematics onto the obstacle centre, by the same formula as the target.
        x, z = arm.OBSTACLE_CENTRE
        q2 = -np.arccos(np.clip((x**2 + z**2 - 2 * arm.LINK_LENGTH**2) / (2 * arm.LINK_LENGTH**2), -1, 1))
        q1 = np.arctan2(x, -z) - np.arctan2(np.sin(q2), 1 + np.cos(q2))
        np.testing.assert_allclose(arm.fingertip_position(np.array([q1, q2])), arm.OBSTACLE_CENTRE, atol=1e-9)
        unwrapped.set_state(np.array([q1, q2]), np.zeros(2))
        _, reward, _, _, info = unwrapped.step(np.zeros(2))
        self.assertEqual(info["cost"], 1.0)
        self.assertAlmostEqual(reward, -info["distance"])
        unwrapped.set_state(np.zeros(2), np.zeros(2))
        _, _, _, _, info = unwrapped.step(np.zeros(2))
        self.assertEqual(info["cost"], 0.0)


class TestScalarLQR(unittest.TestCase):
    def test_riccati_matches_scipy(self):
        p, K = scalar_lqr.riccati_solution()
        root = np.sqrt(scalar_lqr.GAMMA)
        P = solve_discrete_are(np.array([[root * scalar_lqr.A]]), np.array([[root * scalar_lqr.B]]),
                               np.array([[scalar_lqr.Q]]), np.array([[scalar_lqr.R]]))
        self.assertAlmostEqual(p, P.item(), places=8)
        self.assertLess(abs(scalar_lqr.A - scalar_lqr.B * K), 1.0)

    def test_optimal_return_matches_a_rollout(self):
        _, K = scalar_lqr.riccati_solution()
        env = gym.make("ScalarLQR-v0")
        total, n = 0.0, 4000
        for seed in range(n):
            obs, _ = env.reset(seed=seed)
            done, discount = False, 1.0
            while not done:
                obs, reward, terminated, truncated, _ = env.step(-K * obs)
                total += discount * reward
                discount *= scalar_lqr.GAMMA
                done = terminated or truncated
        env.close()
        self.assertAlmostEqual(total / n, scalar_lqr.optimal_return(), delta=0.15)


if __name__ == "__main__":
    unittest.main()
