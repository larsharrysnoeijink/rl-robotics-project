"""The harness: registry, results format, evaluation, training loop and command-line tools."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import numpy as np

import envs  # noqa: F401
from harness.evaluation import EVAL_SEEDS, evaluate
from harness.plots import group_runs, plot_comparison
from harness.registry import available_algorithms, build_agent, is_trainable
from harness.results import RunLog, find_runs, load_run
from harness.training import train

ROOT = Path(__file__).resolve().parent.parent


class Temporary(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)


class TestRegistry(unittest.TestCase):
    def test_finds_reinforce_and_baselines(self):
        algorithms = available_algorithms()
        self.assertEqual(algorithms["REINFORCE"], "models.reinforce")
        self.assertEqual(algorithms["LQR"], "baselines.lqr")
        self.assertFalse(is_trainable("LQR"))
        self.assertTrue(is_trainable("REINFORCE"))

    def test_unknown_hyperparameter_is_an_error(self):
        env = gym.make("PlanarArm-v0")
        with self.assertRaises(SystemExit):
            build_agent("REINFORCE", env, {"learning_rate_typo": 1.0})
        env.close()

    def test_override_is_applied(self):
        env = gym.make("PlanarArm-v0")
        _, params = build_agent("REINFORCE", env, {"lr": 0.5})
        env.close()
        self.assertEqual(params["lr"], 0.5)


class TestResults(Temporary):
    def test_round_trip(self):
        log = RunLog(self.folder / "a", {"algo": "X", "env": "E", "seed": 3, "variation": "base"})
        log.log_evaluation(100, 1.5, {"return": -5.0, "return_std": 1.0, "cost": 2.0, "distance": 0.1})
        log.log_episode(50, -3.0, 1.0)
        log.save()
        run = load_run(self.folder / "a")
        self.assertEqual(run["config"]["seed"], 3)
        np.testing.assert_array_equal(run["eval_steps"], [100])
        np.testing.assert_array_equal(run["train_cost"], [1.0])

    def test_find_runs_goes_into_group_folders(self):
        for group in ("group1", "group2"):
            log = RunLog(self.folder / group / "run", {"algo": "X", "env": "E", "seed": 1, "variation": "base"})
            log.save()
        self.assertEqual(len(find_runs(self.folder)), 2)

    def test_runs_group_by_algorithm_and_variation(self):
        runs = [{"config": {"algo": "A", "variation": "base", "seed": s}} for s in (0, 1)]
        runs.append({"config": {"algo": "A", "variation": "lr3", "seed": 0}})
        self.assertEqual({k: len(v) for k, v in group_runs(runs).items()}, {"A": 2, "A (lr3)": 1})


class TestEvaluation(unittest.TestCase):
    def test_same_policy_same_result(self):
        env = gym.make("PlanarArm-v0")
        agent, _ = build_agent("LQR", env)
        seeds = EVAL_SEEDS[:2]
        self.assertEqual(evaluate(agent, env, seeds), evaluate(agent, env, seeds))
        env.close()


class TestTraining(Temporary):
    def test_budget_is_in_steps_and_results_are_written(self):
        env = gym.make("PlanarArm-v0")
        agent, _ = build_agent("RANDOM", env)
        env.close()
        log = RunLog(self.folder / "run", {"algo": "RANDOM", "env": "PlanarArm-v0", "seed": 0, "variation": "base"})
        train("PlanarArm-v0", agent, total_steps=450, eval_every=200, log=log, seed=0)
        run = load_run(self.folder / "run")
        np.testing.assert_array_equal(run["eval_steps"], [0, 200, 400, 450])
        np.testing.assert_array_equal(run["train_steps"], [200, 400])
        self.assertTrue(np.all(np.diff(run["eval_wall_clock"]) >= 0))
        self.assertTrue((self.folder / "run" / "final.pt").exists())

    def test_dataset_holds_every_transition(self):
        env = gym.make("PlanarArm-v0")
        agent, _ = build_agent("RANDOM", env)
        env.close()
        log = RunLog(self.folder / "run", {"algo": "RANDOM", "env": "PlanarArm-v0", "seed": 0, "variation": "base"})
        train("PlanarArm-v0", agent, total_steps=250, eval_every=250, log=log, seed=0, save_data=True)
        with np.load(self.folder / "run" / "dataset.npz") as data:
            self.assertEqual(data["obs"].shape, (250, 4))
            self.assertEqual(data["action"].shape, (250, 2))
            self.assertEqual(int(data["truncated"].sum()), 1)
            self.assertTrue(data["truncated"][199])
            np.testing.assert_array_equal(data["next_obs"][:199], data["obs"][1:200])
            self.assertLessEqual(np.abs(data["action"]).max(), 1.0)

    def test_comparison_plot_with_seeds_and_a_baseline(self):
        runs = []
        for seed in (0, 1):
            runs.append({"config": {"algo": "A", "variation": "base", "seed": seed},
                         "eval_steps": np.array([0.0, 100.0]), "eval_wall_clock": np.array([0.0, 1.0]),
                         "eval_return": np.array([-3.0, -1.0 - seed])})
        runs.append({"config": {"algo": "B", "variation": "base", "seed": 0},
                     "eval_steps": np.array([0.0]), "eval_wall_clock": np.array([0.1]),
                     "eval_return": np.array([-2.0])})
        path = self.folder / "plot.png"
        plot_comparison(runs, save_path=path)
        self.assertTrue(path.exists())


class TestCommandLine(Temporary):
    """Run the tools the way students do, from a copy of the project so runs/ stays clean."""

    def setUp(self):
        super().setUp()
        for name in ("train.py", "compare.py", "watch.py"):
            shutil.copy(ROOT / name, self.folder)
        for name in ("envs", "harness", "models", "baselines", "utils"):
            shutil.copytree(ROOT / name, self.folder / name, ignore=shutil.ignore_patterns("__pycache__"))

    def run_tool(self, *arguments):
        result = subprocess.run([sys.executable, *arguments], cwd=self.folder, capture_output=True, text=True,
                                env={**__import__("os").environ, "MPLBACKEND": "Agg"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_train_compare_watch(self):
        self.run_tool("train.py", "--algo", "REINFORCE", "--seed", "2", "--steps", "400", "--eval-every", "200",
                      "--variation", "lr3", "--set", "lr=3e-4", "hidden_sizes=(8,8)")
        run_dir = self.folder / "runs" / "PlanarArm-v0_REINFORCE_lr3_seed2"
        config = load_run(run_dir)["config"]
        self.assertEqual(config["seed"], 2)
        self.assertEqual(config["hyperparameters"]["lr"], "0.0003")
        self.run_tool("train.py", "--algo", "LQR")
        output = self.run_tool("compare.py")
        self.assertIn("REINFORCE (lr3)", output)
        self.assertIn("LQR", output)
        self.assertTrue((self.folder / "outputs" / "comparison_PlanarArm-v0_return_steps.png").exists())
        self.run_tool("watch.py", "--algo", "REINFORCE", "--run", str(run_dir), "--path")
        self.assertTrue((self.folder / "outputs" / "path_REINFORCE.png").exists())


if __name__ == "__main__":
    unittest.main()
