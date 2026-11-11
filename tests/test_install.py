"""Smoke test: does this machine have the course stack? Takes a few seconds, opens no window.

    python tests/test_install.py
"""

import re
import sys
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

COURSE_LIBRARIES = ("numpy", "scipy", "matplotlib", "gymnasium", "torch", "mujoco")


def pinned_versions():
    """{library: version} read from requirements.txt, for the libraries the project imports."""
    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9_-]+)(?:\[[^\]]*\])?==([^\s#]+)", line)
        if match and match.group(1) in COURSE_LIBRARIES:
            pins[match.group(1)] = match.group(2)
    return pins


class TestInstall(unittest.TestCase):
    def test_python_version(self):
        self.assertEqual(sys.version_info[:2], (3, 12), "The course uses Python 3.12.")

    def test_library_versions(self):
        pins = pinned_versions()
        self.assertEqual(set(pins), set(COURSE_LIBRARIES))
        for library, pinned in pins.items():
            try:
                installed = version(library)
            except PackageNotFoundError:
                self.fail(f"{library} is not installed. Run: python -m pip install -r requirements.txt")
            # A local build tag such as 2.11.0+cpu is the same version.
            self.assertEqual(installed.split("+")[0], pinned, f"{library}: found {installed}, the course pins {pinned}")

    def test_simulator_runs(self):
        import gymnasium as gym

        env = gym.make("InvertedPendulum-v5")
        env.reset(seed=0)
        for _ in range(10):
            env.step(env.action_space.sample())
        env.close()

    def test_torch_runs(self):
        import torch

        self.assertEqual(torch.ones(2).sum().item(), 2.0)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=1).result
    print("\nsetup OK" if result.wasSuccessful() else "\nSETUP NOT OK: read the messages above and raise your hand.")
    sys.exit(0 if result.wasSuccessful() else 1)
