"""Project environments. Importing this package registers them with Gymnasium."""

from gymnasium.envs.registration import register

register(id="PlanarArm-v0", entry_point="envs.planar_arm:PlanarArmEnv", max_episode_steps=200)
register(id="ScalarLQR-v0", entry_point="envs.scalar_lqr:ScalarLQREnv", max_episode_steps=50)

# Training episodes may start differently from the task they are evaluated on (see envs/planar_arm.py).
TRAINING_KWARGS = {"PlanarArm-v0": {"start": "random"}}
