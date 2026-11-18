# Reinforcement Learning in Robotics: cumulative project

One task, several methods, compared on the same axes.

## The task

A planar arm with two links moves in a vertical plane under gravity and starts hanging down. Its
fingertip must reach a target above the shoulder without passing through a circular obstacle. The elbow
bends one way only.

- **State:** joint angles and joint velocities (4 numbers).
- **Action:** the fraction of the torque limit applied at each joint, between -1 and 1 (2 numbers).
- **Reward:** minus the distance to the target, with a small penalty on the action.
- **Cost:** 1 for each step the fingertip spends inside the obstacle. The reward does not see it.
- **Episode:** 200 steps at 50 Hz, 4 seconds.

The shoulder cannot lift the outstretched arm, so the torque limit matters. Try `LQR` from the start
state and you will see what that does to a controller linearised at the target.

Evaluation always starts from hanging. Training episodes start from a random pose, because the arm has
to move away from the target before it can reach it, and a learner that only ever starts from hanging
rarely sees the states that matter. This is a privilege of the simulator. On a real robot, someone would
have to put the arm there.

## Install

Python 3.12 with Miniforge or Anaconda, from the project folder:

```
conda create -n rl python=3.12
conda activate rl
python -m pip install -r requirements.txt
python tests/test_install.py
```

The last line must end with `setup OK`. On Linux, read the note at the top of `requirements.txt` first.

## Use

```
python train.py --algo REINFORCE --seed 3 --steps 100000      # train, writes to runs/
python train.py --algo LQR                                    # baselines have nothing to train
python compare.py                                             # table and plot of everything in runs/
python compare.py --x wall --metric cost                      # other axes and metrics
python watch.py --algo REINFORCE --run runs/PlanarArm-v0_REINFORCE_base_seed3       # opens a window
python watch.py --algo LQR --path                             # saves the fingertip path as an image
```

To try a change of hyperparameters, name the experiment so its runs stay apart:

```
python train.py --algo REINFORCE --seed 3 --variation lr3e4 --set lr=3e-4
```

`python train.py --help` lists every option. Each run writes `results.npz` and `final.pt` to its folder
under `runs/`. Press Ctrl+C to stop a run early and keep what it has.

## Homework

Each homework comes with a brief in `briefs/`, which says what to run and which questions to prepare
for the oral check. New code arrives as a zip to unpack over this folder. Releases only add files, so
your work stays as it is.

Complete the sections marked `YOUR CODE HERE`, and check them with the test the brief names:

```
python -m unittest tests.test_c1_baseline
```

Run `python -m unittest discover -s tests -t .` if you suspect your installation. Tests named after a
homework you have not done yet fail with `NotImplementedError`, which is expected.
