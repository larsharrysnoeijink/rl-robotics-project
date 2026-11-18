# Homework 1: REINFORCE with and without a baseline

**What you will see.** The same algorithm on two problems, one whose optimum the Riccati equation gives you and one, the project arm, where no such answer exists. In both, you measure what a baseline does to the variance of the gradient estimate, and what that buys in learning.

**Theory you need:** the Week 9 deck, in particular the frame "REINFORCE with a baseline".

**Seeds.** Your group number is on the course page. Your seeds are `10·g`, `10·g+1` and `10·g+2`: group 3 uses 30, 31 and 32. Everyone uses different seeds, and the session compares them. Write `S` below for each of your seeds in turn.

**Time.** About 15 minutes of computing on a laptop, which can run while you do something else, plus your own work on parts 1, 5 and 6.

## 1. Complete the code

Open `models/reinforce_baseline.py` and complete the two sections marked `YOUR CODE HERE`. `models/reinforce.py` next to it is already complete, and worth reading first. Check your work:

```
python -m unittest tests.test_c1_baseline tests.test_c1_variance
```

Both tests must pass before you move on, everything below uses your code.

## 2. Scalar LQR, against the Riccati answer

The system is $x_{k+1} = 1.1\,x_k + u_k$, with stage cost $x^2 + 0.1\,u^2$ and $\gamma = 0.98$. The policy is linear, $u = -Kx$ plus Gaussian noise, and it starts from the stabilising gain $K = 0.6$.

```
python train.py --algo RICCATI --env ScalarLQR-v0
```

Then, for each of your three seeds:

```
python train.py --algo REINFORCE --env ScalarLQR-v0 --seed S --steps 150000 --eval-every 10000 --checkpoint-every 10000 --set "hidden_sizes=()" gamma=0.98 initial_gain=0.6
python train.py --algo REINFORCE-BASELINE --env ScalarLQR-v0 --seed S --steps 150000 --eval-every 10000 --checkpoint-every 10000 --set "hidden_sizes=()" gamma=0.98 initial_gain=0.6
```

Then read the results. Replace `S` in the last line with one of your seeds:

```
python compare.py --env ScalarLQR-v0
python experiments/lqr_gain.py "runs/ScalarLQR-v0_REINFORCE*"
python experiments/gradient_variance.py --run runs/ScalarLQR-v0_REINFORCE_base_seedS
```

## 3. The project arm

For each of your three seeds:

```
python train.py --algo REINFORCE --seed S --steps 300000 --eval-every 25000
python train.py --algo REINFORCE-BASELINE --seed S --steps 300000 --eval-every 25000
```

Then:

```
python train.py --algo LQR
python compare.py --env PlanarArm-v0
python experiments/gradient_variance.py --run runs/PlanarArm-v0_REINFORCE_base_seedS
python watch.py --algo REINFORCE-BASELINE --run runs/PlanarArm-v0_REINFORCE-BASELINE_base_seedS --path
```

## 4. Your assigned variation

Run this for each of your three seeds, with the settings of your group's variation. Group numbers above 7 start again from variation 1.

| Group | Name | Settings |
|-------|------|----------|
| 1 | `plr3e4` | `policy_lr=3e-4` |
| 2 | `plr3e3` | `policy_lr=3e-3` |
| 3 | `gam95` | `gamma=0.95` |
| 4 | `gam999` | `gamma=0.999` |
| 5 | `blr1e4` | `baseline_lr=1e-4` |
| 6 | `small` | `"hidden_sizes=(16,16)"` |
| 7 | `large` | `"hidden_sizes=(256,256)"` |

```
python train.py --algo REINFORCE-BASELINE --seed S --steps 300000 --eval-every 25000 --variation NAME --set SETTINGS
python compare.py --env PlanarArm-v0
```

## 5. Your own experiment

Choose one setting to change, and test what you expect it to do. Bring these four lines to the oral check:

1. **Change:** one setting, different from your assigned variation.
2. **Prediction:** written before you run it, with the reason from the theory.
3. **Result:** your plot, on at least 2 seeds.
4. **Explanation:** why the prediction held or failed.

A wrong prediction that you can explain scores as well as a right one. Pick an option from the list below or propose your own, as long as it rests on theory already shown in class. Do not use the exact same values as the suggestions below.

| Problem | Setting | Question |
|---------|---------|----------|
| LQR | `initial_gain=0.0` | Does the start have to stabilise the plant? |
| LQR | `initial_gain=1.3` | Does the gain also approach $K^\star$ from above? |
| Arm | `baseline_lr=1e-2` | Can the baseline learn too fast? |
| Arm | `"hidden_sizes=()"` | How far does a linear policy get on the arm? |

Give your experiment a name of your own, with no underscore. For the two LQR settings, keep the other settings of part 2:

```
python train.py --algo REINFORCE-BASELINE --seed S --steps 300000 --eval-every 25000 --variation NAME --set baseline_lr=1e-2
```

## 6. Prepare the pre-session questions

Be ready to answer these at the start of the session, with your own run on screen. Any member of the group may be asked.

1. Show the gain plot from part 2. What final $K$ did you reach with and without the baseline, and how far is that from $K^\star$?
2. What relative variance did you measure for the three estimators on LQR, and did the means move?
3. Open your `baseline_delta`. Why does it detach the baseline, and what would go wrong if it did not?
4. Show the comparison on the arm. Which did better on your seeds, and by how much?
5. What did your assigned variation change?
6. What was your own experiment: your prediction, your result, and why they agree or differ?

Your results need not look good. A run that failed is a valid result, as long as you can show it and say what you checked. The marking criteria are in the milestone rubric on the course page.

<!-- ## 7. Upload -->
<!--  -->
<!-- The day before the session, zip your `runs/` folder and upload it to the course page under your group's name. -->
