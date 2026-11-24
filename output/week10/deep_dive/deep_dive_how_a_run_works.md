# Deep dive: how a training run works

Optional reading. The homework itself is in `../homework1_summary.md`; this note goes one level deeper, with the maths and four detailed figures. Section 4 has six questions about the figures, each with a short answer.

What happens when you type

```
python train.py --algo REINFORCE-BASELINE --seed 80 --steps 300000 --eval-every 25000
```

Every figure below is drawn from our own runs in `runs/` (group 8, seeds 80 to 82). To redraw them after new runs, use `python output/week10/deep_dive/explain_runs.py`. Open this file with **Ctrl+Shift+V** to see the formulas and figures.

## 1. The loop, step by step

**The problem** is a Markov decision process. The state is $x$, the action is $u$, the dynamics are $x_{k+1} = f(x_k, u_k)$, and each step gives a reward $r$. The agent never sees $f$: it only plays and observes. That is what *model-free* means.

**The policy** is a Gaussian around a network's output:

$$\pi_\theta(u \mid x) = \mathcal N\big(\mu_\theta(x),\ \operatorname{diag}(\sigma^2)\big), \qquad \sigma = e^{s}$$

where $s$ is the parameter `log_std` in the code.

Both the network weights and $\log\sigma$ are in $\theta$, so the amount of exploration is learned too.

**One episode** (`harness/training.py`, `agent.update`):

1. At every step: sample $u_t \sim \pi_\theta(\cdot \mid x_t)$. The environment applies $\operatorname{clip}(u_t, -1, 1)$ and returns $r_{t+1}$ and $x_{t+1}$. The agent stores $(x_t, u_t, r_{t+1})$.
2. At the end of the episode (`_learn_from_episode`):

| Step | Mathematics | Code |
|---|---|---|
| Return-to-go | $G_t = r_{t+1} + \gamma G_{t+1}$, computed backwards from $G_T = 0$ | `reward_to_go` |
| Advantage estimate | $\delta_t = G_t - b_w(x_t)$, using $w$ as it was before this episode | `baseline_delta` |
| Baseline step | $w \leftarrow w - \alpha_w \nabla_w \frac1T \sum_t \big(b_w(x_t) - G_t\big)^2$ | `baseline_loss` + Adam |
| Policy step | $\theta \leftarrow \theta + \alpha_\theta \frac1T \sum_t \gamma^t\, \delta_t\, \nabla_\theta \log \pi_\theta(u_t \mid x_t)$ | `policy_loss` + Adam |

The code *minimises* $-\frac1T\sum_t \gamma^t \delta_t \log\pi_\theta$ because optimisers minimise. A descent step on that loss is an ascent step on the return.

**What "push $u_t$ up" means for a Gaussian.** For one action dimension, $\log\pi = -\frac{(u-\mu)^2}{2\sigma^2} - \log\sigma + \text{const}$, so

$$\frac{\partial \log\pi}{\partial \mu} = \frac{u-\mu}{\sigma^2}, \qquad \frac{\partial \log\pi}{\partial \log\sigma} = \frac{(u-\mu)^2}{\sigma^2} - 1 .$$

With $\delta_t > 0$, the mean moves *towards* the action that was tried. The spread grows if that action was far from the mean ($|u-\mu| > \sigma$) and shrinks if it was close.

**Evaluation** runs every `--eval-every` steps and is a separate experiment. It plays 20 episodes on fixed seeds (1000 to 1019), uses the mean action $\mu_\theta(x)$ with **no noise**, and on the arm always starts **from hanging**. It reports the **undiscounted** sum of rewards. That is the line printed during training:

```
step   300000 | return -140.1 | cost 173.4 | distance 0.529 | training time 212.1 s
```

Here *return* is the mean over the 20 episodes, *cost* is the mean number of steps the fingertip spent in the obstacle, and *distance* is how far the fingertip is from the target at the last step.

**What a run saves** (`runs/<env>_<algo>_<variation>_seed<S>/`): `results.npz` with every evaluation and the return of every training episode, `final.pt` with the policy and the baseline, and `step_*.pt` checkpoints when you pass `--checkpoint-every`.

**How many updates.** The arm: 300 000 steps ÷ 200 steps per episode = **1 500 episodes**, so 1 500 policy steps and 1 500 baseline steps. Scalar LQR: 150 000 ÷ 50 = **3 000** of each.

## 2. Scalar LQR, where the answer is known (Figure 1)

![Figure 1](fig1_lqr.png)

**The system:** $x_{k+1} = a x_k + b u_k$ with $a = 1.1$ and $b = 1$. The reward is $r = -(q x^2 + r_u u^2)$ with $q = 1$ and $r_u = 0.1$. The discount is $\gamma = 0.98$, the start is $x_0 \sim \mathcal N(0,1)$, and an episode lasts 50 steps. With $|a| > 1$ the plant is unstable without control.

**The policy.** With `hidden_sizes=()` the network is a single linear layer, so

$$u = -Kx + \sigma\varepsilon, \qquad \varepsilon \sim \mathcal N(0,1),$$

and `initial_gain=0.6` starts it at $K = 0.6$. The closed loop is $x_{k+1} = (a - bK)\,x_k + b\sigma\varepsilon_k$, stable while $|a - bK| < 1$.

**1a. The objective.** Without noise, the expected discounted return of a gain $K$ is

$$J(K) = -(q + r_u K^2)\ \mathbb E[x_0^2] \sum_{k=0}^{49} \big(\gamma (a-bK)^2\big)^k .$$

This is the black curve. REINFORCE never sees this formula: it estimates the curve's slope from sampled episodes and takes a step uphill. Each dot is the gain of one run at a 10k-step checkpoint, so you can watch every run climb from $K = 0.6$ (the square) towards the top. The optimum comes from the discounted Riccati equation:

$$p = q + \gamma a^2 p - \frac{(\gamma a b p)^2}{r_u + \gamma b^2 p}, \qquad K^\star = \frac{\gamma a b\, p}{r_u + \gamma b^2 p} = 1.007 .$$

`experiments/lqr_gain.py` prints the final $K$ of each run and draws it against $K^\star$ over training (`output/week10/part2_lqr/lqr_gain.png`).

**1b. The policy itself** is a line through the origin with slope $-K$. Learning rotates the line from the grey start towards the dashed optimum. The shaded band is $\pm\sigma$: the actions actually tried around the mean.

**1c. The exploration noise** $\sigma$ falls in every run. Use the $\partial\log\pi/\partial\log\sigma$ formula from section 1 to work out why.

**1d. The value.** For a fixed gain without noise the value is quadratic:

$$V_K(x) = -p_K x^2, \qquad p_K = \frac{q + r_u K^2}{1 - \gamma (a-bK)^2},$$

and $V^\star(x) = -p^\star x^2$ is the optimum; at $K = 0.94$ the two curves almost coincide. The black dots are observed returns $(x_t, G_t)$, and the blue curve is what the baseline learned from such returns. Three things to notice:

- Near $x = 0$ it sits *below* $-p_K x^2$. Even from $x = 0$, the noise $\sigma\varepsilon$ keeps pushing the state away, and every step away costs reward. (Late steps pull the other way: with little episode left, their $G_t$ is closer to 0.)
- Beyond $|x| \approx 1.5$ it is a straight line. That is the ReLU network extrapolating where it has seen almost no states.
- It is a regression on observed returns, not a Bellman solution: nothing is bootstrapped.

## 3. The arm (Figures 2 to 4)

**State** $x = (q_1, q_2, \dot q_1, \dot q_2)$: $q_1$ is the shoulder angle from hanging straight down, $q_2$ the elbow angle relative to the first link. Each link is $L = 0.5$ m long, so the fingertip is at

$$p(q) = L \begin{pmatrix} \sin q_1 + \sin(q_1+q_2) \\ -\cos q_1 - \cos(q_1 + q_2) \end{pmatrix}.$$

**Action and torque:** $\tau = (6\,u_1,\ 5\,u_2)$ N m, with $u$ clipped to $[-1, 1]$.

**Reward** $r = -\lVert p - p_\text{target}\rVert - 0.05\lVert u\rVert^2$ with $p_\text{target} = (0.35, 0.6)$. **Cost** is 1 while $\lVert p - c\rVert < 0.25$, with $c = (0.22, 0.2)$. The reward ignores the cost.

**The physics the agent never sees:** MuJoCo integrates $M(q)\ddot q + C(q,\dot q)\dot q + g(q) + d\,\dot q = \tau$ in 0.01 s substeps, two per step, so 50 Hz. Holding the arm still needs $\tau = g(q)$, with

$$g_2 = m g \tfrac L2 \sin(q_1+q_2), \qquad g_1 = m g \tfrac{3L}{2} \sin q_1 + g_2 .$$

Outstretched horizontally, $g_1 = 9.81$ N m, more than the shoulder's 6 N m. So the arm cannot lift itself straight up. It has to swing and build up energy: the hatched bands in Figure 3.

**Starts:** training episodes start from a random pose at rest; evaluation always starts from hanging.

**Networks:** the policy is $4 \to 64 \to 64 \to 2$ (the means) plus 2 log-stds. The baseline is $4 \to 64 \to 64 \to 1$.

### Figure 2: one episode, from the inside

![Figure 2](fig2_arm_episode.png)

One episode with the trained seed-80 policy, sampled with noise as in training but from hanging. The dashed path in 2a is the same policy without noise, as in evaluation.

- **2a.** The fingertip path in the vertical plane, coloured by time. The grey sticks are the arm at a few instants.
- **2b, 2c.** The state: angles and velocities. The dashed lines in 2b are the target configuration.
- **2d.** The actions. The thick lines are the means $\mu_\theta(x_t)$; the thin lines are the sampled $u_t$ that the update uses. The environment applies only the part inside the dashed $\pm 1$ lines.
- **2e.** The reward at each step; red marks the steps inside the obstacle.
- **2f.** The quantity the policy update multiplies $\nabla\log\pi$ by: $\delta_t = G_t - b_w(x_t)$. Green steps make their action more likely, red steps less likely.

The baseline is a function of the **state only**, and time is not part of the state. Near the end of the episode, $G_t$ goes to 0 because few steps are left. The baseline gives a resting state the same value at any time, so part of $\delta_t$ is "how much episode is left", not "how good was $u_t$".

### Figure 3: the policy and the value as maps

![Figure 3](fig3_arm_policy_value.png)

The state has 4 numbers, so these maps are slices: every joint-angle pair with **zero velocity**.

- **3a, 3b.** The mean torque each joint would apply. Red is positive, blue negative, saturated beyond $\pm 1$.
- **3c.** The baseline, the expected return-to-go from each pose.
- **Overlays:** the black line is the evaluation path from hanging (○); the star is the target configuration $q^\star = (3.42, -1.61)$; red curves mark the obstacle edge; hatching marks the poses the motors cannot hold against gravity.

### Figure 4: what learning looks like

![Figure 4](fig4_arm_learning.png)

Two different measurements of the same run:

- **Thin lines:** the training episodes, from random starts with noise, averaged over 25 episodes.
- **Dots:** the evaluations, from hanging without noise, every 25k steps.

The dashed line is the LQR controller from `python train.py --algo LQR`. `python compare.py --env PlanarArm-v0` puts the evaluation curves of all experiments in one plot (`output/week10/part3_arm/comparison_PlanarArm-v0_return_steps.png`).

## 4. Six questions about the figures, with answers

These are not the oral questions. They explain the most surprising things in the figures.

### 1. Figure 2d: the policy asks for more than full torque. What happens then?

> In 2d the mean torques sit outside $\pm 1$ for most of the episode. What does the arm actually receive then, and what is left of the exploration that $\sigma$ was supposed to provide? What does $\partial\log\pi/\partial\mu = (u-\mu)/\sigma^2$ still do to $\mu$ when the action is clipped anyway?

- **What the arm receives:** exactly full torque. The environment cuts every command off at $\pm 1$.
- **What is left of the exploration:** very little. The policy aims at about 1.3 and adds noise of about 0.6. Most of the noisy commands are still above 1, so the arm gets the same full torque. Trying "a bit more" or "a bit less" changes nothing the arm can feel.
- **What the update still does:** it keeps nudging $\mu$ towards or away from each sampled command. But for all the commands above the limit the arm did the same thing, so those nudges carry no information. They are noise. Nothing tells the policy that less torque would be better.

This is the main reason the arm never learns to brake or to hold still at the target.

### 2. Figure 2b: where do the joints end up?

> In 2b, where do the two joints end up, and is that the target? Why could a gradient method stay there?

- **Where:** at $q_1 = 4.0$ and $q_2 = -2.8$. Both are the mechanical end stops of the joints.
- **Is that the target?** No. The target needs $q^\star = (3.42, -1.61)$. The fingertip ends 0.53 m from the star, inside the obstacle.
- **Why it stays:** a gradient method only tries small changes. Here every small change gives the same result: the motors still push the joints into their end stops. To get closer, the arm would first have to back off and unfold, and that makes the reward worse before it gets better. There is no small step uphill, so the policy stays.

### 3. Figure 2f: why is $\delta_t$ red in the middle and green at the end?

> In 2f, why is $\delta_t$ red in the middle and green at the end, even though the arm hardly moves after 1 s?

Because the baseline does not know what time it is.

- From 1 s on, the arm sits still in the same state. The baseline only sees the state, so it gives the same guess every step, about −30.
- The real remaining return is not the same every step. With a lot of episode left, a lot of negative reward is still to come: more negative than −30, so $\delta_t < 0$ (red). Near the end almost nothing is left: closer to 0 than −30, so $\delta_t > 0$ (green).

So here red and green say "how much time is left", not "was this action good". A baseline that also saw the time would not have this problem.

### 4. Figure 1c: why does $\sigma$ fall in every LQR run?

> In 1c, $\sigma$ falls in every run. Using $\partial\log\pi/\partial\log\sigma$, what kind of episodes make it fall?

The formula $\frac{(u-\mu)^2}{\sigma^2} - 1$ is positive for an action far from the mean (further than $\sigma$) and negative for one close to it. The update multiplies it by $\delta$. So $\sigma$ shrinks in two cases:

- an action **far** from the mean did **worse** than usual;
- an action **close** to the mean did **better** than usual.

On the LQR both are the normal case. The cost grows with the square of $x$ and $u$, so straying far from the mean almost always costs reward. Nearly every episode says "less noise would have been better", and $\sigma$ falls.

### 5. Figure 4: why do training and evaluation curves disagree?

> In Figure 4, training curves and evaluation curves disagree. Which part of the difference comes from the start state, and which from the noise?

They are two different tests.

- **Training episodes** start from a random pose and use the noisy actions.
- **Evaluation episodes** always start from hanging and use no noise.

**Most of the difference comes from the start state.** Hanging is the hardest start: the arm must first get up. A random start is sometimes already high or close to the star. So a run that never learned to get up scores much worse in evaluation than in training. Baseline seed 81 is the clearest case: about −241 in training, −349 in evaluation. A run that does get up scores almost the same in both: seed 80 has −138 and −140.

**The noise matters little here**, because the policies ask for more than full torque (question 1), and the cut-off swallows most of the noise.

### 6. Figure 1d: the baseline is wrong for large $x$. Does that hurt?

> In 1d the baseline is far from $-p_K x^2$ for $|x| > 1.5$. Does that hurt the policy gradient? How often are those states visited?

Hardly.

- **It never adds bias.** Any baseline that depends only on the state leaves the average gradient unchanged. A wrong baseline only removes less noise.
- **Those states are rare.** The start is drawn around 0, so $|x| > 1.5$ happens in about 13 % of the starts. The controller then pulls $x$ back within a step or two, and the rest of the 50 steps are near 0.

So the baseline is poor exactly where it is almost never used.

## 5. Where everything is

| File | What it is |
|---|---|
| `output/week10/part2_lqr/lqr_gain.png` | $K$ over training for every LQR run, against $K^\star$ (oral question 1) |
| `output/week10/part2_lqr/ and part3_arm/gradient_variance_*.npz` | The variance measurement; the printed table is in the terminal output (question 2) |
| `output/week10/part2_lqr/ and part3_arm/comparison_*_return_steps.png` | Evaluation curves of every experiment (questions 4 and 5) |
| `output/week10/part3_arm/path_REINFORCE-BASELINE.png` | Fingertip path of the seed-80 policy, from `watch.py --path` |
| `output/week10/deep_dive/fig*.png` | The figures in this file |
| `output/week10/part1_code/part1_baseline_loss.png` | What `baseline_loss` measures |
