# RL in Robotics: course project

Group project for 240MAR45 Reinforcement Learning in Robotics (MSc in Automatic Control and Robotics, UPC Barcelona).

The project code is released on Atenea week by week, from Week 9 on, and each week covers a different RL approach. Every week's homework comes with a document that says where the new code goes. Keep the provided structure, so that later releases fit in without conflicts.

## The task

A torque-limited planar arm with two joints has to reach a target position.

- The arm moves in a vertical plane under gravity and starts hanging down.
- Its torque limits are too small to lift it straight to the target, so it has to build up momentum first.
- It is fully actuated, with one motor per joint.
- State: x = (θ1, θ2, θ1', θ2').
- Action: u = (τ1, τ2) ∈ [−1, 1]².
- Reward, as written on slide 4: r = ‖p − p_target‖ − 0.05 · ‖u‖², where p is the end-effector position. The distance term is presumably meant as a cost (negative). Check the sign in the environment code when it is released.
- An episode lasts 200 steps.

## Setup (once per laptop)

The course uses Python 3.12 in a conda environment called `rl`. The full steps are in `Project Setup Slides.pdf`.

1. Install [Miniforge](https://conda-forge.org/download/) or Miniconda, and VS Code with the Python and Jupyter extensions.
2. Let PowerShell use conda. In PowerShell, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Then, in the Miniforge (or Anaconda) Prompt, run `conda init powershell`.
3. Open this folder in VS Code, open a new terminal, and run:

   ```
   conda create -n rl python=3.12
   conda activate rl
   python -m pip install -r requirements.txt
   python tests/test_install.py
   ```

   The last command should end with `setup OK`.
4. Check that the VS Code status bar shows `rl (3.12.x)`. If it doesn't, press Ctrl+Shift+P, choose **Python: Select Interpreter**, and pick the `rl` environment (for example `C:\Users\<you>\miniforge3\envs\rl\python.exe`).

## Layout

| Path | What it is |
|---|---|
| `requirements.txt` | The pinned course stack, the same versions as Colab runtime 2026.07 |
| `tests/test_install.py` | Smoke test for the stack: versions, MuJoCo, PyTorch |
| `context/slides/`, `context/lecture notes/` | Course slides and lecture notes |
| `context/practicals/` | Lab notebooks for Weeks 1 to 7 |
| `context/rl-robotics/` | The course's concept knowledge base. It is not in this repo. Get it with `git clone https://github.com/AndreuCecilia/rl-robotics context/rl-robotics`. `llms-full.txt` in it holds the full text of every page. |
| `Project Details Slides.pdf`, `Project Setup Slides.pdf` | Project brief and setup guide |

## Contact

Questions about the homework go to aniol.civit@upc.edu. Finish each week's homework before the class: some checkpoint quizzes need your trained models.
