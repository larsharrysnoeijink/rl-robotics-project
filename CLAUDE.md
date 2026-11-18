# Project notes

Course project for 240MAR45 Reinforcement Learning in Robotics (UPC). README.md describes the task and the setup.

- Python is the conda env `rl` (Python 3.12) at `%USERPROFILE%\miniforge3\envs\rl\python.exe`. Run project code with that interpreter. `python tests/test_install.py` checks the installed stack against `requirements.txt`.
- Code arrives weekly from Atenea as a zip unpacked over this folder; releases only add files. Keep the provided structure; do not reorganise or rename the released files. Each homework has a brief in `briefs/`; the parts to write are marked `YOUR CODE HERE`.
- Training runs go to `runs/` and the course scripts write plots to `outputs/`. Both are committed, so teammates get the trained models. `output/week10/` holds our own study figures.
- Course knowledge: `context/rl-robotics/llms-full.txt` holds the full text of the concept knowledge base. It is a committed copy of https://github.com/AndreuCecilia/rl-robotics. To refresh it, delete `context/rl-robotics`, run `git clone --depth 1 https://github.com/AndreuCecilia/rl-robotics context/rl-robotics`, then delete `context/rl-robotics/.git`. The slides and lecture notes are in `context/`, the project slides in `context/project/`, and the lab notebooks in `context/practicals/`.
- The course uses control-engineering notation: state x, action u, dynamics f, reward r, discount γ.
- The course's rule for AI assistants: use them to explain, quiz and check derivations, not to produce graded exercise or milestone answers. The exams are taken without an assistant.
