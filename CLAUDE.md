# Project notes

Course project for 240MAR45 Reinforcement Learning in Robotics (UPC). README.md describes the task and the setup.

- Python is the conda env `rl` (Python 3.12) at `%USERPROFILE%\miniforge3\envs\rl\python.exe`. Run project code with that interpreter. `python tests/test_install.py` checks the installed stack against `requirements.txt`.
- Code arrives weekly from Atenea, with a document that says where it goes. Keep the provided structure; do not reorganise or rename the released files.
- Course knowledge: `context/rl-robotics/llms-full.txt` holds the full text of the concept knowledge base (update it with `git -C context/rl-robotics pull`). The slides and lecture notes are in `context/`, and the lab notebooks are in `context/practicals/`.
- The course uses control-engineering notation: state x, action u, dynamics f, reward r, discount γ.
- The course's rule for AI assistants: use them to explain, quiz and check derivations, not to produce graded exercise or milestone answers. The exams are taken without an assistant.
