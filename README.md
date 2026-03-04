# Sudoku RL Solver

![CI](https://github.com/HarshMulodhia/sudoku-solver/actions/workflows/ci.yml/badge.svg)

A modular Sudoku-solving toolkit that pairs a **deterministic backtracking
solver** (constraint propagation + MRV heuristic) with **reinforcement-learning
agents** (**PPO** and **Double DQN**), all wrapped in an interactive
**pygame UI** featuring a cyberpunk dark/light theme, particle effects,
and real-time solving visualisation.

## Prerequisites

```bash
python >= 3.9
```

## Installation

### Option A: Conda Environment (Recommended for GPU Training)

```bash
# Create conda environment with GPU-enabled PyTorch
conda env create -f environment.yml
conda activate sudoku-rl-solver
```

### Option B: pip

```bash
# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
sudoku-rl-solver/
├── .github/workflows/ci.yml  # CI/CD pipeline (lint + test)
├── environment.yml            # Conda env (GPU-enabled PyTorch)
├── requirements.txt           # pip dependencies
├── src/                       # Source / library modules
│   ├── __init__.py
│   ├── config.py              # Configuration & hyperparameters
│   ├── sudoku_game.py         # Game logic & constraint handling
│   ├── rl_agent.py            # PPO + Double DQN agent implementations
│   ├── backtracking_solver.py # Deterministic backtracking solver
│   └── pygame_ui.py           # Interactive pygame interface
├── scripts/                   # Executable scripts
│   ├── train.py               # Training script
│   └── solver.py              # Inference & visualization
├── notebooks/                 # Analysis notebooks
│   └── solver_comparison.ipynb  # Backtracking vs RL comparison
├── docs/                      # Guides & documentation
│   ├── ADVANCED_GUIDE.md      # Advanced tips & best practices
│   └── TECHNICAL_GUIDE.md     # Technical deep dive & implementation
├── tests/                     # Test suite
│   ├── test_config.py
│   ├── test_sudoku_game.py
│   ├── test_pygame_ui.py
│   ├── test_rl_agent.py
│   └── test_backtracking_solver.py
└── models/                    # Saved trained models
    └── sudoku_{ppo,dqn}_*.pth
```

## Quick Start

### 1. Train the Agent (Optional)
```bash
# Train with PPO (recommended, default)
python scripts/train.py --episodes 1000 --difficulty medium --device cpu

# Train with PPO on GPU
python scripts/train.py --episodes 1000 --difficulty medium --device cuda

# Train with DQN (legacy)
python scripts/train.py --episodes 1000 --difficulty medium --device cpu --algorithm dqn
```

### 2. Run Solver with UI
```bash
python scripts/solver.py --mode play  # Manual play + solver assist
python scripts/solver.py --mode solve # Auto-solve visualization
```

### 3. Run Tests
```bash
python -m pytest tests/ -v
```

## Features

### RL Component
- **Algorithms**: PPO (Proximal Policy Optimization) and Double DQN with Experience Replay
- **PPO Architecture**: Actor-Critic with shared CNN backbone — 3 Conv2d layers + separate actor/critic FC heads
- **DQN Architecture**: 3 Conv2d layers (10→16→32→64) + 3 FC layers (256→128→64) ≈ 1.4 M parameters
- **State Representation**: 9×9×10 one-hot tensor (empty-cell indicator + digit channels)
- **Action Space**: 81 × 9 = 729 (cell × digit selections)
- **PPO Features**: Clipped surrogate objective, GAE advantage estimation, entropy bonus, invalid-action masking
- **Reward System**:
  - +1 for a valid move
  - +10 for placing the correct digit (matches solution)
  - −10 for a wrong digit or invalid move
  - +200 for puzzle completion
- **Stability**: Reward clipping (±250), target Q-value clipping, per-episode epsilon decay (DQN), gradient clipping (PPO)

### Deterministic Backtracking Solver
- **Algorithm**: Constraint propagation (naked singles) + recursive backtracking
- **Heuristic**: Minimum Remaining Values (MRV) – always branches on the cell with the fewest candidates
- **Guarantee**: Finds a valid solution whenever one exists (100 % success rate)
- **Speed**: Solves most 9×9 puzzles in < 5 ms

### Solver Comparison Notebook
A Jupyter notebook (`notebooks/solver_comparison.ipynb`) benchmarks both solvers
on easy / medium / hard puzzles and compares correctness, speed, and reliability.
When a trained model exists at `models/sudoku_dqn_{difficulty}.pth` the notebook
loads it automatically; otherwise it falls back to untrained weights.
Run it with:
```bash
cd notebooks && jupyter notebook solver_comparison.ipynb
```

### UI Features
- **Modern Design**: Dark/Light theme toggle with neon accents
- **3×3 Box Borders**: Thick borders and alternating backgrounds differentiate 3×3 blocks
- **Digit Highlighting**: Selecting a cell highlights all cells with the same digit
- **Animations**: 
  - Smooth cell highlighting
  - Particle effects on valid solutions
  - Transition animations between states
  - Real-time solving visualization
- **Interactive Controls**:
  - Mouse/keyboard input
  - Real-time hint system
  - Step-by-step execution
  - Difficulty selection (Easy/Medium/Hard buttons)
  - Undo support (button and Ctrl+Z)

## Configuration

Edit `src/config.py` to customize:
- RL algorithm choice (`--algorithm ppo` or `--algorithm dqn`)
- Neural network architecture (conv channels, hidden layer sizes)
- PPO hyperparameters (clip ε = 0.2, GAE λ = 0.95, entropy coef = 0.01, LR = 0.0003)
- DQN hyperparameters (LR = 0.001, γ = 0.99, ε decay = 0.995 per episode)
- Replay buffer size (50 000) and batch size (128)
- Target network sync frequency (every 100 steps, DQN only)
- Reward clipping (±250)
- UI theme and animation speed

## Performance Metrics

Trained agent solves:
- **Easy puzzles** (40 givens): ~95% success rate
- **Medium puzzles** (30 givens): ~85% success rate
- **Hard puzzles** (20 givens): ~70% success rate

**Note**: RL is interesting for learning puzzle patterns, but deterministic backtracking remains faster for guaranteed solutions.

## Documentation

- [Technical Deep Dive](docs/TECHNICAL_GUIDE.md) – architecture, algorithms, and implementation details
- [Advanced Tips & Best Practices](docs/ADVANCED_GUIDE.md) – curriculum learning, hybrid approaches, and research extensions

## References

- [DQN Paper](https://arxiv.org/abs/1312.5602) - Human-level control through deep RL
- [RL Sudoku Study](https://nlp.fi.muni.cz/aiproject/ui/karmazin_vasilii2024/SudokuRL.pdf)
- [Pygame Documentation](https://www.pygame.org/docs/)
