# Sudoku RL Solver

![CI](https://github.com/HarshMulodhia/sudoku-solver/actions/workflows/ci.yml/badge.svg)

A modular Sudoku-solving toolkit that pairs a **deterministic backtracking
solver** (constraint propagation + MRV heuristic) with a **Deep Q-Network
(DQN) reinforcement-learning agent**, all wrapped in an interactive
**pygame UI** featuring a cyberpunk dark/light theme, particle effects,
and real-time solving visualisation.

![CI](https://github.com/HarshMulodhia/sudoku-solver/actions/workflows/ci.yml/badge.svg)

![CI](https://github.com/HarshMulodhia/sudoku-solver/actions/workflows/ci.yml/badge.svg)

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
<<<<<<< HEAD
│   ├── rl_agent.py            # DQN agent implementation
│   ├── backtracking_solver.py # Deterministic backtracking solver
│   └── pygame_ui.py           # High-tech pygame interface
=======
│   ├── rl_agent.py            # Double DQN agent implementation
│   ├── backtracking_solver.py # Deterministic backtracking solver
│   └── pygame_ui.py           # Interactive pygame interface
>>>>>>> 9e839a1566699572ecf80c319b1a8ea47ee095ac
├── scripts/                   # Executable scripts
│   ├── train.py               # Training script
│   └── solver.py              # Inference & visualization
├── notebooks/                 # Analysis notebooks
│   └── solver_comparison.ipynb  # Backtracking vs RL comparison
├── tests/                     # Test suite
│   ├── test_config.py
│   ├── test_sudoku_game.py
<<<<<<< HEAD
=======
│   ├── test_pygame_ui.py
>>>>>>> 9e839a1566699572ecf80c319b1a8ea47ee095ac
│   ├── test_rl_agent.py
│   └── test_backtracking_solver.py
└── models/                    # Saved trained models
    └── sudoku_dqn_*.pth
```

## Quick Start

### 1. Train the Agent (Optional)
```bash
# Train on CPU
python scripts/train.py --episodes 1000 --difficulty medium --device cpu

# Train on GPU (requires CUDA-enabled conda env)
python scripts/train.py --episodes 1000 --difficulty medium --device cuda
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
- **Algorithm**: Deep Q-Network (DQN) with Experience Replay
- **State Representation**: 9×9×10 tensor (position × 10 digit possibilities)
- **Action Space**: 81 × 9 (cell × digit selections)
- **Reward System**:
  - +1 for a valid move
  - +10 for placing the correct digit (matches solution)
  - −10 for a wrong digit or invalid move
  - +200 for puzzle completion

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

### Deterministic Backtracking Solver
- **Algorithm**: Constraint propagation (naked singles) + recursive backtracking
- **Heuristic**: Minimum Remaining Values (MRV) – always branches on the cell with the fewest candidates
- **Guarantee**: Finds a valid solution whenever one exists (100 % success rate)
- **Speed**: Solves most 9×9 puzzles in < 5 ms

### Solver Comparison Notebook
A Jupyter notebook (`notebooks/solver_comparison.ipynb`) benchmarks both solvers
on easy / medium / hard puzzles and compares correctness, speed, and reliability.
Run it with:
```bash
cd notebooks && jupyter notebook solver_comparison.ipynb
```

### Deterministic Backtracking Solver
- **Algorithm**: Constraint propagation (naked singles) + recursive backtracking
- **Heuristic**: Minimum Remaining Values (MRV) – always branches on the cell with the fewest candidates
- **Guarantee**: Finds a valid solution whenever one exists (100 % success rate)
- **Speed**: Solves most 9×9 puzzles in < 5 ms

### Solver Comparison Notebook
A Jupyter notebook (`notebooks/solver_comparison.ipynb`) benchmarks both solvers
on easy / medium / hard puzzles and compares correctness, speed, and reliability.
Run it with:
```bash
cd notebooks && jupyter notebook solver_comparison.ipynb
```

### UI Features
- **Modern Design**: Dark theme with neon accents
- **Animations**: 
  - Smooth cell highlighting
  - Particle effects on valid solutions
  - Transition animations between states
  - Real-time solving visualization
- **Interactive Controls**:
  - Mouse/keyboard input
  - Real-time hint system
  - Step-by-step execution
  - Difficulty selection

## Configuration

Edit `src/config.py` to customize:
<<<<<<< HEAD
- Neural network architecture
- Learning hyperparameters (α, γ, ε)
- Replay buffer size
- Training episodes
=======
- Neural network architecture (conv channels, hidden layer sizes)
- Learning hyperparameters (LR = 0.0005, γ = 0.99, ε decay = 0.995)
- Replay buffer size (50 000) and batch size (128)
- Target network sync frequency (every 100 steps)
>>>>>>> 9e839a1566699572ecf80c319b1a8ea47ee095ac
- UI theme and animation speed

## Performance Metrics

Trained agent solves:
- **Easy puzzles** (40 givens): ~95% success rate
- **Medium puzzles** (30 givens): ~85% success rate
- **Hard puzzles** (20 givens): ~70% success rate

**Note**: RL is interesting for learning puzzle patterns, but deterministic backtracking remains faster for guaranteed solutions.

## References

- [DQN Paper](https://arxiv.org/abs/1312.5602) - Human-level control through deep RL
- [RL Sudoku Study](https://nlp.fi.muni.cz/aiproject/ui/karmazin_vasilii2024/SudokuRL.pdf)
- [Pygame Documentation](https://www.pygame.org/docs/)
