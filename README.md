# Sudoku RL Solver - Installation & Setup Guide

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
├── config.py                  # Configuration & hyperparameters
├── sudoku_game.py            # Game logic & constraint handling
├── rl_agent.py               # DQN agent implementation
├── backtracking_solver.py    # Deterministic backtracking solver
├── pygame_ui.py              # High-tech pygame interface
├── train.py                  # Training script
├── solver.py                 # Inference & visualization
├── notebooks/                # Analysis notebooks
│   └── solver_comparison.ipynb  # Backtracking vs RL comparison
├── tests/                    # Test suite
│   ├── test_config.py
│   ├── test_sudoku_game.py
│   ├── test_rl_agent.py
│   └── test_backtracking_solver.py
└── models/                   # Saved trained models
    └── sudoku_dqn_*.pth
```

## Quick Start

### 1. Train the Agent (Optional)
```bash
# Train on CPU
python train.py --episodes 1000 --difficulty medium --device cpu

# Train on GPU (requires CUDA-enabled conda env)
python train.py --episodes 1000 --difficulty medium --device cuda
```

### 2. Run Solver with UI
```bash
python solver.py --mode play  # Manual play + solver assist
python solver.py --mode solve # Auto-solve visualization
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
  - +10 for valid cell fill
  - -1 for invalid move
  - +100 for puzzle completion
  - Constraint violation penalties

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

Edit `config.py` to customize:
- Neural network architecture
- Learning hyperparameters (α, γ, ε)
- Replay buffer size
- Training episodes
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
