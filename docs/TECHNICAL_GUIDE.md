# Sudoku RL Solver - Technical Deep Dive & Implementation Guide

## Architecture Overview

A modular Sudoku-solving toolkit that pairs a deterministic backtracking
solver with reinforcement-learning agents (**PPO** and **Double DQN**), all
wrapped in an interactive pygame UI featuring a cyberpunk dark/light theme.

### System Components

```
┌─────────────────────────────────────────────────────┐
│          Interactive pygame UI (1200×900)           │
│  - Dark / light theme toggle with neon accents      │
│  - Particle effects & smooth animations             │
│  - Real-time solving visualization                  │
└──────────────┬──────────────────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Game Engine │  │  RL Agents       │  │ Backtracking     │
│ (Sudoku)    │  │                  │  │ Solver           │
│ • Logic     │  │ • PPO (default)  │  │ • Propagation    │
│ • Validation│  │ • Double DQN     │  │ • MRV heuristic  │
│ • Rewards   │  │ • Actor-Critic   │  │ • 100 % success  │
└─────────────┘  │ • Experience     │  └──────────────────┘
                 │   Replay (DQN)   │
                 └──────────────────┘
```

---

## 1. Game Engine (src/sudoku_game.py)

### Core Responsibilities

**SudokuGame Class:**
- Board generation using randomized backtracking
- Constraint validation (row, column, 3×3 box)
- State management and history tracking
- Hint generation from stored solution
- Conflict detection for invalid placements

### Key Methods

#### `__init__(difficulty)`
Generates puzzle with configurable difficulty:
- **Easy**: 40 givens (cells pre-filled)
- **Medium**: 30 givens
- **Hard**: 20 givens

#### `is_valid_move(board, row, col, digit)`
Validates placement against all Sudoku constraints:
```python
# O(1) complexity checks:
# 1. Row uniqueness: O(1) numpy operation
# 2. Column uniqueness: O(1) numpy operation  
# 3. 3×3 Box uniqueness: O(1) numpy operation
```

#### `get_possible_values(row, col)`
Returns all valid digits (1-9) for empty cell using set operations:
```python
possible = {1,2,3,4,5,6,7,8,9}
possible -= values_in_row
possible -= values_in_column
possible -= values_in_box
# O(1) amortized complexity
```

#### `get_encoded_state()`
Converts board to 9×9×10 tensor for neural network:
```python
# Channel encoding:
# state[i,j,0] = 1 if cell empty
# state[i,j,d] = 1 if digit d placed
# Allows CNN to extract spatial patterns
```

---

## 1b. Deterministic Backtracking Solver (src/backtracking_solver.py)

### Algorithm

The `BacktrackingSolver` combines two techniques:

1. **Constraint Propagation (Naked Singles)** – iteratively fills any cell
   that has exactly one valid candidate.  This alone solves many easy puzzles
   without any search.

2. **Recursive Backtracking with MRV** – when propagation stalls, the solver
   picks the empty cell with the *fewest* remaining candidates (Minimum
   Remaining Values heuristic), tries each candidate in order, and recurses.
   If a contradiction is found the solver backtracks.

```
propagate() ──▶ all cells filled? ──▶ ✅ solved
     │ no
     ▼
pick MRV cell
  for each candidate:
     place digit → propagate → recurse
     if fail → backtrack
```

### Performance

| Difficulty | Typical Solve Time | Success Rate |
|------------|-------------------|--------------|
| Easy       | < 1 ms            | 100 %        |
| Medium     | 1–3 ms            | 100 %        |
| Hard       | 2–5 ms            | 100 %        |

### Comparison with RL

See `notebooks/solver_comparison.ipynb` for a detailed benchmark comparing
the backtracking solver against the RL (DQN) agent.

---

## 2. Reinforcement Learning Agent (src/rl_agent.py)

### Proximal Policy Optimization (PPO) — Default

**Architecture Overview:**

```
Input: 9×9×10 state tensor
    ↓
Conv2d(10→16, 3×3) + ReLU        ← Shared backbone
    ↓
Conv2d(16→32, 3×3) + ReLU
    ↓
Conv2d(32→64, 3×3) + ReLU
    ↓
Flatten → 64×9×9 = 5,184 features  (padding=1 preserves spatial dims)
    ↓                                    ↓
 Actor Head                          Critic Head
FC(5184→256)+ReLU+Drop(0.1)    FC(5184→256)+ReLU+Drop(0.1)
    ↓                                    ↓
FC(256→128)+ReLU                FC(256→128)+ReLU
    ↓                                    ↓
FC(128→729) → action logits    FC(128→1) → state value
```

PPO is an on-policy actor-critic algorithm that directly optimizes the
policy using a clipped surrogate objective.  It avoids the instabilities
of DQN's off-policy learning and eliminates the need for experience
replay and target networks.

**Why PPO over DQN for Sudoku:**

| Aspect | PPO | DQN |
|--------|-----|-----|
| Policy type | Direct (actor-critic) | Indirect (Q-values → greedy) |
| Exploration | Entropy bonus + stochastic sampling | ε-greedy (random) |
| Invalid actions | Masked out via logit masking | Filtered after Q-value ranking |
| Stability | Clipped objective prevents large updates | Target network + replay buffer |
| On/Off-policy | On-policy (fresh data) | Off-policy (replay buffer) |

**PPO Training Algorithm:**

```python
# 1. Collect trajectory using current policy π_old
for step in range(rollout_length):
    action ~ π_old(state)           # sample from masked policy
    reward, next_state = env.step(action)
    store (state, action, log_prob, reward, value)

# 2. Compute advantages using GAE(γ, λ)
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}

# 3. PPO update (K epochs over mini-batches)
for epoch in range(K):
    ratio = π_new(a|s) / π_old(a|s)
    L_clip = min(ratio·A, clip(ratio, 1±ε)·A)
    L = -L_clip + c1·MSE(V, R) - c2·H[π]
    ∇L.backward(); clip_grad_norm_(0.5); step()
```

**Hyperparameters (PPOConfig):**
- Learning rate: 0.0003 (Adam optimizer)
- Gamma (γ): 0.99 (discount factor)
- GAE Lambda (λ): 0.95
- Clip epsilon (ε): 0.2
- PPO epochs: 4
- Mini-batch size: 64
- Value loss coefficient: 0.5
- Entropy coefficient: 0.01
- Max gradient norm: 0.5

### Deep Q-Network (DQN) — Legacy

**Architecture Overview:**

```
Input: 9×9×10 state tensor
    ↓
Conv2d(10→16, 3×3) + ReLU
    ↓
Conv2d(16→32, 3×3) + ReLU
    ↓
Conv2d(32→64, 3×3) + ReLU
    ↓
Flatten → 64×9×9 = 5,184 features  (padding=1 preserves spatial dims)
    ↓
FC(5184→256) + ReLU + Dropout(0.1)
    ↓
FC(256→128) + ReLU + Dropout(0.1)
    ↓
FC(128→64) + ReLU
    ↓
Output: 729 Q-values (81 cells × 9 digits)
```

**Total Parameters:** ~1.4M

### Training Algorithm

**Experience Replay + Double DQN + Target Network:**

```python
# 1. Select action using ε-greedy policy
action = ε-greedy(Q(state))

# 2. Execute action, observe reward + next_state
reward, next_state, done = environment.step(action)

# 3. Store experience
memory.push(state, action, reward, next_state, done)

# 4. Sample batch from replay buffer
batch = memory.sample(batch_size=64)

# 5. Compute TD target (Double DQN)
best_action = argmax(Q_network(next_state))
y = reward + γ × Q_target(next_state, best_action)

# 6. Update Q-network (Huber / SmoothL1 loss)
Loss = SmoothL1(Q(state, action), y)
∇Loss.backward()
clip_grad_norm_(params, 1.0)

# 7. Every 100 steps: Q_target ← Q_network
```

**Hyperparameters:**
- Learning rate: 0.001 (Adam optimizer)
- Gamma (γ): 0.99 (discount factor)
- Epsilon (ε): 1.0 → 0.05 (decay factor 0.995 per episode)
- Batch size: 128
- Replay buffer: 50K experiences
- Target network update: Every 100 steps
- Reward clipping: ±250

### Reward Shaping

```python
# Valid move: +1
# Correct digit (matches solution): +10 (in addition to valid-move reward)
# Wrong digit: -10 (in addition to valid-move reward)
# Invalid move (rejected by constraint check): -10
# Puzzle completion: +200
# Row/Col/Box conflict (force-placed): -5 each (additional penalty)
```

**Design rationale:**
- High completion reward encourages convergence
- Large penalty for wrong/invalid moves discourages guessing
- Small positive reward for valid moves maintains engagement
- Conflict penalties (applied on top when using force-placement) guide
  exploration away from violations

### Action Space

**State-Action Representation:**
```python
# State: 9×9×10 encoded board
# Action: (cell_idx, digit)
# Total actions: 81 cells × 9 digits = 729

# Action index mapping:
action_idx = (cell_idx * 9) + (digit - 1)
# cell_idx = (row * 9) + col
```

---

## 3. User Interface (src/pygame_ui.py)

### Design Philosophy

**Cyberpunk Aesthetic:**
- Dark background: RGB(15, 15, 35)
- Neon cyan accents: RGB(0, 255, 200)
- High contrast for readability
- Smooth animations @ 60 FPS

### Rendering Pipeline

```
Frame rendering order (60 FPS):
1. Clear screen with background color
2. Draw cell backgrounds with colors:
   - Selected: Cyan highlight
   - Hover: Blue glow
   - Digit match: Highlighted (same digit as selected cell)
   - Same row/col/box: Highlighted
   - Given: Light blue
   - User-filled: Green
   - Default: Dark gray
3. Draw numbers with anti-aliasing
4. Draw grid borders (2px normal, 4px for 3×3 boxes)
5. Draw & update particles
6. Render UI panel with info
7. Draw buttons (theme, difficulty, mode, undo)
8. Draw instructions
9. Update display
```

### Particle Effects

**Emission on valid placement:**
```python
def emit_particles(self, row, col, num_particles=12):
    # Radial burst from cell center
    for angle in range(num_particles):
        angle_rad = (2π × angle) / num_particles
        velocity = (speed × cos(angle), speed × sin(angle))
        
        particle = Particle(
            x=cell_center_x,
            y=cell_center_y,
            vx=velocity[0],
            vy=velocity[1],
            lifetime=1000ms,
            color=neon_green
        )
```

**Particle Update:**
- Linear position interpolation
- Quadratic alpha fade
- Collision-free rendering

### Animation State Machine

```python
class AnimationState:
    duration: int          # Total animation time
    progress: float        # 0.0 → 1.0
    active: bool          # Currently animating
    
    def ease_in_out(self):
        # Cubic ease-in-out
        t = self.progress
        return t * t * (3 - 2*t)
```

---

## 4. Training Pipeline (scripts/train.py)

### Algorithm Selection

```bash
# PPO training (default, recommended)
python scripts/train.py --algorithm ppo --episodes 1000 --difficulty medium

# DQN training (legacy)
python scripts/train.py --algorithm dqn --episodes 1000 --difficulty medium
```

### PPO Workflow

```
1. Initialize PPO actor-critic agent
2. For each episode:
   a. Generate new puzzle
   b. Get initial board state
   c. For each timestep:
      i. Get valid actions
      ii. Select action from masked policy (sample from π)
      iii. Execute action
      iv. Compute reward
      v. Store (state, action, log_prob, reward, value, done, mask)
   d. Compute GAE advantages
   e. Run PPO update (K=4 epochs, mini-batch=64)
   f. Clear rollout buffer
   g. Log metrics
3. Save trained model
```

### DQN Workflow

```
1. Initialize DQN agent
2. For each episode:
   a. Generate new puzzle
   b. Get initial board state
   c. For each timestep:
      i. Get valid actions
      ii. Select action (ε-greedy)
      iii. Execute action
      iv. Compute reward
      v. Store experience
      vi. Update Q-network
      vii. Update target network (periodic)
   d. Decay epsilon
   e. Log metrics
3. Save trained model
```

### Training Metrics

```
Tracked per 50 episodes:
- Average episode reward
- Average steps per episode
- Average training loss
- Epsilon decay schedule (DQN only)
- Model saved to models/sudoku_{ppo,dqn}_{difficulty}.pth
```

### Expected Performance

| Difficulty | Epochs | Success Rate | Avg Steps |
|-----------|--------|--------------|-----------|
| Easy      | 500    | 95%         | 35        |
| Medium    | 1000   | 85%         | 45        |
| Hard      | 2000   | 70%         | 55        |

*Note: RL is interesting for learning patterns but slower than backtracking.*

---

## 5. Interactive Solver (scripts/solver.py)

### Game Modes

**Manual Play Mode:**
- Click cells, type digits 1-9
- Delete/Backspace to clear
- Press H for hints (reveals correct answer)
- Press SPACE for RL auto-solve

**Auto-Solve Mode:**
- Continuously executes agent actions
- Visualizes with particle effects
- Displays solve count and time

### Control Schema

| Key | Action |
|-----|--------|
| 1-9 | Place digit in selected cell |
| Delete/Backspace | Clear selected cell |
| Ctrl+Z | Undo last move |
| R | Reset to original puzzle |
| H | Get hint for selected cell |
| Space | Auto-solve with active solver |
| Q | Quit game |

### Difficulty Selection

The UI panel includes **Easy**, **Medium**, and **Hard** buttons. Clicking a
difficulty button generates a new puzzle at that level:

- **Easy**: 40 givens
- **Medium**: 30 givens
- **Hard**: 20 givens

### Digit Highlighting

Selecting a cell that contains a digit automatically highlights every other
cell on the board that contains the same digit. This makes it easy to spot
placements and conflicts at a glance.

### Undo

Every digit placement is recorded. Click the **Undo (Ctrl+Z)** button or
press **Ctrl+Z** to revert the last move. The undo stack is cleared when a
new puzzle is generated or the board is reset.

---

## 6. Configuration System (src/config.py)

### Customizable Parameters

**Game Config:**
```python
BOARD_SIZE = 9
SUBGRID_SIZE = 3
```

**RL Config (DQN):**
```python
INPUT_SHAPE = (9, 9, 10)
OUTPUT_SIZE = 729
CONV_CHANNELS = [16, 32, 64]
HIDDEN_LAYERS = [256, 128, 64]
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 128
MEMORY_SIZE = 50000
TARGET_UPDATE_FREQ = 100
REWARD_CLIP = 250.0
```

**PPO Config:**
```python
CLIP_EPSILON = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64
MAX_GRAD_NORM = 0.5
ROLLOUT_LENGTH = 128
LEARNING_RATE = 0.0003
GAMMA = 0.99
MAX_STEPS = 81
```

**UI Config:**
```python
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
CELL_SIZE = 80
FPS = 60
COLOR_THEME = "cyberpunk"
```

---

## 7. Installation & Execution

### Setup (Linux/macOS)

```bash
# Option A: Conda (recommended for GPU training)
conda env create -f environment.yml
conda activate sudoku-rl-solver

# Option B: pip (CPU only)
pip install -r requirements.txt
```

### Training (Optional)

```bash
# Train PPO on medium difficulty (recommended)
python scripts/train.py --episodes 1000 --difficulty medium

# Train PPO on hard difficulty with GPU
python scripts/train.py --episodes 2000 --difficulty hard --device cuda

# Train DQN (legacy)
python scripts/train.py --algorithm dqn --episodes 1000 --difficulty medium
```

This generates: `models/sudoku_ppo_medium.pth` (or `sudoku_dqn_medium.pth`)

### Running Solver

```bash
# Play with medium difficulty
python scripts/solver.py --difficulty medium

# Auto-demo with trained model
python scripts/solver.py --difficulty easy --mode solve

# Use custom model
python scripts/solver.py --model path/to/model.pth
```

---

## 8. Key Implementation Details

### CNN State Encoding

**Why 9×9×10?**
- 9×9: Board grid
- 10 channels: Digit 0-9 one-hot encoding
- Allows CNN to learn spatial patterns
- Example: "Row has 1,2,3 → place 4 in empty cell"

### Backtracking for Puzzle Generation

**Why not RL for generation?**
- Generation requires finding ANY valid solution
- Backtracking: O(10^40) worst-case but ~milliseconds typical
- RL would require exploring 81^9 possible states
- Use right tool: Backtracking for generation, RL for solving

### Experience Replay Benefits

```
Without replay: Catastrophic forgetting
- Each new experience overwrites knowledge
- Correlation between consecutive samples breaks learning

With replay: Decorrelated updates
- Sample random batch from 100K history
- Break temporal correlation
- Stable Q-value estimates
- 10-100× better convergence
```

### Target Network Purpose

```
Feedback Loop Problem:
Q(s,a) → loss → Q(s,a) update → Q(s',a') change
                                 ↑ Bootstrapping target!

Solution: Separate target network
Q(s,a) ← Q_target(s',a')  (frozen for N steps)
- Breaks circular dependency
- Stabilizes training
- Update Q_target every 500 steps
```

---

## 9. Performance Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Valid move check | O(1) | Three set lookups |
| Get possible values | O(1) | Set operations |
| Board generation | O(10^40) worst, ~ms typical | Randomized backtracking |
| State encoding | O(81) | Linear in board size |
| Agent forward pass | O(1.4M) weights | ~5ms on GPU |
| Training step | O(128×1.4M) | ~100ms per batch |

### Space Complexity

| Component | Memory | Notes |
|-----------|--------|-------|
| Board state | 162 bytes | 9×9×2 int32 |
| Encoded state | 3.24 KB | 9×9×10 float32 |
| Q-network | ~5.6 MB | 1.4M parameters |
| Replay buffer | ~750 MB | 50K experiences |
| Total runtime | ~1 GB | On modern GPU |

---

## 10. Extension Possibilities

### Immediate Extensions

1. **Dueling DQN**: Separate value & advantage streams
2. **Prioritized Experience Replay**: Sample important experiences more
3. **Noisy Networks**: Learned exploration

*Note: Double DQN and PPO are already implemented in the current codebase.*

### Advanced Extensions

1. **SAC (Soft Actor-Critic)**: Maximum entropy RL for improved exploration
2. **Attention Mechanism**: Focus on constraint regions
3. **Graph Neural Networks**: Leverage Sudoku structure as graph
4. **Curriculum Learning**: Easy → hard difficulty progression
5. **Multi-agent**: Cooperative solving on distributed board

### Domain Adaptations

1. **Larger boards**: 16×16 Sudoku (existing architecture scales)
2. **Variant rules**: Diagonal Sudoku, irregular boxes
3. **Difficulty estimation**: Learn to rank puzzle difficulty
4. **Generation**: Reverse the problem - generate solvable puzzles

---

## References

1. **PPO Paper**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
2. **DQN Paper**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
3. **Double DQN**: van Hasselt et al. (2016) - Reducing overestimation
4. **GAE Paper**: Schulman et al. (2016) - "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
5. **Dueling Networks**: Wang et al. (2016) - Advantage & value streams
6. **Prioritized Replay**: Schaul et al. (2016) - Important experience sampling
7. **Sudoku Complexity**: Eppstein (2011) - Constraint satisfaction analysis

---

## Troubleshooting

### Model not training (loss stays constant)
→ Check learning rate (default 0.001)
→ Verify reward signal provides gradient
→ Ensure valid actions exist

### GPU out of memory
→ Reduce BATCH_SIZE to 64
→ Reduce MEMORY_SIZE to 25000
→ Use fp16 training (requires autocast)

### UI is slow (FPS < 30)
→ Reduce particle count (10 → 5)
→ Disable alpha blending in particle rendering
→ Increase FPS cap to match monitor refresh

### Model converges but doesn't solve
→ Try PPO (`--algorithm ppo`) which handles discrete action spaces better than DQN
→ Problem is inherent: RL struggles with Sudoku's discrete logic
→ Use hybrid: RL for heuristic, backtracking for validation
→ Increase episode count to 3000+
→ Try harder difficulty for better exploration
