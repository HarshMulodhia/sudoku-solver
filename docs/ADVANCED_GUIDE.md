# Sudoku RL Solver - Advanced Tips & Best Practices

## Overview

This guide covers advanced techniques and research directions for the
Sudoku RL Solver.  The project already ships with a Double DQN agent
(CNN + FC, ~1.4 M parameters) and a deterministic backtracking solver.
The tips below can help you push the RL component further.

## For Your Research Background (Autonomous Systems)

As an autonomous systems researcher, you'll appreciate the parallels between Sudoku solving and motion planning:

### Sudoku as a Constraint Satisfaction Problem (CSP)

**Analogy to Path Planning:**
```
Motion Planning:
- State: (x, y, θ) robot pose
- Constraints: Obstacles, dynamics limits
- Goal: Reach target while satisfying constraints

Sudoku CSP:
- State: 81 cell values
- Constraints: Row, column, box uniqueness
- Goal: Fill all cells satisfying constraints

Both: Combinatorial search in constrained state spaces
```

### Graph-Based Optimization Approach

**Apply GCS Framework:**

```python
# Graph of Convex Sets for Sudoku (hybrid approach)

# 1. Create nodes for each valid placement
# 2. Edges connect compatible moves
# 3. Convex relaxation of feasibility
# 4. Solve with convex optimizer

# Example: Use CVXPY for constraint propagation
import cvxpy as cp

# Variables: x[i,j,d] ∈ {0,1} = "digit d at (i,j)"
x = cp.Variable((9,9,9), boolean=True)

# Constraints
constraints = [
    # Row uniqueness
    cp.sum(x[i,:,:], axis=0) == 1 for i in range(9),
    # Column uniqueness
    cp.sum(x[:,j,:], axis=0) == 1 for j in range(9),
    # Box uniqueness
    cp.sum(x[3*bi:3*bi+3, 3*bj:3*bj+3, :], axis=(0,1)) == 1 
    for bi in range(3) for bj in range(3)
]

problem = cp.Problem(cp.Minimize(0), constraints)
problem.solve()
```

**Why this matters:**
- Convex relaxation provides lower bounds for search
- Dual variables guide RL exploration
- Combines symbolic and learning-based methods

---

## Performance Optimization for Your Setup

### GPU Acceleration (PyTorch)

**Device auto-detection:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check GPU specs
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Batch processing for faster training:**
```python
# Instead of episode-by-episode
def train_batch(games_batch, agent, batch_size=32):
    """Parallel training on multiple games"""
    states = torch.stack([g.get_encoded_state() for g in games_batch])
    actions = []
    
    # Forward pass (vectorized)
    with torch.no_grad():
        q_values = agent.q_network(states.to(device))  # (batch_size, 729)
    
    # Process all games at once
    return experiences
```

### Memory Optimization

**For large-scale training (100k+ episodes):**

```python
# 1. Use float16 for embeddings
embeddings = torch.nn.Linear(...).half()

# 2. Disk-based replay buffer
class DiskReplayBuffer:
    """Store experiences on disk, load on demand"""
    def __init__(self, buffer_path='./replay_buffer'):
        self.path = buffer_path
        self.idx = 0
    
    def push(self, experience):
        np.save(f'{self.path}/{self.idx}.npy', experience)
        self.idx += 1
    
    def sample(self, batch_size):
        indices = np.random.choice(self.idx, batch_size)
        batch = [np.load(f'{self.path}/{i}.npy') for i in indices]
        return batch

# 3. Gradient accumulation
for i, batch in enumerate(batches):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Advanced RL Techniques

### 1. Dueling DQN (Better Value Estimation)

**Architecture variation:**

```python
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_shape, output_size, hidden_layers):
        super().__init__()
        
        # Shared feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Value stream: estimate state value
        self.value_stream = nn.Sequential(
            nn.Linear(128*9*9, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream: estimate action advantages
        self.advantage_stream = nn.Sequential(
            nn.Linear(128*9*9, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        features = self.features(x).view(x.size(0), -1)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + [A(s,a) - mean(A(s,.))]
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
```

**Benefits:**
- Better value estimation
- Improved convergence
- 20-30% faster learning

### 2. Prioritized Experience Replay

```python
class PrioritizedReplayBuffer:
    """Sample important experiences more frequently"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.buffer = []
        self.priorities = []
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sample weight
    
    def push(self, experience, td_error):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size):
        # Sample based on priorities
        probabilities = self.priorities / np.sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Importance weights (correct for sampling bias)
        weights = (1 / (len(self.buffer) * probabilities[indices])) ** self.beta
        weights /= weights.max()  # Normalize
        
        batch = [self.buffer[i] for i in indices]
        return batch, weights, indices
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
```

### 3. Double DQN (Already Implemented)

The codebase already uses Double DQN in `rl_agent.py`:

```python
# Standard DQN (tends to overestimate Q-values)
target = reward + gamma * max(Q_target(next_state))
#                              ^ This max operation overestimates

# Double DQN (decoupled selection & evaluation) — current implementation
best_action = argmax(Q_network(next_state))  # Select best action
target = reward + gamma * Q_target(next_state, best_action)
#                          ^ Evaluate using different network
```

### 4. Noisy Networks (Learned Exploration)

```python
class NoisyLinear(nn.Module):
    """Linear layer with learnable noise for exploration"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.ones(out_features, in_features) * 0.017)
        
        # Bias
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_sigma = nn.Parameter(torch.ones(out_features) * 0.017)
        
        # Noise buffers
        self.register_buffer('weight_epsilon', torch.randn(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.randn(out_features))
    
    def forward(self, x):
        # Add learnable noise for exploration
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)
    
    def sample_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
```

---

## Curriculum Learning Strategy

### Progressive Difficulty Scheduling

```python
def get_curriculum_difficulty(episode):
    """Easy → Medium → Hard progression"""
    total_episodes = 5000
    
    # Easy for first 30%
    if episode < total_episodes * 0.3:
        return 'easy'
    # Medium for next 50%
    elif episode < total_episodes * 0.8:
        return 'medium'
    # Hard for final 20%
    else:
        return 'hard'

# Training loop
for episode in range(5000):
    difficulty = get_curriculum_difficulty(episode)
    game = SudokuGame(difficulty=difficulty)
    # ... train on this game
```

**Why it works:**
- Easy puzzles teach basic patterns
- Progressive difficulty prevents overfitting
- Hard puzzles refine strategy
- Smoother learning curve

---

## Hybrid Approaches (Recommended for Production)

### Strategy 1: RL + Backtracking

```python
class HybridSudokuSolver:
    """Combine RL heuristics with backtracking certainty"""
    
    def solve(self, game):
        # Step 1: Use RL agent for heuristics (20 steps)
        for _ in range(20):
            valid_actions = self.agent.get_valid_actions(game)
            if not valid_actions:
                break
            
            row, col, digit = self.agent.select_action(
                game.get_encoded_state(),
                valid_actions,
                training=False
            )
            game.place_digit(row, col, digit)
        
        # Step 2: Use backtracking for guaranteed solution
        if not game.is_complete():
            self._backtrack_solve(game.board)
        
        return game.board
    
    def _backtrack_solve(self, board):
        """Standard backtracking - guaranteed to find solution"""
        for i in range(9):
            for j in range(9):
                if board[i,j] == 0:
                    for digit in range(1, 10):
                        if self.game.is_valid_move(board, i, j, digit):
                            board[i,j] = digit
                            if self._backtrack_solve(board):
                                return True
                            board[i,j] = 0
                    return False
        return True
```

### Strategy 2: RL for Constraint Propagation

```python
class ConstraintPropagationSolver:
    """Use RL to learn which constraints to propagate"""
    
    def solve(self, game):
        # RL learns: "When cell has 1 possible value, fill it"
        while True:
            moved = False
            
            # Find cell with minimum possible values
            min_possibilities = 10
            best_cell = None
            
            for i in range(9):
                for j in range(9):
                    if game.board[i,j] == 0:
                        possibilities = game.get_possible_values(i, j)
                        if len(possibilities) == 1:
                            # Naked single: fill immediately
                            digit = possibilities.pop()
                            game.place_digit(i, j, digit)
                            moved = True
                        elif len(possibilities) < min_possibilities:
                            min_possibilities = len(possibilities)
                            best_cell = (i, j, possibilities)
            
            if not moved:
                # Use RL agent for next move
                if best_cell:
                    i, j, possibilities = best_cell
                    digit = self.agent_select_from_possibilities(possibilities)
                    game.place_digit(i, j, digit)
                else:
                    break
```

---

## Visualization & Analysis

### Training Progress Monitoring

```python
import matplotlib.pyplot as plt

def plot_training_metrics(rewards, losses, epsilons):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Reward curve with moving average
    axes[0].plot(rewards, alpha=0.3, label='Episode reward')
    axes[0].plot(pd.Series(rewards).rolling(50).mean(), label='MA(50)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid()
    
    # Loss curve
    axes[1].plot(losses, alpha=0.7)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss (Huber)')
    axes[1].set_yscale('log')
    axes[1].grid()
    
    # Epsilon decay
    axes[2].plot(epsilons)
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Epsilon')
    axes[2].grid()
    
    plt.tight_layout()
    plt.show()
```

### Q-Value Analysis

```python
def analyze_q_values(agent, game):
    """Analyze learned Q-values for a given state"""
    state = game.get_encoded_state()
    
    with torch.no_grad():
        q_values = agent.q_network(torch.tensor(state).unsqueeze(0))
        q_values = q_values.squeeze(0).cpu().numpy()
    
    # Find highest Q-value moves
    top_k = 10
    top_actions = np.argsort(q_values)[-top_k:][::-1]
    
    print("Top 10 moves by Q-value:")
    for rank, action_idx in enumerate(top_actions):
        cell_idx = action_idx // 9
        digit = (action_idx % 9) + 1
        row, col = cell_idx // 9, cell_idx % 9
        
        if game.board[row, col] == 0:  # Only if cell is empty
            print(f"{rank+1}. Place {digit} at ({row},{col}): Q={q_values[action_idx]:.2f}")
```

---

## Testing & Validation

### Unit Tests

```python
import pytest

def test_valid_move():
    game = SudokuGame('easy')
    game.board[0, 0] = 0  # Clear a cell
    
    # Valid move should succeed
    assert game.is_valid_move(game.board, 0, 0, 5) == True
    
    # Duplicate in row should fail
    game.board[0, 1] = 5
    assert game.is_valid_move(game.board, 0, 0, 5) == False

def test_agent_determinism():
    """Agent with same state should select same action"""
    agent = SudokuRLAgent()
    game = SudokuGame('easy')
    state = game.get_encoded_state()
    valid_actions = agent.get_valid_actions(game)
    
    action1 = agent.select_action(state, valid_actions, training=False)
    action2 = agent.select_action(state, valid_actions, training=False)
    
    assert action1 == action2
```

---

## Deployment Best Practices

### Model Checkpointing

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_reward = -np.inf
    
    def save_if_best(self, agent, episode_reward, episode):
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            path = f'{self.checkpoint_dir}/best_model.pth'
            agent.save_model(path)
            print(f"New best model saved (reward={episode_reward:.1f})")
    
    def save_periodic(self, agent, episode, interval=100):
        if episode % interval == 0:
            path = f'{self.checkpoint_dir}/checkpoint_ep{episode}.pth'
            agent.save_model(path)
```

### Model Quantization (For deployment)

```python
# Reduce model size 75% without significant accuracy loss
def quantize_model(agent):
    model_fp32 = agent.q_network
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return model_int8
```

---

## Common Pitfalls & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Loss not decreasing | Bad learning rate | Default is 0.0001; try decaying by 0.1 every 500 steps |
| Agent never learns | No valid actions | Verify `get_valid_actions()` returns moves |
| Training diverges | Gradient explosion | Gradient clipping (max norm 1.0) and reward clipping (±50) are already enabled |
| Overfitting to difficulty | No variety | Use curriculum learning or data augmentation |
| Slow training | Memory bottleneck | Use mixed precision (float16) or batch accumulation |
| GPU memory full | Large batch | Reduce BATCH_SIZE (default 64) or MEMORY_SIZE (default 50 000) |

---

## Research Extensions

### 1. Neuro-Symbolic Integration

Combine neural networks with symbolic reasoning:
```python
# Neural: Learn patterns
# Symbolic: Enforce hard constraints

def neuro_symbolic_solve(game, agent):
    for step in range(100):
        # Neural component: agent suggests moves
        move = agent.select_best_action(game)
        
        # Symbolic component: validate and constrain
        if not game.is_valid_move(*move):
            # Fallback to constraint propagation
            move = constraint_propagate(game)
        
        if move:
            game.place_digit(*move)
        else:
            break
    
    return game.is_solved()
```

### 2. Meta-Learning (Learn to Learn)

Train agent on diverse puzzles to generalize:
```python
# MAML (Model-Agnostic Meta-Learning)
def meta_train(agent, task_distribution):
    for meta_step in range(100):
        # Sample task (puzzle)
        task = task_distribution.sample()
        
        # Inner loop: adapt to this task
        for inner_step in range(10):
            gradient = compute_gradient(agent, task)
            agent.inner_update(gradient)
        
        # Outer loop: update for next task
        meta_gradient = compute_meta_gradient(agent)
        agent.outer_update(meta_gradient)
```

### 3. Adversarial Training

Generate hard puzzles that fool the agent:
```python
# Adversary generates puzzles, agent solves
# Iterate to improve both

def adversarial_training(solver_agent, adversary_agent):
    for iteration in range(100):
        # Adversary generates hard puzzle
        puzzle = adversary_agent.generate_puzzle()
        
        # Solver attempts to solve
        success = solver_agent.try_solve(puzzle)
        
        # Update both agents
        if success:
            # Adversary learns to make harder puzzles
            adversary_agent.increase_difficulty()
        else:
            # Solver learns from failure
            solver_agent.learn_from_failure(puzzle)
```

---

## Final Recommendations for Your Project

Given your robotics background:

1. **Start with the hybrid approach** - RL for exploration, backtracking for guarantees
2. **Benchmark against classical solvers** - Understand where RL adds value
3. **Use this as RL learning foundation** - Apply techniques to motion planning later
4. **Publish results** - Interesting for RL in discrete domains
5. **Consider graph-based extensions** - Your GCS work could inspire Sudoku solving

The codebase is intentionally modular to support research extensions!
