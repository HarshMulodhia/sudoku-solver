# rl_agent.py - Deep Q-Network Agent for Sudoku Solving

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple
from config import rl_config, ppo_config, reward_config
from sudoku_game import SudokuGame

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, input_shape: Tuple[int, ...], output_size: int,
                 hidden_layers: list, conv_channels: list = None):
        """
        Initialize DQN
        
        Args:
            input_shape: Shape of input tensor (9, 9, 10)
            output_size: Size of output Q-values (81*9)
            hidden_layers: List of hidden layer sizes
            conv_channels: List of conv layer channel sizes,
                defaults to rl_config.CONV_CHANNELS
        """
        super(DQNNetwork, self).__init__()
        
        if conv_channels is None:
            conv_channels = rl_config.CONV_CHANNELS
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(10, conv_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1],
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2],
                               kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc_input_size = conv_channels[2] * 9 * 9  # After conv layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc_out = nn.Linear(hidden_layers[2], output_size)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network
        
        Args:
            x: Input tensor of shape (batch_size, 9, 9, 10)
        
        Returns:
            Q-values tensor of shape (batch_size, output_size)
        """
        # Reshape for conv layers: (batch, 10, 9, 9)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        x = x.permute(0, 3, 1, 2)  # (batch, 10, 9, 9)
        
        # Convolutional feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Flatten and fully connected layers
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Output Q-values
        q_values = self.fc_out(x)
        
        return q_values


class ExperienceReplay:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store experience in buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample random batch from buffer
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


def auto_detect_device() -> str:
    """Auto-detect the best available device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class SudokuRLAgent:
    """Deep Q-Learning agent for Sudoku solving"""
    
    def __init__(self, device: str = None):
        """
        Initialize RL agent
        
        Args:
            device: 'cuda', 'mps', 'cpu', or None for auto-detection
        """
        if device is None:
            device = auto_detect_device()
        self.device = torch.device(device)
        
        # Q-networks
        self.q_network = DQNNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
            conv_channels=rl_config.CONV_CHANNELS,
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
            conv_channels=rl_config.CONV_CHANNELS,
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=rl_config.LEARNING_RATE
        )
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()
        
        # Experience replay
        self.memory = ExperienceReplay(rl_config.MEMORY_SIZE)
        
        # Training parameters
        self.epsilon = rl_config.EPSILON_START
        self.training_steps = 0
    
    def select_action(self, state: np.ndarray, valid_actions: list, 
                     training: bool = False) -> Tuple[int, int, int]:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Board state (9, 9, 10)
            valid_actions: List of valid (cell_idx, digit) actions
            training: Whether in training mode
        
        Returns:
            Tuple of (cell_row, cell_col, digit)
        """
        if not valid_actions:
            return None
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            cell_idx, digit = random.choice(valid_actions)
            row, col = cell_idx // 9, cell_idx % 9
            return row, col, digit
        
        # Exploit: choose best action
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()
        
        # Convert to (cell, digit) index space
        best_q = -np.inf
        best_action = None
        
        for cell_idx, digit in valid_actions:
            action_idx = cell_idx * 9 + (digit - 1)
            if q_values[action_idx] > best_q:
                best_q = q_values[action_idx]
                best_action = (cell_idx, digit)
        
        if best_action is None:
            best_action = random.choice(valid_actions)
        
        cell_idx, digit = best_action
        row, col = cell_idx // 9, cell_idx % 9
        return row, col, digit
    
    def compute_reward(self, game: SudokuGame, cell: Tuple[int, int], 
                      digit: int, was_valid: bool) -> float:
        """
        Compute reward for action
        
        Args:
            game: Current game state
            cell: Cell position (row, col)
            digit: Placed digit
            was_valid: Whether move was valid
        
        Returns:
            Reward value
        """
        if not was_valid:
            return reward_config.INVALID_MOVE_PENALTY

        reward = reward_config.VALID_MOVE_REWARD
        row, col = cell

        # Bonus for correct digit matching solution
        if game.solution[row, col] == digit:
            reward += reward_config.CORRECT_MOVE_REWARD
        else:
            reward += reward_config.WRONG_MOVE_PENALTY

        # Bonus for completion
        if game.is_complete() and game.is_solved():
            reward += reward_config.COMPLETION_REWARD

        # Check for conflicts
        conflicts = game.get_conflicts()
        if cell in conflicts:
            reward += reward_config.ROW_CONFLICT_PENALTY
        
        return reward
    
    def remember(self, state: np.ndarray, action: Tuple[int, int, int], 
                reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        row, col, digit = action
        action_idx = (row * 9 + col) * 9 + (digit - 1)
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def train_step(self) -> float:
        """
        Perform one training step
        
        Returns:
            Loss value
        """
        if len(self.memory) < rl_config.BATCH_SIZE:
            return 0.0
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(
            rl_config.BATCH_SIZE
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Clip rewards to prevent extreme values
        reward_clip = rl_config.REWARD_CLIP
        rewards = rewards.clamp(-reward_clip, reward_clip)
        
        # Compute Q-values for current states
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(
                1, next_actions
            ).squeeze(1)
            target_q_values = rewards + rl_config.GAMMA * next_q_values * (1 - dones)
            # Clip target Q-values to prevent runaway estimates
            target_q_values = target_q_values.clamp(
                -reward_clip * 10, reward_clip * 10
            )
        
        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update training parameters
        self.training_steps += 1
        if self.training_steps % rl_config.TARGET_UPDATE_FREQ == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon once per episode (not per training step)."""
        self.epsilon = max(
            rl_config.EPSILON_END,
            self.epsilon * rl_config.EPSILON_DECAY
        )
    
    def get_valid_actions(self, game: SudokuGame) -> list:
        """
        Get list of valid actions for current board state
        
        Returns:
            List of (cell_idx, digit) tuples
        """
        valid_actions = []
        
        for row in range(9):
            for col in range(9):
                if game.board[row, col] == 0:
                    cell_idx = row * 9 + col
                    possible = game.get_possible_values(row, col)
                    for digit in possible:
                        valid_actions.append((cell_idx, digit))
        
        return valid_actions
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save(self.q_network.state_dict(), path)
    
    def load_model(self, path: str):
        """Load trained model"""
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO agent"""

    def __init__(self, input_shape: Tuple[int, ...], output_size: int,
                 hidden_layers: list, conv_channels: list = None):
        """
        Initialize Actor-Critic network with shared CNN backbone.

        Args:
            input_shape: Shape of input tensor (9, 9, 10)
            output_size: Number of actions (81*9 = 729)
            hidden_layers: List of hidden layer sizes
            conv_channels: List of conv layer channel sizes
        """
        super(ActorCriticNetwork, self).__init__()

        if conv_channels is None:
            conv_channels = rl_config.CONV_CHANNELS

        # Shared convolutional backbone
        self.conv1 = nn.Conv2d(10, conv_channels[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1],
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2],
                               kernel_size=3, padding=1)

        self.fc_input_size = conv_channels[2] * 9 * 9

        # Actor head (policy)
        self.actor_fc1 = nn.Linear(self.fc_input_size, hidden_layers[0])
        self.actor_fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.actor_out = nn.Linear(hidden_layers[1], output_size)

        # Critic head (value)
        self.critic_fc1 = nn.Linear(self.fc_input_size, hidden_layers[0])
        self.critic_fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.critic_out = nn.Linear(hidden_layers[1], 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action logits and state value.

        Args:
            x: Input tensor of shape (batch_size, 9, 9, 10)

        Returns:
            Tuple of (action_logits, state_value)
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        x = x.permute(0, 3, 1, 2)  # (batch, 10, 9, 9)

        # Shared CNN features
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)

        # Actor head
        a = self.relu(self.actor_fc1(x))
        a = self.dropout(a)
        a = self.relu(self.actor_fc2(a))
        action_logits = self.actor_out(a)

        # Critic head
        v = self.relu(self.critic_fc1(x))
        v = self.dropout(v)
        v = self.relu(self.critic_fc2(v))
        state_value = self.critic_out(v)

        return action_logits, state_value


class RolloutBuffer:
    """Buffer for storing PPO rollout trajectories"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []

    def push(self, state: np.ndarray, action: int, log_prob: float,
             reward: float, value: float, done: bool,
             action_mask: np.ndarray):
        """Store one transition"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self):
        """Clear all stored data"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.states)


class SudokuPPOAgent:
    """PPO agent for Sudoku solving"""

    def __init__(self, device: str = None):
        """
        Initialize PPO agent.

        Args:
            device: 'cuda', 'mps', 'cpu', or None for auto-detection
        """
        if device is None:
            device = auto_detect_device()
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
            conv_channels=rl_config.CONV_CHANNELS,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=ppo_config.LEARNING_RATE
        )

        self.rollout = RolloutBuffer()

    def _build_action_mask(self, valid_actions: list) -> np.ndarray:
        """Build a binary mask over the 729 action space."""
        mask = np.zeros(rl_config.OUTPUT_SIZE, dtype=np.float32)
        for cell_idx, digit in valid_actions:
            action_idx = cell_idx * 9 + (digit - 1)
            mask[action_idx] = 1.0
        return mask

    def select_action(self, state: np.ndarray, valid_actions: list,
                      training: bool = False) -> Tuple[int, int, int]:
        """
        Select action using the current policy.

        Args:
            state: Board state (9, 9, 10)
            valid_actions: List of valid (cell_idx, digit) actions
            training: Whether in training mode

        Returns:
            Tuple of (cell_row, cell_col, digit)
        """
        if not valid_actions:
            return None

        action_mask = self._build_action_mask(valid_actions)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32
                                   ).to(self.device)

        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float32
            ).to(self.device)
            logits, _ = self.network(state_tensor)
            logits = logits.squeeze(0)

            # Mask invalid actions with large negative value
            logits = logits + (mask_tensor + 1e-8).log()

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if training:
                action = dist.sample()
            else:
                action = probs.argmax()

        action_idx = action.item()
        cell_idx = action_idx // 9
        digit = (action_idx % 9) + 1
        row, col = cell_idx // 9, cell_idx % 9
        return row, col, digit

    def select_action_with_log_prob(
        self, state: np.ndarray, valid_actions: list
    ) -> Tuple[Tuple[int, int, int], int, float, float]:
        """
        Select action and return log probability and value for training.

        Returns:
            Tuple of ((row, col, digit), action_idx, log_prob, value)
        """
        if not valid_actions:
            return None, -1, 0.0, 0.0

        action_mask = self._build_action_mask(valid_actions)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32
                                   ).to(self.device)

        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).to(self.device)
        logits, value = self.network(state_tensor)
        logits = logits.squeeze(0)
        value = value.squeeze()

        # Mask invalid actions
        logits = logits + (mask_tensor + 1e-8).log()

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action_idx = action.item()
        cell_idx = action_idx // 9
        digit = (action_idx % 9) + 1
        row, col = cell_idx // 9, cell_idx % 9

        return (
            (row, col, digit), action_idx,
            log_prob.item(), value.item()
        )

    def compute_reward(self, game: SudokuGame, cell: Tuple[int, int],
                       digit: int, was_valid: bool) -> float:
        """Compute reward for action (same logic as DQN agent)."""
        if not was_valid:
            return reward_config.INVALID_MOVE_PENALTY

        reward = reward_config.VALID_MOVE_REWARD
        row, col = cell

        if game.solution[row, col] == digit:
            reward += reward_config.CORRECT_MOVE_REWARD
        else:
            reward += reward_config.WRONG_MOVE_PENALTY

        if game.is_complete() and game.is_solved():
            reward += reward_config.COMPLETION_REWARD

        conflicts = game.get_conflicts()
        if cell in conflicts:
            reward += reward_config.ROW_CONFLICT_PENALTY

        return reward

    def remember(self, state: np.ndarray, action_idx: int, log_prob: float,
                 reward: float, value: float, done: bool,
                 action_mask: np.ndarray):
        """Store transition in rollout buffer."""
        self.rollout.push(
            state, action_idx, log_prob, reward, value, done, action_mask
        )

    def compute_gae(self, next_value: float) -> Tuple[list, list]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            next_value: Value estimate for the state after the rollout

        Returns:
            Tuple of (advantages, returns)
        """
        gamma = ppo_config.GAMMA
        lam = ppo_config.GAE_LAMBDA
        rewards = self.rollout.rewards
        values = self.rollout.values
        dones = self.rollout.dones

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def train_step(self) -> float:
        """
        Perform PPO update on collected rollout data.

        Returns:
            Average loss over all PPO epochs
        """
        if len(self.rollout) == 0:
            return 0.0

        # Get last value for GAE
        last_state = self.rollout.states[-1]
        with torch.no_grad():
            state_t = torch.tensor(
                last_state, dtype=torch.float32
            ).to(self.device)
            _, last_value = self.network(state_t)
            next_value = last_value.squeeze().item()

        if self.rollout.dones[-1]:
            next_value = 0.0

        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        states = torch.tensor(
            np.array(self.rollout.states), dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            self.rollout.actions, dtype=torch.long
        ).to(self.device)
        old_log_probs = torch.tensor(
            self.rollout.log_probs, dtype=torch.float32
        ).to(self.device)
        advantages_t = torch.tensor(
            advantages, dtype=torch.float32
        ).to(self.device)
        returns_t = torch.tensor(
            returns, dtype=torch.float32
        ).to(self.device)
        action_masks = torch.tensor(
            np.array(self.rollout.action_masks), dtype=torch.float32
        ).to(self.device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (
                (advantages_t - advantages_t.mean())
                / (advantages_t.std() + 1e-8)
            )

        total_loss = 0.0
        num_updates = 0
        n = len(self.rollout)

        for _ in range(ppo_config.PPO_EPOCHS):
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, ppo_config.MINI_BATCH_SIZE):
                end = min(start + ppo_config.MINI_BATCH_SIZE, n)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_masks = action_masks[mb_idx]

                logits, values = self.network(mb_states)
                values = values.squeeze(-1)

                # Mask invalid actions
                logits = logits + (mb_masks + 1e-8).log()

                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - ppo_config.CLIP_EPSILON,
                    1.0 + ppo_config.CLIP_EPSILON,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Total loss
                loss = (
                    policy_loss
                    + ppo_config.VALUE_LOSS_COEF * value_loss
                    - ppo_config.ENTROPY_COEF * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    ppo_config.MAX_GRAD_NORM
                )
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        self.rollout.clear()
        return total_loss / max(num_updates, 1)

    def get_valid_actions(self, game: SudokuGame) -> list:
        """Get list of valid actions for current board state."""
        valid_actions = []
        for row in range(9):
            for col in range(9):
                if game.board[row, col] == 0:
                    cell_idx = row * 9 + col
                    possible = game.get_possible_values(row, col)
                    for digit in possible:
                        valid_actions.append((cell_idx, digit))
        return valid_actions

    def save_model(self, path: str):
        """Save trained model"""
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str):
        """Load trained model"""
        self.network.load_state_dict(
            torch.load(path, map_location=self.device)
        )
