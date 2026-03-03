# rl_agent.py - Deep Q-Network Agent for Sudoku Solving

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple
from config import rl_config, reward_config
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
            conv_channels: List of conv layer channel sizes
        """
        super(DQNNetwork, self).__init__()
        
        if conv_channels is None:
            conv_channels = [16, 32, 64]
        
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
        
        # Decay epsilon
        self.epsilon = max(
            rl_config.EPSILON_END,
            self.epsilon * rl_config.EPSILON_DECAY
        )
        
        return loss.item()
    
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
