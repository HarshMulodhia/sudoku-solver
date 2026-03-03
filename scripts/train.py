"""Training Script for Sudoku RL Agent"""

import argparse
import numpy as np
from tqdm import tqdm
import os
import sys
import torch

# Allow running directly: add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import rl_config, DIFFICULTY_LEVELS
from sudoku_game import SudokuGame
from rl_agent import SudokuRLAgent

def train_agent(episodes: int = 1000, difficulty: str = 'medium', device: str = 'cuda'):
    """
    Train the RL agent on Sudoku puzzles
    
    Args:
        episodes: Number of training episodes
        difficulty: 'easy', 'medium', or 'hard'
        device: 'cuda' or 'cpu'
    """
    agent = SudokuRLAgent(device=device)
    
    print(f"Training on {difficulty.upper()} puzzles for {episodes} episodes...")
    print(f"Device: {device}")
    print(f"Network: {sum(p.numel() for p in agent.q_network.parameters())} parameters\n")
    
    episode_rewards = []
    episode_steps = []
    episode_losses = []
    
    for episode in tqdm(range(episodes)):
        # Create new game
        game = SudokuGame(difficulty=difficulty)
        state = game.get_encoded_state()
        
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        
        for step in range(rl_config.MAX_STEPS):
            # Get valid actions
            valid_actions = agent.get_valid_actions(game)
            
            if not valid_actions:
                break  # No more moves possible
            
            # Select action
            row, col, digit = agent.select_action(state, valid_actions, training=True)
            
            # Take action
            was_valid = game.place_digit(row, col, digit)
            
            # Compute reward
            reward = agent.compute_reward(game, (row, col), digit, was_valid)
            episode_reward += reward
            
            # Undo wrong placements to keep board solvable
            is_correct = was_valid and game.solution[row, col] == digit
            if was_valid and not is_correct:
                game.board[row, col] = 0
            
            # Get next state
            next_state = game.get_encoded_state()
            done = game.is_complete() and game.is_solved()
            
            # Store experience
            agent.remember(state, (row, col, digit), reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss > 0:
                episode_loss += loss
                loss_count += 1
            
            state = next_state
            
            if done:
                break
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(step + 1)
        if loss_count > 0:
            episode_losses.append(episode_loss / loss_count)
        
        # Log progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_steps = np.mean(episode_steps[-50:])
            avg_loss = np.mean(episode_losses[-50:]) if episode_losses else 0
            
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f'models/sudoku_dqn_{difficulty}.pth'
    agent.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    return agent, episode_rewards, episode_steps

def main():
    parser = argparse.ArgumentParser(description='Train Sudoku RL Agent')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--difficulty', type=str, default='medium', choices=['easy', 'medium', 'hard'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    train_agent(episodes=args.episodes, difficulty=args.difficulty, device=args.device)

if __name__ == '__main__':
    main()
