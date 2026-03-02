"""Tests for RL agent and DQN network."""

import numpy as np
import torch
import pytest

from rl_agent import DQNNetwork, ExperienceReplay, SudokuRLAgent
from sudoku_game import SudokuGame
from config import rl_config


class TestDQNNetwork:
    """Test the DQN neural network."""

    def test_output_shape(self):
        net = DQNNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
        )
        x = torch.randn(1, 9, 9, 10)
        out = net(x)
        assert out.shape == (1, 729)

    def test_batch_forward(self):
        net = DQNNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
        )
        x = torch.randn(4, 9, 9, 10)
        out = net(x)
        assert out.shape == (4, 729)

    def test_unbatched_input(self):
        net = DQNNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
        )
        x = torch.randn(9, 9, 10)
        out = net(x)
        assert out.shape == (1, 729)


class TestExperienceReplay:
    """Test experience replay buffer."""

    def test_push_and_len(self):
        buf = ExperienceReplay(capacity=100)
        state = np.zeros((9, 9, 10), dtype=np.float32)
        buf.push(state, 0, 1.0, state, False)
        assert len(buf) == 1

    def test_capacity_limit(self):
        buf = ExperienceReplay(capacity=5)
        state = np.zeros((9, 9, 10), dtype=np.float32)
        for i in range(10):
            buf.push(state, i, 1.0, state, False)
        assert len(buf) == 5

    def test_sample(self):
        buf = ExperienceReplay(capacity=100)
        state = np.zeros((9, 9, 10), dtype=np.float32)
        for i in range(10):
            buf.push(state, i, float(i), state, False)
        states, actions, rewards, next_states, dones = buf.sample(4)
        assert states.shape == (4, 9, 9, 10)
        assert actions.shape == (4,)
        assert rewards.shape == (4,)


class TestSudokuRLAgent:
    """Test the RL agent."""

    def test_agent_creation(self):
        agent = SudokuRLAgent(device="cpu")
        assert agent.device == torch.device("cpu")

    def test_get_valid_actions(self):
        agent = SudokuRLAgent(device="cpu")
        game = SudokuGame("easy")
        actions = agent.get_valid_actions(game)
        assert len(actions) > 0
        for cell_idx, digit in actions:
            assert 0 <= cell_idx < 81
            assert 1 <= digit <= 9

    def test_select_action(self):
        agent = SudokuRLAgent(device="cpu")
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        valid_actions = agent.get_valid_actions(game)
        row, col, digit = agent.select_action(state, valid_actions, training=False)
        assert 0 <= row < 9
        assert 0 <= col < 9
        assert 1 <= digit <= 9

    def test_select_action_deterministic(self):
        agent = SudokuRLAgent(device="cpu")
        agent.q_network.eval()
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        valid_actions = agent.get_valid_actions(game)
        a1 = agent.select_action(state, valid_actions, training=False)
        a2 = agent.select_action(state, valid_actions, training=False)
        assert a1 == a2

    def test_compute_reward_valid(self):
        agent = SudokuRLAgent(device="cpu")
        game = SudokuGame("easy")
        reward = agent.compute_reward(game, (0, 0), 1, True)
        assert reward > 0

    def test_compute_reward_invalid(self):
        agent = SudokuRLAgent(device="cpu")
        game = SudokuGame("easy")
        reward = agent.compute_reward(game, (0, 0), 1, False)
        assert reward < 0

    def test_correct_move_higher_reward_than_wrong(self):
        """Placing the correct digit should yield higher reward"""
        agent = SudokuRLAgent(device="cpu")
        game = SudokuGame("easy")
        # Find an empty cell and its solution digit
        empty = np.argwhere(game.board == 0)[0]
        row, col = int(empty[0]), int(empty[1])
        correct_digit = int(game.solution[row, col])
        wrong_digit = (correct_digit % 9) + 1  # different digit
        correct_reward = agent.compute_reward(game, (row, col), correct_digit, True)
        wrong_reward = agent.compute_reward(game, (row, col), wrong_digit, True)
        assert correct_reward > wrong_reward

    def test_remember_and_train(self):
        agent = SudokuRLAgent(device="cpu")
        state = np.zeros((9, 9, 10), dtype=np.float32)
        next_state = np.zeros((9, 9, 10), dtype=np.float32)
        # Fill buffer enough for one training step
        for i in range(rl_config.BATCH_SIZE + 1):
            agent.remember(state, (0, 0, 1), 1.0, next_state, False)
        loss = agent.train_step()
        assert loss > 0

    def test_save_and_load(self, tmp_path):
        agent = SudokuRLAgent(device="cpu")
        model_path = str(tmp_path / "test_model.pth")
        agent.save_model(model_path)
        agent2 = SudokuRLAgent(device="cpu")
        agent2.load_model(model_path)
        # Weights should match
        for p1, p2 in zip(
            agent.q_network.parameters(), agent2.q_network.parameters()
        ):
            assert torch.equal(p1, p2)
