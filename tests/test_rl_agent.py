"""Tests for RL agent and DQN network."""

import numpy as np
import torch
import pytest

from rl_agent import DQNNetwork, ExperienceReplay, SudokuRLAgent, auto_detect_device
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

    def test_compute_reward_valid_correct(self):
        agent = SudokuRLAgent(device="cpu")
        game = SudokuGame("easy")
        # Find an empty cell and use the correct digit
        empty = np.argwhere(game.board == 0)[0]
        row, col = int(empty[0]), int(empty[1])
        correct_digit = int(game.solution[row, col])
        reward = agent.compute_reward(game, (row, col), correct_digit, True)
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

    def test_wrong_valid_move_negative_reward(self):
        """A valid move with the wrong digit should get negative reward."""
        agent = SudokuRLAgent(device="cpu")
        game = SudokuGame("easy")
        empty = np.argwhere(game.board == 0)[0]
        row, col = int(empty[0]), int(empty[1])
        correct_digit = int(game.solution[row, col])
        wrong_digit = (correct_digit % 9) + 1
        reward = agent.compute_reward(game, (row, col), wrong_digit, True)
        assert reward < 0

    def test_double_dqn_train_step(self):
        """Training step should use Double DQN and produce a positive loss."""
        agent = SudokuRLAgent(device="cpu")
        state = np.random.rand(9, 9, 10).astype(np.float32)
        next_state = np.random.rand(9, 9, 10).astype(np.float32)
        for _ in range(rl_config.BATCH_SIZE + 1):
            agent.remember(state, (0, 0, 1), 1.0, next_state, False)
        loss = agent.train_step()
        assert loss > 0

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

    def test_reduced_network_size(self):
        """Network should have fewer than 2M parameters."""
        agent = SudokuRLAgent(device="cpu")
        param_count = sum(
            p.numel() for p in agent.q_network.parameters()
        )
        assert param_count < 2_000_000

    def test_auto_detect_device(self):
        """auto_detect_device should return a valid device string."""
        device = auto_detect_device()
        assert device in ('cuda', 'mps', 'cpu')

    def test_agent_auto_device(self):
        """Agent with device=None should auto-detect."""
        agent = SudokuRLAgent(device=None)
        assert agent.device.type in ('cuda', 'mps', 'cpu')

    def test_conv_channels_configurable(self):
        """DQNNetwork should accept custom conv_channels."""
        net = DQNNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=[128, 64, 32],
            conv_channels=[8, 16, 32],
        )
        x = torch.randn(1, 9, 9, 10)
        out = net(x)
        assert out.shape == (1, 729)

    def test_training_device_consistency(self):
        """All tensors should be on the same device during training."""
        device_str = "cpu"
        agent = SudokuRLAgent(device=device_str)

        # Verify networks are on the correct device
        for p in agent.q_network.parameters():
            assert p.device.type == device_str
        for p in agent.target_network.parameters():
            assert p.device.type == device_str

        # Fill replay buffer and run a training step
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        valid_actions = agent.get_valid_actions(game)
        row, col, digit = agent.select_action(
            state, valid_actions, training=True
        )
        was_valid = game.place_digit(row, col, digit)
        reward = agent.compute_reward(
            game, (row, col), digit, was_valid
        )
        next_state = game.get_encoded_state()

        for _ in range(rl_config.BATCH_SIZE + 1):
            agent.remember(
                state, (row, col, digit), reward, next_state, False
            )

        loss = agent.train_step()
        assert loss > 0

        # Verify weights are still on the correct device after training
        for p in agent.q_network.parameters():
            assert p.device.type == device_str

    def test_decay_epsilon_per_episode(self):
        """decay_epsilon should reduce epsilon once per call."""
        agent = SudokuRLAgent(device="cpu")
        agent.epsilon = 1.0
        agent.decay_epsilon()
        assert agent.epsilon == 1.0 * rl_config.EPSILON_DECAY

    def test_decay_epsilon_respects_minimum(self):
        """decay_epsilon should not go below EPSILON_END."""
        agent = SudokuRLAgent(device="cpu")
        agent.epsilon = rl_config.EPSILON_END
        agent.decay_epsilon()
        assert agent.epsilon == rl_config.EPSILON_END

    def test_train_step_does_not_decay_epsilon(self):
        """train_step should not change epsilon (decay is per-episode)."""
        agent = SudokuRLAgent(device="cpu")
        state = np.random.rand(9, 9, 10).astype(np.float32)
        next_state = np.random.rand(9, 9, 10).astype(np.float32)
        for _ in range(rl_config.BATCH_SIZE + 1):
            agent.remember(state, (0, 0, 1), 1.0, next_state, False)
        eps_before = agent.epsilon
        agent.train_step()
        assert agent.epsilon == eps_before

    def test_reward_clipping_in_train_step(self):
        """Rewards should be clipped during training to prevent divergence."""
        agent = SudokuRLAgent(device="cpu")
        state = np.random.rand(9, 9, 10).astype(np.float32)
        next_state = np.random.rand(9, 9, 10).astype(np.float32)
        # Store experiences with extreme reward values
        for _ in range(rl_config.BATCH_SIZE + 1):
            agent.remember(state, (0, 0, 1), 99999.0, next_state, False)
        loss = agent.train_step()
        # Loss should be finite (not NaN/Inf) thanks to clipping
        assert loss > 0
        assert not np.isnan(loss)
        assert not np.isinf(loss)
