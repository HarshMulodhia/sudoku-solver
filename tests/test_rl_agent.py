"""Tests for RL agent and DQN network."""

import numpy as np
import torch
import pytest

from rl_agent import (
    DQNNetwork, ExperienceReplay, SudokuRLAgent, auto_detect_device,
    ActorCriticNetwork, RolloutBuffer, SudokuPPOAgent,
)
from sudoku_game import SudokuGame
from config import rl_config, ppo_config


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


class TestActorCriticNetwork:
    """Test the Actor-Critic network for PPO."""

    def test_output_shapes(self):
        net = ActorCriticNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
        )
        x = torch.randn(1, 9, 9, 10)
        logits, value = net(x)
        assert logits.shape == (1, 729)
        assert value.shape == (1, 1)

    def test_batch_forward(self):
        net = ActorCriticNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
        )
        x = torch.randn(4, 9, 9, 10)
        logits, value = net(x)
        assert logits.shape == (4, 729)
        assert value.shape == (4, 1)

    def test_unbatched_input(self):
        net = ActorCriticNetwork(
            input_shape=rl_config.INPUT_SHAPE,
            output_size=rl_config.OUTPUT_SIZE,
            hidden_layers=rl_config.HIDDEN_LAYERS,
        )
        x = torch.randn(9, 9, 10)
        logits, value = net(x)
        assert logits.shape == (1, 729)
        assert value.shape == (1, 1)


class TestRolloutBuffer:
    """Test rollout buffer for PPO."""

    def test_push_and_len(self):
        buf = RolloutBuffer()
        state = np.zeros((9, 9, 10), dtype=np.float32)
        mask = np.ones(729, dtype=np.float32)
        buf.push(state, 0, -0.5, 1.0, 0.5, False, mask)
        assert len(buf) == 1

    def test_clear(self):
        buf = RolloutBuffer()
        state = np.zeros((9, 9, 10), dtype=np.float32)
        mask = np.ones(729, dtype=np.float32)
        for _ in range(5):
            buf.push(state, 0, -0.5, 1.0, 0.5, False, mask)
        assert len(buf) == 5
        buf.clear()
        assert len(buf) == 0


class TestSudokuPPOAgent:
    """Test the PPO agent."""

    def test_agent_creation(self):
        agent = SudokuPPOAgent(device="cpu")
        assert agent.device == torch.device("cpu")

    def test_get_valid_actions(self):
        agent = SudokuPPOAgent(device="cpu")
        game = SudokuGame("easy")
        actions = agent.get_valid_actions(game)
        assert len(actions) > 0
        for cell_idx, digit in actions:
            assert 0 <= cell_idx < 81
            assert 1 <= digit <= 9

    def test_select_action(self):
        agent = SudokuPPOAgent(device="cpu")
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        valid_actions = agent.get_valid_actions(game)
        row, col, digit = agent.select_action(
            state, valid_actions, training=False
        )
        assert 0 <= row < 9
        assert 0 <= col < 9
        assert 1 <= digit <= 9

    def test_select_action_deterministic(self):
        agent = SudokuPPOAgent(device="cpu")
        agent.network.eval()
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        valid_actions = agent.get_valid_actions(game)
        a1 = agent.select_action(state, valid_actions, training=False)
        a2 = agent.select_action(state, valid_actions, training=False)
        assert a1 == a2

    def test_select_action_with_log_prob(self):
        agent = SudokuPPOAgent(device="cpu")
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        valid_actions = agent.get_valid_actions(game)
        action, action_idx, log_prob, value = \
            agent.select_action_with_log_prob(state, valid_actions)
        row, col, digit = action
        assert 0 <= row < 9
        assert 0 <= col < 9
        assert 1 <= digit <= 9
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert log_prob <= 0  # log probs are non-positive

    def test_compute_reward_valid_correct(self):
        agent = SudokuPPOAgent(device="cpu")
        game = SudokuGame("easy")
        empty = np.argwhere(game.board == 0)[0]
        row, col = int(empty[0]), int(empty[1])
        correct_digit = int(game.solution[row, col])
        reward = agent.compute_reward(
            game, (row, col), correct_digit, True
        )
        assert reward > 0

    def test_compute_reward_invalid(self):
        agent = SudokuPPOAgent(device="cpu")
        game = SudokuGame("easy")
        reward = agent.compute_reward(game, (0, 0), 1, False)
        assert reward < 0

    def test_build_action_mask(self):
        agent = SudokuPPOAgent(device="cpu")
        valid_actions = [(0, 1), (0, 5), (40, 9)]
        mask = agent._build_action_mask(valid_actions)
        assert mask.shape == (729,)
        assert mask[0 * 9 + 0] == 1.0  # cell 0, digit 1
        assert mask[0 * 9 + 4] == 1.0  # cell 0, digit 5
        assert mask[40 * 9 + 8] == 1.0  # cell 40, digit 9
        assert mask.sum() == 3

    def test_ppo_train_step(self):
        """PPO train step should produce a positive loss."""
        agent = SudokuPPOAgent(device="cpu")
        game = SudokuGame("easy")
        state = game.get_encoded_state()
        valid_actions = agent.get_valid_actions(game)

        # Collect a few transitions
        for _ in range(10):
            action, action_idx, log_prob, value = \
                agent.select_action_with_log_prob(state, valid_actions)
            row, col, digit = action
            was_valid = game.place_digit(row, col, digit)
            reward = agent.compute_reward(
                game, (row, col), digit, was_valid
            )
            if was_valid and game.solution[row, col] != digit:
                game.board[row, col] = 0

            mask = agent._build_action_mask(valid_actions)
            agent.remember(state, action_idx, log_prob, reward, value,
                           False, mask)
            state = game.get_encoded_state()
            valid_actions = agent.get_valid_actions(game)
            if not valid_actions:
                break

        loss = agent.train_step()
        assert loss > 0
        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_save_and_load(self, tmp_path):
        agent = SudokuPPOAgent(device="cpu")
        model_path = str(tmp_path / "test_ppo_model.pth")
        agent.save_model(model_path)
        agent2 = SudokuPPOAgent(device="cpu")
        agent2.load_model(model_path)
        for p1, p2 in zip(
            agent.network.parameters(), agent2.network.parameters()
        ):
            assert torch.equal(p1, p2)

    def test_compute_gae(self):
        """GAE computation should return correct-length lists."""
        agent = SudokuPPOAgent(device="cpu")
        state = np.zeros((9, 9, 10), dtype=np.float32)
        mask = np.ones(729, dtype=np.float32)
        for i in range(5):
            agent.remember(state, 0, -1.0, float(i), 0.5,
                           i == 4, mask)
        advantages, returns = agent.compute_gae(next_value=0.0)
        assert len(advantages) == 5
        assert len(returns) == 5
