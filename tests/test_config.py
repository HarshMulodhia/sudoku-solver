"""Tests for config module."""

from config import GameConfig, RLConfig, UIConfig, RewardConfig, DIFFICULTY_LEVELS


class TestGameConfig:
    def test_defaults(self):
        cfg = GameConfig()
        assert cfg.BOARD_SIZE == 9
        assert cfg.SUBGRID_SIZE == 3


class TestRLConfig:
    def test_defaults(self):
        cfg = RLConfig()
        assert cfg.INPUT_SHAPE == (9, 9, 10)
        assert cfg.OUTPUT_SIZE == 729
        assert cfg.HIDDEN_LAYERS == [256, 256, 128]

    def test_post_init_hidden_layers(self):
        cfg = RLConfig(HIDDEN_LAYERS=None)
        assert cfg.HIDDEN_LAYERS == [256, 256, 128]


class TestUIConfig:
    def test_defaults(self):
        cfg = UIConfig()
        assert cfg.WINDOW_WIDTH == 1200
        assert cfg.WINDOW_HEIGHT == 900
        assert cfg.FPS == 60


class TestRewardConfig:
    def test_valid_move_positive(self):
        cfg = RewardConfig()
        assert cfg.VALID_MOVE_REWARD > 0

    def test_invalid_move_negative(self):
        cfg = RewardConfig()
        assert cfg.INVALID_MOVE_PENALTY < 0


class TestDifficultyLevels:
    def test_all_difficulties_present(self):
        assert "easy" in DIFFICULTY_LEVELS
        assert "medium" in DIFFICULTY_LEVELS
        assert "hard" in DIFFICULTY_LEVELS

    def test_ordering(self):
        assert DIFFICULTY_LEVELS["easy"] > DIFFICULTY_LEVELS["medium"]
        assert DIFFICULTY_LEVELS["medium"] > DIFFICULTY_LEVELS["hard"]
