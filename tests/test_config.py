"""Tests for config module."""

from config import (
    GameConfig, RLConfig, UIConfig, RewardConfig,
    ThemeColors, DARK_THEME, LIGHT_THEME, DIFFICULTY_LEVELS
)


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
        assert cfg.HIDDEN_LAYERS == [256, 128, 64]
        assert cfg.CONV_CHANNELS == [16, 32, 64]

    def test_post_init_hidden_layers(self):
        cfg = RLConfig(HIDDEN_LAYERS=None)
        assert cfg.HIDDEN_LAYERS == [256, 128, 64]

    def test_post_init_conv_channels(self):
        cfg = RLConfig(CONV_CHANNELS=None)
        assert cfg.CONV_CHANNELS == [16, 32, 64]

    def test_custom_hidden_layers(self):
        cfg = RLConfig(HIDDEN_LAYERS=[64, 32, 16])
        assert cfg.HIDDEN_LAYERS == [64, 32, 16]

    def test_optimized_hyperparameters(self):
        cfg = RLConfig()
        assert cfg.LEARNING_RATE == 0.001
        assert cfg.GAMMA == 0.99
        assert cfg.EPSILON_DECAY == 0.995
        assert cfg.BATCH_SIZE == 128
        assert cfg.MEMORY_SIZE == 50000
        assert cfg.TARGET_UPDATE_FREQ == 100
        assert cfg.REWARD_CLIP == 250.0


class TestUIConfig:
    def test_defaults(self):
        cfg = UIConfig()
        assert cfg.WINDOW_WIDTH == 1200
        assert cfg.WINDOW_HEIGHT == 900
        assert cfg.FPS == 60

    def test_dark_mode_default(self):
        cfg = UIConfig()
        assert cfg.dark_mode is True

    def test_get_theme_dark(self):
        cfg = UIConfig(dark_mode=True)
        assert cfg.get_theme() == DARK_THEME

    def test_get_theme_light(self):
        cfg = UIConfig(dark_mode=False)
        assert cfg.get_theme() == LIGHT_THEME


class TestThemeColors:
    def test_dark_theme_has_all_colors(self):
        assert isinstance(DARK_THEME.COLOR_BG, tuple)
        assert isinstance(DARK_THEME.COLOR_CELL_DEFAULT, tuple)
        assert isinstance(DARK_THEME.COLOR_BOX_ALT_BG, tuple)

    def test_light_theme_has_all_colors(self):
        assert isinstance(LIGHT_THEME.COLOR_BG, tuple)
        assert isinstance(LIGHT_THEME.COLOR_CELL_DEFAULT, tuple)
        assert isinstance(LIGHT_THEME.COLOR_BOX_ALT_BG, tuple)

    def test_themes_differ(self):
        assert DARK_THEME.COLOR_BG != LIGHT_THEME.COLOR_BG
        assert DARK_THEME.COLOR_CELL_DEFAULT != LIGHT_THEME.COLOR_CELL_DEFAULT
        assert DARK_THEME.COLOR_TEXT != LIGHT_THEME.COLOR_TEXT


class TestRewardConfig:
    def test_valid_move_positive(self):
        cfg = RewardConfig()
        assert cfg.VALID_MOVE_REWARD > 0

    def test_invalid_move_negative(self):
        cfg = RewardConfig()
        assert cfg.INVALID_MOVE_PENALTY < 0

    def test_correct_move_reward_positive(self):
        cfg = RewardConfig()
        assert cfg.CORRECT_MOVE_REWARD > 0

    def test_wrong_move_penalty_negative(self):
        cfg = RewardConfig()
        assert cfg.WRONG_MOVE_PENALTY < 0

    def test_completion_reward_positive(self):
        cfg = RewardConfig()
        assert cfg.COMPLETION_REWARD > 0


class TestDifficultyLevels:
    def test_all_difficulties_present(self):
        assert "easy" in DIFFICULTY_LEVELS
        assert "medium" in DIFFICULTY_LEVELS
        assert "hard" in DIFFICULTY_LEVELS

    def test_ordering(self):
        assert DIFFICULTY_LEVELS["easy"] > DIFFICULTY_LEVELS["medium"]
        assert DIFFICULTY_LEVELS["medium"] > DIFFICULTY_LEVELS["hard"]
