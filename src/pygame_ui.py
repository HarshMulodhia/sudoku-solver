"""High-Tech UI with Enhanced 3x3 Box Visualization"""

import pygame
import numpy as np
from typing import Tuple, Optional, List
import math
import time as _time
from config import ui_config, game_config
from sudoku_game import SudokuGame


class Button:
    """Clickable UI button with hover and active states"""

    def __init__(self, x: int, y: int, w: int, h: int, text: str,
                 active: bool = False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.hovered = False
        self.active = active

    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)

    def update_hover(self, pos: Tuple[int, int]):
        self.hovered = self.rect.collidepoint(pos)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, colors):
        if self.active:
            bg = colors.COLOR_ACCENT
            fg = colors.COLOR_BG
        elif self.hovered:
            bg = colors.COLOR_CELL_HOVER
            fg = colors.COLOR_TEXT
        else:
            bg = colors.COLOR_PANEL_BG
            fg = colors.COLOR_TEXT
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, colors.COLOR_BORDER, self.rect, 1,
                         border_radius=6)
        label = font.render(self.text, True, fg)
        label_rect = label.get_rect(center=self.rect.center)
        surface.blit(label, label_rect)

class Particle:
    """Particle effect for animations"""
    
    def __init__(self, x: float, y: float, vx: float, vy: float, 
                 lifetime: int, color: Tuple[int, ...]):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.lifetime = lifetime
        self.age = 0
        self.color = color
    
    def update(self, dt: int):
        """Update particle position and age"""
        self.x += self.vx * dt / 1000.0
        self.y += self.vy * dt / 1000.0
        self.age += dt
    
    def is_alive(self) -> bool:
        """Check if particle is still alive"""
        return self.age < self.lifetime
    
    def get_alpha(self) -> float:
        """Get alpha (0-1) based on age"""
        return max(0, 1.0 - (self.age / self.lifetime))
    
    def draw(self, surface: pygame.Surface):
        """Draw particle with fading effect"""
        alpha = int(self.get_alpha() * 255)
        size = max(2, int(5 * self.get_alpha()))
        
        color_with_alpha = (*self.color[:3], alpha)
        temp_surface = pygame.Surface((size*2, size*2))
        temp_surface.set_colorkey((0, 0, 0))
        
        pygame.draw.circle(temp_surface, self.color, (size, size), size)
        temp_surface.set_alpha(alpha)
        
        surface.blit(temp_surface, (int(self.x - size), int(self.y - size)))


class AnimationState:
    """Manages animation states and transitions"""
    
    def __init__(self, duration: int = ui_config.ANIMATION_DURATION):
        self.duration = duration
        self.progress = 0.0
        self.active = False
        self.start_time = 0
    
    def start(self):
        """Start animation"""
        self.active = True
        self.progress = 0.0
        self.start_time = pygame.time.get_ticks()
    
    def update(self):
        """Update animation progress"""
        if not self.active:
            return
        
        elapsed = pygame.time.get_ticks() - self.start_time
        self.progress = min(1.0, elapsed / self.duration)
        
        if self.progress >= 1.0:
            self.active = False
    
    def ease_in_out(self) -> float:
        """Easing function (ease in-out cubic)"""
        t = self.progress
        return t * t * (3 - 2 * t)


class SudokuUI:
    """High-tech Sudoku UI with enhanced 3x3 box visualization"""
    
    def __init__(self, game: SudokuGame):
        """
        Initialize UI
        
        Args:
            game: SudokuGame instance
        """
        self.game = game
        self.width = ui_config.WINDOW_WIDTH
        self.height = ui_config.WINDOW_HEIGHT
        self.cell_size = ui_config.CELL_SIZE
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Sudoku RL Solver - Neural Network Powered")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, ui_config.FONT_SIZE_CELL)
        self.font_normal = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, ui_config.FONT_SIZE_BUTTON)
        
        # Text cache for optimization
        self.text_cache = {}
        
        # Active color theme
        self.colors = ui_config.get_theme()
        
        # UI State
        self.selected_cell: Optional[Tuple[int, int]] = None
        self.hover_cell: Optional[Tuple[int, int]] = None
        self.particles: List[Particle] = []
        self.animations = {
            'cell_fill': AnimationState(),
            'solution_complete': AnimationState(800)
        }
        self.glow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Board position
        self.board_x = ui_config.BOARD_X
        self.board_y = ui_config.BOARD_Y
        
        # NEW: 3x3 box configuration
        self.box_size = 3 * self.cell_size
        self.box_border_width = 4
        self.box_highlight_alpha = 20  # Subtle background

        # Mode and timer state
        self.mode = 'manual'  # 'manual', 'rl', 'backtracking'
        self.start_time = _time.time()
        self.score = 0

        # Undo stack: list of (row, col, previous_digit) tuples
        self.undo_stack = []

        # Create buttons
        self._init_buttons()

    def _init_buttons(self):
        """Create all UI buttons"""
        panel_x = self.board_x + 9 * self.cell_size + 50
        btn_w = 135
        btn_h = 32
        gap = 8

        # Theme toggle
        y = self.board_y + 130
        self.btn_theme = Button(panel_x, y, btn_w * 2 + gap, btn_h,
                                "Toggle Dark/Light")

        # Difficulty buttons
        y += btn_h + 20
        third_w = (btn_w * 2 + gap - 2 * gap) // 3
        self.btn_easy = Button(panel_x, y, third_w, btn_h,
                               "Easy",
                               active=(self.game.difficulty == 'easy'))
        self.btn_medium = Button(
            panel_x + third_w + gap, y, third_w, btn_h,
            "Medium",
            active=(self.game.difficulty == 'medium'))
        self.btn_hard = Button(
            panel_x + 2 * (third_w + gap), y, third_w, btn_h,
            "Hard",
            active=(self.game.difficulty == 'hard'))

        # Mode buttons
        y += btn_h + 20
        self.btn_manual = Button(panel_x, y, btn_w, btn_h,
                                 "Manual Play", active=True)
        self.btn_rl = Button(panel_x + btn_w + gap, y, btn_w, btn_h,
                             "RL Solver")
        y += btn_h + gap
        self.btn_backtrack = Button(panel_x, y, btn_w * 2 + gap, btn_h,
                                    "Backtracking Solver")

        # Reset / undo buttons
        y += btn_h + 20
        self.btn_new_puzzle = Button(panel_x, y, btn_w, btn_h,
                                     "New Puzzle")
        self.btn_reset_entries = Button(panel_x + btn_w + gap, y, btn_w, btn_h,
                                        "Reset Entries")
        y += btn_h + gap
        self.btn_undo = Button(panel_x, y, btn_w * 2 + gap, btn_h,
                               "Undo (Ctrl+Z)")

        self.buttons = [
            self.btn_theme,
            self.btn_easy, self.btn_medium, self.btn_hard,
            self.btn_manual, self.btn_rl, self.btn_backtrack,
            self.btn_new_puzzle, self.btn_reset_entries,
            self.btn_undo,
        ]

        self.mode_buttons = [self.btn_manual, self.btn_rl, self.btn_backtrack]
        self.difficulty_buttons = [self.btn_easy, self.btn_medium,
                                   self.btn_hard]
    
    def set_mode(self, mode: str):
        """Set the active solver mode"""
        self.mode = mode
        for btn in self.mode_buttons:
            btn.active = False
        if mode == 'manual':
            self.btn_manual.active = True
        elif mode == 'rl':
            self.btn_rl.active = True
        elif mode == 'backtracking':
            self.btn_backtrack.active = True

    def set_difficulty(self, difficulty: str):
        """Set the active difficulty and update button states"""
        for btn in self.difficulty_buttons:
            btn.active = False
        if difficulty == 'easy':
            self.btn_easy.active = True
        elif difficulty == 'medium':
            self.btn_medium.active = True
        elif difficulty == 'hard':
            self.btn_hard.active = True

    def reset_timer(self):
        """Reset the game timer"""
        self.start_time = _time.time()
        self.score = 0

    def get_elapsed(self) -> str:
        """Return elapsed time as mm:ss"""
        elapsed = int(_time.time() - self.start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes:02d}:{seconds:02d}"

    def compute_score(self) -> int:
        """Compute score as number of correctly placed digits"""
        score = 0
        for i in range(9):
            for j in range(9):
                if self.game.original_board[i, j] == 0 and self.game.board[i, j] != 0:
                    if self.game.board[i, j] == self.game.solution[i, j]:
                        score += 1
        self.score = score
        return score
    
    def toggle_theme(self):
        """Toggle between dark and light mode"""
        ui_config.dark_mode = not ui_config.dark_mode
        self.colors = ui_config.get_theme()
        self.text_cache.clear()
    
    def get_cell_from_pos(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Convert pixel position to cell coordinates"""
        if (x < self.board_x or x > self.board_x + 9 * self.cell_size or
            y < self.board_y or y > self.board_y + 9 * self.cell_size):
            return None
        
        col = (x - self.board_x) // self.cell_size
        row = (y - self.board_y) // self.cell_size
        
        if 0 <= row < 9 and 0 <= col < 9:
            return (row, col)
        return None
    
    def get_cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Get pygame Rect for a cell"""
        x = self.board_x + col * self.cell_size
        y = self.board_y + row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)
    
    def get_box_rect(self, box_row: int, box_col: int) -> pygame.Rect:
        """
        Get pygame Rect for a 3x3 box
        
        Args:
            box_row: Box row index (0-2)
            box_col: Box column index (0-2)
        """
        x = self.board_x + box_col * self.box_size
        y = self.board_y + box_row * self.box_size
        return pygame.Rect(x, y, self.box_size, self.box_size)
    
    def emit_particles(self, row: int, col: int, num_particles: int = 12):
        """Create particle effects for cell"""
        rect = self.get_cell_rect(row, col)
        cx, cy = rect.centerx, rect.centery
        
        for _ in range(num_particles):
            angle = (2 * math.pi * _) / num_particles
            speed = 200 + np.random.rand() * 100
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            
            particle = Particle(
                cx, cy, vx, vy,
                ui_config.PARTICLE_LIFETIME,
                self.colors.COLOR_SOLVED
            )
            self.particles.append(particle)
    
    def get_text_cached(self, text: str, font: pygame.font.Font, color: Tuple[int, ...]) -> pygame.Surface:
        """Get cached text surface or render and cache"""
        cache_key = (text, id(font), color)
        
        if cache_key not in self.text_cache:
            self.text_cache[cache_key] = font.render(text, True, color)
        
        return self.text_cache[cache_key]
    
    def draw_3x3_boxes(self, surface: pygame.Surface):
        """
        Draw enhanced 3x3 box outlines
        Creates clear visual separation between 3x3 regions
        """
        # Draw box backgrounds (subtle)
        for box_row in range(3):
            for box_col in range(3):
                box_rect = self.get_box_rect(box_row, box_col)
                
                # Subtle alternating background for readability
                if (box_row + box_col) % 2 == 0:
                    pygame.draw.rect(surface, self.colors.COLOR_BOX_ALT_BG, box_rect)
        
        # Draw thick borders around each 3x3 box
        for box_row in range(3):
            for box_col in range(3):
                box_rect = self.get_box_rect(box_row, box_col)
                # Thick border
                pygame.draw.rect(surface, self.colors.COLOR_BOX_BORDER, box_rect, self.box_border_width)
    
    def is_highlighted(self, row: int, col: int) -> bool:
        """
        Check if a cell should be highlighted based on the selected cell.
        A cell is highlighted if it shares the same row, column, or 3x3 box
        as the selected cell.
        """
        if self.selected_cell is None:
            return False
        sel_row, sel_col = self.selected_cell
        if row == sel_row or col == sel_col:
            return True
        if row // 3 == sel_row // 3 and col // 3 == sel_col // 3:
            return True
        return False

    def is_digit_highlighted(self, row: int, col: int) -> bool:
        """
        Check if a cell should be highlighted because it contains the
        same digit as the currently selected cell.
        """
        if self.selected_cell is None:
            return False
        sel_row, sel_col = self.selected_cell
        sel_digit = self.game.board[sel_row, sel_col]
        if sel_digit == 0:
            return False
        return self.game.board[row, col] == sel_digit

    def draw_cell(self, surface: pygame.Surface, row: int, col: int):
        """Draw a single Sudoku cell with styling"""
        rect = self.get_cell_rect(row, col)
        digit = self.game.board[row, col]
        is_given = self.game.original_board[row, col] != 0
        
        # Determine cell color
        if (row, col) == self.selected_cell:
            cell_color = self.colors.COLOR_CELL_SELECTED
            glow = True
        elif (row, col) == self.hover_cell:
            cell_color = self.colors.COLOR_CELL_HOVER
            glow = False
        elif self.is_digit_highlighted(row, col):
            cell_color = self.colors.COLOR_CELL_HIGHLIGHT
            glow = False
        elif self.is_highlighted(row, col):
            cell_color = self.colors.COLOR_CELL_HIGHLIGHT
            glow = False
        elif is_given:
            cell_color = self.colors.COLOR_GIVEN
            glow = False
        else:
            cell_color = self.colors.COLOR_CELL_DEFAULT
            glow = False
        
        # Draw cell background
        pygame.draw.rect(surface, cell_color, rect)
        
        # Draw glow effect
        if glow:
            pygame.draw.rect(surface, self.colors.COLOR_ACCENT, rect, 3)
        
        # Draw thin cell borders (grid lines)
        pygame.draw.rect(surface, self.colors.COLOR_BORDER, rect, ui_config.BORDER_WIDTH)
        
        # Draw digit - always visible
        if digit != 0:
            text_color = self.colors.COLOR_TEXT if is_given else self.colors.COLOR_SOLVED
            text_surface = self.get_text_cached(str(digit), self.font_large, text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            surface.blit(text_surface, text_rect)
    
    def draw_board(self, surface: pygame.Surface):
        """Draw entire Sudoku board with cells and boxes"""
        # Draw 3x3 boxes FIRST (background layer)
        self.draw_3x3_boxes(surface)
        
        # Draw all cells (middle layer)
        for row in range(9):
            for col in range(9):
                self.draw_cell(surface, row, col)
    
    def draw_ui_info(self, surface: pygame.Surface, fps: float, status: str = ""):
        """Draw UI information panel"""
        panel_height = 120
        panel_rect = pygame.Rect(self.board_x + 9 * self.cell_size + 50, 
                                 self.board_y, 300, panel_height)
        
        # Draw panel background with border
        pygame.draw.rect(surface, self.colors.COLOR_PANEL_BG, panel_rect)
        pygame.draw.rect(surface, self.colors.COLOR_ACCENT, panel_rect, 2)
        
        # Title
        title = self.get_text_cached("Sudoku Solver", self.font_normal, self.colors.COLOR_ACCENT)
        surface.blit(title, (panel_rect.x + 20, panel_rect.y + 10))
        
        # Info text
        difficulty_text = self.get_text_cached(
            f"Difficulty: {self.game.difficulty.upper()}", 
            self.font_small, self.colors.COLOR_TEXT
        )
        surface.blit(difficulty_text, (panel_rect.x + 20, panel_rect.y + 35))

        # Score and time (rendered fresh each frame)
        score = self.compute_score()
        empty = int(np.sum(self.game.original_board == 0))
        score_surf = self.font_small.render(
            f"Score: {score}/{empty}", True, self.colors.COLOR_SOLVED)
        surface.blit(score_surf, (panel_rect.x + 20, panel_rect.y + 55))

        time_surf = self.font_small.render(
            f"Time: {self.get_elapsed()}", True, self.colors.COLOR_TEXT)
        surface.blit(time_surf, (panel_rect.x + 160, panel_rect.y + 55))
        
        if status:
            status_text = self.get_text_cached(status, self.font_small, self.colors.COLOR_SOLVED)
            surface.blit(status_text, (panel_rect.x + 20, panel_rect.y + 75))

        # Mode label
        mode_label = f"Mode: {self.mode.replace('_', ' ').title()}"
        mode_surf = self.font_small.render(mode_label, True, self.colors.COLOR_TEXT)
        surface.blit(mode_surf, (panel_rect.x + 20, panel_rect.y + 95))

    def draw_buttons(self, surface: pygame.Surface):
        """Draw all UI buttons"""
        for btn in self.buttons:
            btn.draw(surface, self.font_small, self.colors)
    
    def draw_instructions(self, surface: pygame.Surface):
        """Draw control instructions"""
        instructions = [
            "CONTROLS:",
            "Click cell + type digit: Place number",
            "Delete/Backspace: Clear cell",
            "Ctrl+Z: Undo last move",
            "H: Get hint   Q: Quit",
            "SPACE: Run active solver",
        ]
        
        y_offset = self.board_y + 9 * self.cell_size + 30
        for i, instruction in enumerate(instructions):
            color = self.colors.COLOR_ACCENT if i == 0 else self.colors.COLOR_TEXT
            text = self.get_text_cached(instruction, self.font_small, color)
            surface.blit(text, (self.board_x, y_offset + i * 20))
    
    def update(self):
        """Update UI state"""
        for animation in self.animations.values():
            animation.update()
        
        # Update particles
        current_time = pygame.time.get_ticks()
        self.particles = [p for p in self.particles if p.is_alive()]
        
        for particle in self.particles:
            particle.update(16)  # ~60 FPS
    
    def draw(self, surface: pygame.Surface, fps: float = 60, status: str = ""):
        """Draw entire UI"""
        # Clear screen
        surface.fill(self.colors.COLOR_BG)
        
        # Draw board (with 3x3 boxes)
        self.draw_board(surface)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(surface)
        
        # Draw info panel
        self.draw_ui_info(surface, fps, status)
        
        # Draw instructions
        self.draw_instructions(surface)

        # Draw HUD
        self.draw_glows()
        surface.blit(self.glow_surf, (0, 0))

        # Update display
        pygame.display.flip()

    def draw_glow_rect(self, rect, color, strength=180, rings=4):
        # draw multiple expanding outlines with decreasing alpha
        for i in range(rings):
            a = max(0, strength - i * (strength // rings))
            r = rect.inflate(i * 10, i * 10)
            pygame.draw.rect(self.glow_surf, (*color, a), r, width=2, border_radius=6)

    def draw_glows(self):
        self.glow_surf.fill((0, 0, 0, 0))  # clear alpha surface

        # Glow around selected cell
        if self.selected_cell:
            r, c = self.selected_cell
            self.draw_glow_rect(self.get_cell_rect(r, c), self.colors.COLOR_ACCENT, strength=200)

        # Subtle hover glow
        if self.hover_cell:
            r, c = self.hover_cell
            self.draw_glow_rect(self.get_cell_rect(r, c), self.colors.COLOR_BOX_BORDER, strength=120, rings=3)

        # Optional: glow the whole 3x3 box you're in
        cell = self.hover_cell or self.selected_cell
        if cell:
            br, bc = cell[0] // 3, cell[1] // 3
            box_rect = self.get_box_rect(br, bc)
            self.draw_glow_rect(box_rect, self.colors.COLOR_BOX_BORDER, strength=90, rings=2)
    
    def handle_mouse_motion(self, pos: Tuple[int, int]):
        """Handle mouse motion"""
        self.hover_cell = self.get_cell_from_pos(pos[0], pos[1])
    
    def handle_mouse_click(self, pos: Tuple[int, int]):
        """Handle mouse click"""
        cell = self.get_cell_from_pos(pos[0], pos[1])
        if cell:
            self.selected_cell = cell
    
    def handle_key_press(self, key: int) -> Optional[int]:
        """
        Handle key press
        
        Returns:
            Digit (1-9) or None
        """
        if key >= pygame.K_1 and key <= pygame.K_9:
            return key - pygame.K_0
        
        return None
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.clock.get_fps()
    
    def tick(self, fps: int = 60):
        """Tick clock"""
        self.clock.tick(fps)
    
    def quit(self):
        """Clean up and quit"""
        pygame.quit()
