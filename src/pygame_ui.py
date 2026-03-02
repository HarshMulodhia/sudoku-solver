"""High-Tech UI with Enhanced 3x3 Box Visualization"""

import pygame
import numpy as np
from typing import Tuple, Optional, List
import math
from config import ui_config, game_config
from sudoku_game import SudokuGame

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
        title = self.get_text_cached("RL Solver", self.font_normal, self.colors.COLOR_ACCENT)
        surface.blit(title, (panel_rect.x + 20, panel_rect.y + 10))
        
        # Info text
        difficulty_text = self.get_text_cached(
            f"Difficulty: {self.game.difficulty.upper()}", 
            self.font_small, self.colors.COLOR_TEXT
        )
        surface.blit(difficulty_text, (panel_rect.x + 20, panel_rect.y + 35))
        
        fps_text = self.get_text_cached(f"FPS: {fps:.1f}", self.font_small, self.colors.COLOR_TEXT)
        surface.blit(fps_text, (panel_rect.x + 20, panel_rect.y + 55))
        
        if status:
            status_text = self.get_text_cached(status, self.font_small, self.colors.COLOR_SOLVED)
            surface.blit(status_text, (panel_rect.x + 20, panel_rect.y + 75))

        # Theme mode indicator
        mode_label = "Dark Mode" if ui_config.dark_mode else "Light Mode"
        mode_text = self.get_text_cached(mode_label, self.font_small, self.colors.COLOR_TEXT)
        surface.blit(mode_text, (panel_rect.x + 20, panel_rect.y + 95))
    
    def draw_instructions(self, surface: pygame.Surface):
        """Draw control instructions"""
        instructions = [
            "CONTROLS:",
            "Click cell + type digit: Place number",
            "R: Reset board",
            "H: Get hint",
            "T: Toggle dark/light mode",
            "SPACE: Auto-solve (RL Agent)",
            "Q: Quit"
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
