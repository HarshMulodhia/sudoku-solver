"""Interactive Sudoku Solver with RL Agent"""

import argparse
import pygame
import numpy as np
import torch
import os
import sys
from typing import Optional

# Allow running directly: add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import ui_config
from sudoku_game import SudokuGame
from rl_agent import SudokuRLAgent
from pygame_ui import SudokuUI

class InteractiveSudokuSolver:
    """Interactive Sudoku solver with pygame UI and RL agent"""
    
    def __init__(self, difficulty: str = 'medium', mode: str = 'play', 
                 model_path: Optional[str] = None):
        """
        Initialize solver
        
        Args:
            difficulty: 'easy', 'medium', or 'hard'
            mode: 'play' or 'solve'
            model_path: Path to trained model weights
        """
        self.game = SudokuGame(difficulty=difficulty)
        self.ui = SudokuUI(self.game)
        self.mode = mode
        self.difficulty = difficulty
        
        # Initialize agent
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = SudokuRLAgent(device=device)
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.agent.load_model(model_path)
        else:
            print("No trained model found. Using untrained network.")
        
        # Solver state
        self.solving = False
        self.solve_steps_list = []
        self.last_solved_cell = None
    
    def place_number(self, row: int, col: int, digit: int) -> bool:
        """Place a number on the board"""
        if self.game.place_digit(row, col, digit):
            self.ui.emit_particles(row, col, 8)
            return True
        return False
    
    def get_hint(self, row: int, col: int) -> Optional[int]:
        """Get hint for a cell"""
        hint = self.game.solve_hint(row, col)
        if hint:
            self.ui.emit_particles(row, col, 5)
        return hint
    
    def compute_next_solve_step(self) -> Optional[tuple]:
        """
        Compute next solve step using RL agent
        
        Returns:
            Tuple of (row, col, digit) or None if no valid moves
        """
        state = self.game.get_encoded_state()
        valid_actions = self.agent.get_valid_actions(self.game)
        
        if not valid_actions:
            return None
        
        # Use agent to select action (greedy, no exploration)
        row, col, digit = self.agent.select_action(state, valid_actions, training=False)
        
        return (row, col, digit)
    
    def execute_solve_step(self) -> bool:
        """Execute one solve step."""
        if self.game.is_complete():
            return False
        
        next_step = self.compute_next_solve_step()
        if not next_step:
            return False
        
        row, col, digit = next_step
        self.game.place_digit(row, col, digit)
        self.ui.emit_particles(row, col, 12)
        self.last_solved_cell = (row, col)
        self.solve_steps_list.append((row, col, digit))
        
        return True
    
    def auto_solve(self):
        """Automatically solve puzzle"""
        self.solving = True
        self.solve_steps_list = []
        
        while not self.game.is_complete():
            if not self.execute_solve_step():
                break
    
    def reset_board(self):
        """Reset board to original state"""
        self.game.reset()
        self.solving = False
        self.solve_steps_list = []
        self.last_solved_cell = None
        self.ui.selected_cell = None
    
    def get_status(self) -> str:
        """Get current game status string"""
        if self.game.is_complete():
            return "✓ SOLVED"
        
        filled = np.sum(self.game.board > 0)
        total = 81
        return f"{filled}/{total} filled"
    
    def run(self):
        """Main game loop"""
        running = True
        auto_solving = False
        frame_count = 0
        
        print(f"\nSudoku RL Solver - {self.difficulty.upper()} mode")
        print("Press SPACE to start auto-solve with RL agent")
        print("Press H for hints, R to reset, Q to quit\n")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEMOTION:
                    self.ui.handle_mouse_motion(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.ui.handle_mouse_click(event.pos)
                
                elif event.type == pygame.KEYDOWN:
                    # Number input
                    if event.key >= pygame.K_1 and event.key <= pygame.K_9:
                        if self.ui.selected_cell:
                            digit = event.key - pygame.K_0
                            row, col = self.ui.selected_cell
                            self.place_number(row, col, digit)
                    
                    # Delete/Clear
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        if self.ui.selected_cell:
                            row, col = self.ui.selected_cell
                            self.place_number(row, col, 0)
                    
                    # Reset
                    elif event.key == pygame.K_r:
                        self.reset_board()
                        print("Board reset")
                    
                    # Hint
                    elif event.key == pygame.K_h:
                        if self.ui.selected_cell:
                            row, col = self.ui.selected_cell
                            hint = self.get_hint(row, col)
                            if hint:
                                self.place_number(row, col, hint)
                                print(f"Hint: Place {hint} at ({row}, {col})")
                    
                    # Auto-solve
                    elif event.key == pygame.K_SPACE:
                        if not auto_solving:
                            auto_solving = True
                            print("Starting RL auto-solve...")
                            self.auto_solve()
                            auto_solving = False
                            if self.game.is_complete():
                                print(f"✓ Puzzle solved in {len(self.solve_steps_list)} steps!")
                    
                    # Quit
                    elif event.key == pygame.K_q:
                        running = False
            
            # Update UI
            self.ui.update()
            
            # Draw
            status = self.get_status()
            self.ui.draw(self.ui.screen, self.ui.get_fps(), status)
            
            # Maintain FPS
            self.ui.tick(ui_config.FPS)
            frame_count += 1
        
        self.ui.quit()
        print("Exiting...")

def main():
    parser = argparse.ArgumentParser(description='Interactive Sudoku RL Solver')
    parser.add_argument('--difficulty', type=str, default='medium',
                       choices=['easy', 'medium', 'hard'])
    parser.add_argument('--mode', type=str, default='play',
                       choices=['play', 'solve'],
                       help='play: manual + hints, solve: auto-solve demo')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model weights')
    
    args = parser.parse_args()
    
    # Auto-determine model path if not provided
    if args.model is None:
        model_path = f'models/sudoku_dqn_{args.difficulty}.pth'
    else:
        model_path = args.model
    
    solver = InteractiveSudokuSolver(difficulty=args.difficulty, mode=args.mode, model_path=model_path)
    
    solver.run()

if __name__ == '__main__':
    main()
