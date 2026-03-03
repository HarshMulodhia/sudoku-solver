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
from backtracking_solver import BacktrackingSolver
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
        
        # Initialize RL agent
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = SudokuRLAgent(device=device)
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.agent.load_model(model_path)
        else:
            print("No trained model found. Using untrained network.")

        # Backtracking solver
        self.bt_solver = BacktrackingSolver()
        
        # Solver state
        self.solving = False
        self.solve_steps_list = []
        self.last_solved_cell = None
    
    def place_number(self, row: int, col: int, digit: int,
                     force: bool = False) -> bool:
        """Place a number on the board and record undo history"""
        previous = int(self.game.board[row, col])
        if self.game.place_digit(row, col, digit, force=force):
            self.ui.undo_stack.append((row, col, previous))
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
    
    def auto_solve_rl(self):
        """Automatically solve puzzle with RL agent"""
        self.solving = True
        self.solve_steps_list = []
        
        while not self.game.is_complete():
            if not self.execute_solve_step():
                break

    def auto_solve_backtracking(self):
        """Automatically solve puzzle with backtracking solver"""
        self.solving = True
        self.solve_steps_list = []
        solver = BacktrackingSolver()
        solver.solve(self.game)
    
    def reset_board(self):
        """Reset board to original state (clears manual/solver entries)"""
        self.game.reset()
        self.solving = False
        self.solve_steps_list = []
        self.last_solved_cell = None
        self.ui.selected_cell = None
        self.ui.undo_stack.clear()

    def new_puzzle(self):
        """Generate a completely new puzzle"""
        self.game = SudokuGame(difficulty=self.difficulty)
        self.ui.game = self.game
        self.solving = False
        self.solve_steps_list = []
        self.last_solved_cell = None
        self.ui.selected_cell = None
        self.ui.undo_stack.clear()
        self.ui.reset_timer()

    def change_difficulty(self, difficulty: str):
        """Change difficulty and generate a new puzzle"""
        self.difficulty = difficulty
        self.new_puzzle()

    def undo_last_move(self) -> bool:
        """Undo the last digit placement.

        Returns:
            True if a move was undone, False if nothing to undo.
        """
        if not self.ui.undo_stack:
            return False
        row, col, prev_digit = self.ui.undo_stack.pop()
        self.game.board[row, col] = prev_digit
        return True

    def _try_undo(self):
        """Attempt undo and log result."""
        if self.undo_last_move():
            print("Move undone")
    
    def get_status(self) -> str:
        """Get current game status string"""
        if self.game.is_complete():
            if self.game.is_solved():
                return "✓ SOLVED"
            return "Complete but has errors"
        
        filled = np.sum(self.game.board > 0)
        total = 81
        return f"{filled}/{total} filled"
    
    def run(self):
        """Main game loop"""
        running = True
        auto_solving = False
        frame_count = 0
        
        print(f"\nSudoku Solver - {self.difficulty.upper()} mode")
        print("Use buttons to switch mode and theme")
        print("Press SPACE to run the active solver, H for hints, Q to quit\n")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEMOTION:
                    self.ui.handle_mouse_motion(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    action = self.ui.handle_mouse_click(event.pos)
                    if action == 'new_puzzle':
                        self.new_puzzle()
                        print("New puzzle generated")
                    elif action == 'reset_entries':
                        self.reset_board()
                        print("Entries reset")
                    elif action == 'undo':
                        self._try_undo()
                    elif action in ('difficulty_easy',
                                    'difficulty_medium',
                                    'difficulty_hard'):
                        diff = action.split('_')[1]
                        self.change_difficulty(diff)
                        print(f"Difficulty changed to {diff}")
                
                elif event.type == pygame.KEYDOWN:
                    # Undo (Ctrl+Z)
                    ctrl = event.mod & pygame.KMOD_CTRL
                    if event.key == pygame.K_z and ctrl:
                        self._try_undo()

                    # Number input (manual mode uses force=True)
                    elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
                        if self.ui.selected_cell:
                            digit = event.key - pygame.K_0
                            row, col = self.ui.selected_cell
                            is_manual = self.ui.mode == 'manual'
                            self.place_number(row, col, digit, force=is_manual)
                    
                    # Delete/Clear
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                        if self.ui.selected_cell:
                            row, col = self.ui.selected_cell
                            self.place_number(row, col, 0)
                    
                    # Hint
                    elif event.key == pygame.K_h:
                        if self.ui.selected_cell:
                            row, col = self.ui.selected_cell
                            hint = self.get_hint(row, col)
                            if hint:
                                self.place_number(row, col, hint)
                                print(f"Hint: Place {hint} at ({row}, {col})")
                    
<<<<<<< HEAD
                    # Toggle theme
                    elif event.key == pygame.K_t:
                        self.ui.toggle_theme()
                    
                    # Auto-solve
=======
                    # Auto-solve with active mode
>>>>>>> 9e839a1566699572ecf80c319b1a8ea47ee095ac
                    elif event.key == pygame.K_SPACE:
                        if not auto_solving:
                            active_mode = self.ui.mode
                            if active_mode == 'rl':
                                auto_solving = True
                                print("Starting RL auto-solve...")
                                self.auto_solve_rl()
                                auto_solving = False
                                if self.game.is_complete():
                                    print(f"✓ Puzzle solved in {len(self.solve_steps_list)} steps!")
                            elif active_mode == 'backtracking':
                                auto_solving = True
                                print("Starting backtracking solve...")
                                self.auto_solve_backtracking()
                                auto_solving = False
                                if self.game.is_complete():
                                    print("✓ Puzzle solved with backtracking!")
                    
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
