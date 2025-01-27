import numpy as np

class Connect4Env:
    def __init__(self):
        """
        Initialize the Connect4 game environment.
        - The board has 6 rows and 7 columns.
        - Player 1 starts the game (denoted by +1).
        - Player 2 is denoted by -1.
        """
        self.rows = 6
        self.columns = 7
        self.board = self._initialize_board()
        self.done = False
        self.current_player = 1  # +1 for Player 1, -1 for Player 2

    def _initialize_board(self):
        """
        Create an empty Connect4 board.
        """
        return np.zeros((self.rows, self.columns), dtype=int)

    def reset(self):
        """
        Reset the game environment to its initial state.
        """
        self.board = self._initialize_board()
        self.done = False
        self.current_player = 1
        return self.board

    def valid_actions(self):
        """
        Return a list of valid column indices where a piece can be dropped.
        """
        return [c for c in range(self.columns) if self.board[0, c] == 0]

    def step(self, action):
        """
        Drop a piece in the specified column, then check for wins or tie.

        Returns:
            (board, reward, done)
        """
        if action not in self.valid_actions():
            raise ValueError("Invalid action")

        # Drop the piece into the lowest available cell in the selected column
        for row in reversed(range(self.rows)):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        # Check if current player wins by this move
        if self._check_win(self.current_player):
            self.done = True
            reward = 1.0  # <-- Reward for a winning move
        else:
            # Check for tie
            if len(self.valid_actions()) == 0:
                self.done = True
                reward = 0.5  # <-- Reward for a draw
            else:
                # Ongoing game
                reward = -0.05  # <-- Small negative penalty each step
                self.done = False

        # Switch player
        self.current_player *= -1

        # Clip reward (usually a safe practice)
        reward = np.clip(reward, -1.0, 1.0)

        return self.board.copy(), reward, self.done

    def _check_win(self, player):
        """
        Check if the given player has won the game (4 in a row).
        """
        # Check horizontal wins
        for row in range(self.rows):
            for col in range(self.columns - 3):
                if np.all(self.board[row, col:col+4] == player):
                    return True
        # Check vertical wins
        for row in range(self.rows - 3):
            for col in range(self.columns):
                if np.all(self.board[row:row+4, col] == player):
                    return True
        # Check diagonal wins (bottom-left to top-right)
        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                if np.all([self.board[row + i, col + i] == player for i in range(4)]):
                    return True
                # Check diagonal (top-left to bottom-right)
                if np.all([self.board[row+3 - i, col + i] == player for i in range(4)]):
                    return True
        return False
