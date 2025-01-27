import numpy as np


class Connect4Env:
    def __init__(self):
        """
        Initialize the Connect4 game environment.
        - The board has 6 rows and 7 columns.
        - Player 1 starts the game.
        """
        self.rows = 6
        self.columns = 7
        self.board = self._initialize_board()
        self.done = False
        self.current_player = 1

    def _initialize_board(self):
        """
        Create an empty Connect4 board.
        - A board is represented as a 2D NumPy array of zeros.
        - Zero indicates an empty space.
        """
        return np.zeros((self.rows, self.columns), dtype=int)

    def reset(self):
        """
        Reset the game environment to its initial state.
        - The board is cleared, the game is marked as not done,
          and Player 1 is set to start.
        """
        self.board = self._initialize_board()
        self.done = False
        self.current_player = 1
        return self.board

    def valid_actions(self):
        """
        Return a list of valid column indices where a piece can be dropped.
        - A column is valid if its topmost cell (row 0) is empty (value 0).
        """
        return [c for c in range(self.columns) if self.board[0, c] == 0]

    def step(self, action):
        """
        Perform an action (drop a piece in the specified column).
        Args:
        - action (int): The column where the current player wants to drop their piece.

        Returns:
        - board (2D array): The current state of the board after the move.
        - reward (int): The reward for the move (1 if current player wins, 0 otherwise).
        - done (bool): Whether the game has ended.
        """
        if action not in self.valid_actions():
            raise ValueError("Invalid action")

        # Drop the piece into the lowest available cell in the selected column
        for row in reversed(range(self.rows)):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break
        # Check if the move resulted in a win for the current player
        if self._check_win(self.current_player):
            self.done = True
            reward = 1
        # Check if the board is full, resulting in a tie
        elif len(self.valid_actions()) == 0:
            self.done = True
            reward = 0 # Reward for a tie
        else:
            reward = 0 # No reward for ongoing games

        # Switch to the other player
        self.current_player *= -1
        # Ensure the reward is between -1 and 1
        reward = np.clip(reward, -1, 1)
        # Return a copy of the board, the reward, and the game's status
        return self.board.copy(), reward, self.done

    def _check_win(self, player):
        """
        Check if the given player has won the game.
        - A player wins by connecting four pieces in a row, column, or diagonal.
        Args:
        - player (int): The player to check (1 for Player 1, -1 for Player 2).

        Returns:
        - bool: True if the player has won, False otherwise.
        """
        # Check horizontal wins
        for row in range(self.rows):
            for col in range(self.columns - 3):
                if np.all(self.board[row, col:col + 4] == player):
                    return True
        # Check vertical wins
        for row in range(self.rows - 3):
            for col in range(self.columns):
                if np.all(self.board[row:row + 4, col] == player):
                    return True
        # Check diagonal wins (bottom-left to top-right)
        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                if np.all([self.board[row + i, col + i] == player for i in range(4)]):
                    return True
                # Check diagonal wins (top-left to bottom-right)
                if np.all([self.board[row + 3 - i, col + i] == player for i in range(4)]):
                    return True

        # No win found
        return False
