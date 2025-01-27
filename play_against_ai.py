import tensorflow as tf
import numpy as np

from environment import Connect4Env

def print_board(board):
    """
    Print the Connect4 board in a human-readable way.
    Player 1 =  1
    Player 2 = -1
    Empty    =  0
    """
    print("\nCurrent board:")
    for row in board:
        row_str = " ".join(
            f"{int(x):2d}" for x in row
        )
        print(row_str)
    print("Columns: 0  1  2  3  4  5  6\n")

def play_game(model_path):
    # 1) Load the trained model
    model = tf.keras.models.load_model(model_path)

    # 2) Initialize environment
    env = Connect4Env()
    board = env.reset()

    # 3) Decide who goes first. Let's say:
    #    - Human is "Player 1" (env.current_player == 1)
    #    - AI is "Player 2" (env.current_player == -1)
    # The environment already sets current_player = 1 at reset.

    done = False
    while not done:
        print_board(board)

        # === HUMAN MOVE ===
        valid_cols = env.valid_actions()
        user_move = None
        while user_move not in valid_cols:
            try:
                user_input = input(f"Your move (valid columns {valid_cols}): ")
                user_move = int(user_input)
                if user_move not in valid_cols:
                    print("Invalid column, try again.")
            except ValueError:
                print("Please enter a valid integer column.")

        board, reward, done = env.step(user_move)
        if done:
            print_board(board)
            if reward == 1:
                print("You won!")
            else:
                print("Game ended in a tie!")
            break

        # === AI MOVE ===
        valid_cols = env.valid_actions()
        # Preprocess board for the model (need shape (1, rows, cols, 1))
        board_input = board[..., np.newaxis][np.newaxis, ...]  # shape = (1,6,7,1)
        q_values = model.predict(board_input)[0]

        # Choose the column with the highest Q among the valid columns
        best_valid_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]
        board, reward, done = env.step(best_valid_col)
        if done:
            print_board(board)
            if reward == 1:
                print("AI won!")
            else:
                print("Game ended in a tie!")
            break

if __name__ == "__main__":
    # Adjust the path to one of your saved models
    # For instance, if you saved at "models/connect4_model_episode_50"
    model_checkpoint = "models/connect4_model_episode_950"

    play_game(model_checkpoint)
