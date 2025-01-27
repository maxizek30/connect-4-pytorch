import torch
import numpy as np
from environment import Connect4Env
from model import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_board(board):
    print("\nCurrent board:")
    for row in board:
        print(" ".join(f"{int(x):2d}" for x in row))
    print("Columns: 0  1  2  3  4  5  6\n")

def play_game(model_path):
    # 1) Create model and load weights
    model = QNetwork(num_actions=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2) Create environment
    env = Connect4Env()
    board = env.reset()

    # Human is Player 1, AI is Player 2
    # env.current_player = 1 at start

    done = False
    while not done:
        print_board(board)
        # === HUMAN MOVE ===
        valid_cols = env.valid_actions()
        user_move = None
        while user_move not in valid_cols:
            user_input = input(f"Your move (valid columns {valid_cols}): ")
            try:
                user_move = int(user_input)
            except ValueError:
                user_move = None
            if user_move not in valid_cols:
                print("Invalid column, try again.")

        board, reward, done = env.step(user_move)
        if done:
            print_board(board)
            if reward == 1:
                print("You won!")
            else:
                print("It's a tie!")
            break

        # === AI MOVE ===
        valid_cols = env.valid_actions()
        board_t = torch.from_numpy(board).float().unsqueeze(0).unsqueeze(0).to(device)
        q_values = model(board_t)[0].cpu().detach().numpy()
        best_valid_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]
        board, reward, done = env.step(best_valid_col)

        if done:
            print_board(board)
            if reward == 1:
                print("AI won!")
            else:
                print("It's a tie!")
            break

if __name__ == "__main__":
    # Example usage: python play_game.py
    model_checkpoint = "models/connect4_model_10000.pth"
    play_game(model_checkpoint)
