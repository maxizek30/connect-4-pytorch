import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np

from environment import Connect4Env
from replay_buffer import ReplayBuffer
from model import QNetwork
from training import train_step

# Detect if we have a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def evaluate_agent_selfplay(env, model, num_games=10):
    """
    Evaluate the agent by letting the same model play both sides.
    """
    model.eval()  # switch to eval mode
    wins_p1 = 0
    wins_p2 = 0

    with torch.no_grad():
        for _ in range(num_games):
            state = env.reset()
            done = False

            while not done:
                # === Player 1 move ===
                valid_cols = env.valid_actions()
                state_t = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
                q_values = model(state_t)[0].cpu().numpy()
                best_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]
                state, reward, done = env.step(best_col)
                if done:
                    winner = -env.current_player
                    if winner == 1:
                        wins_p1 += 1
                    elif winner == -1:
                        wins_p2 += 1
                    break

                # === Player 2 move ===
                valid_cols = env.valid_actions()
                state_t = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
                q_values = model(state_t)[0].cpu().numpy()
                best_col = valid_cols[np.argmax([q_values[c] for c in valid_cols])]
                state, reward, done = env.step(best_col)
                if done:
                    winner = -env.current_player
                    if winner == 1:
                        wins_p1 += 1
                    elif winner == -1:
                        wins_p2 += 1
                    break

    ties = num_games - wins_p1 - wins_p2
    print(f"[Self-Play Eval] P1 wins: {wins_p1}, P2 wins: {wins_p2}, Ties: {ties}")
    model.train()  # back to train mode

def main():
    # Initialize environment and replay buffer
    env = Connect4Env()
    replay_buffer = ReplayBuffer(max_size=1_000_000)

    # Create DQN model + target model
    model = QNetwork(num_actions=7).to(device)
    target_model = QNetwork(num_actions=7).to(device)
    target_model.load_state_dict(model.state_dict())

    # Hyperparameters
    batch_size = 128
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.99995
    target_update_freq = 10

    # Optimizer + Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    episode = 0
    while True:
        episode += 1

        # Optionally evaluate every N episodes
        if episode % 1000 == 0:
            torch.save(model.state_dict(), f"models/connect4_model_{episode}.pth")
            evaluate_agent_selfplay(env, model, num_games=50)

        # Reset environment
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(env.valid_actions())
            else:
                valid_cols = env.valid_actions()
                state_t = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(device)
                q_values = model(state_t)[0].detach().cpu().numpy()
                action = valid_cols[np.argmax([q_values[c] for c in valid_cols])]

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Save transition
            replay_buffer.add((state, action, reward, next_state, done))

            # Training step
            train_step(model, target_model, replay_buffer, batch_size, gamma, optimizer, loss_fn, device)

            # Move to next state
            state = next_state

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {episode} => Reward: {total_reward}, Epsilon: {epsilon:.4f}")



if __name__ == "__main__":
    main()



