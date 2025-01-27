import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_step(model, target_model, replay_buffer, batch_size, gamma, optimizer, loss_fn, device):
    """
    One step of DQN training:
      - Sample from replay buffer
      - Compute Q(s,a) and target Q(s,a)
      - Backprop to update model
    """
    if replay_buffer.size() < batch_size:
        return  # Not enough samples yet

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert to torch tensors
    # States shape: (batch, 6, 7) => reshape to (batch,1,6,7)
    states_t = torch.from_numpy(states).float().unsqueeze(1).to(device)
    next_states_t = torch.from_numpy(next_states).float().unsqueeze(1).to(device)
    actions_t = torch.from_numpy(actions).long().to(device)
    rewards_t = torch.from_numpy(rewards).float().to(device)
    dones_t = torch.from_numpy(dones).float().to(device)

    # Q-values for current states
    q_values = model(states_t)  # shape (batch,7)
    # Extract Q for the chosen action
    q_value_chosen = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # shape (batch,)

    # Q-values for next states (using target network)
    with torch.no_grad():
        next_q_values = target_model(next_states_t)  # shape (batch,7)
        max_next_q, _ = torch.max(next_q_values, dim=1)  # shape (batch,)

    # Bellman target
    target = rewards_t + gamma * (1 - dones_t) * max_next_q

    # Compute loss
    loss = loss_fn(q_value_chosen, target)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
