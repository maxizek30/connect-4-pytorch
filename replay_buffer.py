from collections import deque
import random
import numpy as np

# Class to implement a Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, max_size=100_000):
        """
        Initialize the replay buffer.

        Args:
        - max_size (int): Maximum number of experiences the buffer can hold. Defaults to 100,000.

        The buffer is implemented as a deque (double-ended queue) with a fixed maximum length.
        When the buffer exceeds the max size, the oldest experiences are automatically discarded.
        """
        self.buffer = deque(maxlen=max_size) # Create a deque with a maximum size

    def add(self, experience):
        """
        Add a new experience to the replay buffer.

        Args:
        - experience (tuple): A tuple containing (state, action, reward, next_state, done).
        """
        self.buffer.append(experience) # Add the experience to the deque (FIFO if full)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.

        Args:
        - batch_size (int): The number of experiences to sample.

        Returns:
        - states (np.array): Array of states from the sampled experiences.
        - actions (np.array): Array of actions from the sampled experiences.
        - rewards (np.array): Array of rewards from the sampled experiences.
        - next_states (np.array): Array of next states from the sampled experiences.
        - dones (np.array): Array of 'done' flags from the sampled experiences.
        """
        # Randomly sample a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        # Separate the batch into individual arrays for states, actions, rewards, next_states, and dones
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def size(self):
        """
        Get the current size of the replay buffer.

        Returns:
        - int: The number of experiences currently stored in the buffer.
        """
        return len(self.buffer) # Return the number of items currently in the buffer