import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Store and sample transitions for experience replay in reinforcement learning."""

    def __init__(self, max_size):
        """Initialize buffer with maximum capacity."""

        # Create a double-ended queue with fixed maximum size. When the buffer is full,
        # adding new items automatically removes the oldest ones.
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        """Add a new transition to the replay buffer."""

        # Store a complete transition as a tuple containing all information needed
        # for learning: current state, action taken, reward received, next state,
        # and whether the episode ended.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, continuous: bool = False):
        """Randomly sample a batch of transitions for training."""

        # Take a random sample of transitions from our buffer to break correlations
        # in sequential data.
        transitions = random.sample(self.buffer, batch_size)

        # Reorganize the batch of transitions into separate arrays for each component.
        # This transforms a list of (state, action, reward, next_state, done) tuples
        # into separate arrays for states, actions, rewards, etc.
        batch = list(zip(*transitions))

        # Convert to numpy arrays with appropriate shapes and types for training.
        states = np.array(batch[0], dtype=np.float32)
        actions = (
            np.array(batch[1], dtype=np.float32).reshape(batch_size, -1)
            if continuous
            else np.array(batch[1], dtype=np.int32)
        )
        rewards = np.array(batch[2], dtype=np.float32)
        next_states = np.array(batch[3], dtype=np.float32)
        dones = np.array(batch[4], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def get_latest(self):
        """Retrieve the most recently added transition."""

        latest = self.buffer[-1]

        state = np.array(latest[0], dtype=np.float32)
        action = np.array(latest[1], dtype=np.float32)
        reward = np.array(latest[2], dtype=np.float32)
        next_state = np.array(latest[3], dtype=np.float32)
        done = np.array(latest[4], dtype=np.float32)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
