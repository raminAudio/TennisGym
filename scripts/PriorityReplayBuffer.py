import numpy as np
import torch
from collections import namedtuple, deque

# Define the Experience namedtuple
Experience = namedtuple("Experience", ("state", "action", "reward", "next_state", "done"))

# Priority Replay Buffer
class PriorityReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha  # Prioritization hyperparameter
        self.epsilon = 1e-6  # Small constant to prevent priority equal to zero
        self.buffer = []  # List to store experiences
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.pointer = 0

    def add(self, experience, td_error):
        priority = (td_error + self.epsilon) ** self.alpha
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pointer] = experience
        self.priorities[self.pointer] = priority
        self.pointer = (self.pointer + 1) % self.buffer_size

    def sample(self, beta=0.4):
        priorities = self.priorities[:len(self.buffer)]
        sampling_probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.buffer), size=self.batch_size, p=sampling_probabilities)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * sampling_probabilities[indices]) ** (-beta)
        weights /= np.max(weights)

        return indices, samples, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            priority = (td_error + self.epsilon) ** self.alpha
            self.priorities[i] = priority

    def __len__(self):
        return len(self.buffer)
