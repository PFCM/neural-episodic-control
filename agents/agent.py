"""Base agent, and a simple example implementation"""
import numpy as np


class AbstractAgent(object):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def act(self, observation):
        pass

    def reward(self, value):
        pass

    def episode_ended(self):
        pass


class RandomAgent(AbstractAgent):
    """An agent which just makes random moves."""

    def __init__(self, num_actions):
        self._num_actions = num_actions

    def act(self, observation):
        return np.random.randint(self._num_actions)
