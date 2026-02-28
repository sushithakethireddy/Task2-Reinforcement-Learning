"""
Random Agent for Chef's Hat GYM
Student ID: 16387858 | ID mod 7 = 4
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "ChefsHatGYM", "src"))
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, name, log_directory="", verbose_console=False, seed=None):
        super().__init__(name=name, log_directory=log_directory, verbose_console=verbose_console)
        self.rng = np.random.RandomState(seed)

    def request_action(self, info):
        valid_actions = info.get("possible_actions", None)
        if valid_actions is not None:
            valid = [i for i, v in enumerate(valid_actions) if v == 1]
            if valid:
                return int(self.rng.choice(valid))
        action_size = len(info.get("possible_actions", [200]))
        return int(self.rng.randint(0, max(action_size, 1)))