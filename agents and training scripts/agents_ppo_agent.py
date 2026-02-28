"""
PPO Agent for Chef's Hat GYM - Partial Observability Variant
Student ID: 16387858 | ID mod 7 = 4
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.getcwd(), "ChefsHatGYM", "src"))
from agents.base_agent import BaseAgent


class PPONetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=128):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_size, action_size)
        self.critic_head = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)

    def forward(self, x):
        features = self.shared(x)
        return self.actor_head(features), self.critic_head(features)


class PPOAgent(BaseAgent):
    """
    PPO Feedforward Agent extending ChefsHat BaseAgent.
    Memory-free baseline for Partial Observability comparison.
    Reference: Schulman et al. (2017) - https://arxiv.org/abs/1707.06347
    """

    def __init__(
        self,
        name,
        log_directory="",
        obs_size=200,
        action_size=200,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=4,
        hidden_size=128,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        obs_level=2,
        verbose_console=False
    ):
        super().__init__(
            name=name,
            log_directory=log_directory,
            verbose_console=verbose_console
        )
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.obs_level = obs_level

        self.network = PPONetwork(obs_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self._reset_buffers()
        self.training_stats = {
            "actor_losses": [],
            "critic_losses": [],
            "entropies": [],
            "episode_scores": []
        }
        self._last_obs = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None
        self._last_possible_actions = None

    def _reset_buffers(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def _apply_partial_obs(self, obs):
        """Apply observation masking based on observability level."""
        obs = np.array(obs, dtype=np.float32)
        if len(obs) != self.obs_size:
            obs = np.resize(obs, self.obs_size)
        mask = np.ones(self.obs_size, dtype=np.float32)
        q = self.obs_size // 4
        if self.obs_level == 2:
            mask[q:2*q] = 0.0
        elif self.obs_level == 3:
            mask[q:3*q] = 0.0
        elif self.obs_level == 4:
            mask[q:] = 0.0
        return obs * mask

    def request_action(self, info):
        """
        Called by ChefsHat Room when this agent must play.
        possible_actions is a list of action strings e.g. ['C11;Q1;J0', 'pass']
        """
        hand = np.array(info.get("hand", [])).flatten() / 13.0
        board = np.array(info.get("board", [])).flatten() / 13.0
        possible_actions = list(info.get("possible_actions", []))
        obs_raw = np.concatenate([hand, board]) if len(hand) > 0 else np.zeros(self.obs_size)
        obs = self._apply_partial_obs(obs_raw)
        self._last_obs = obs
        self._last_possible_actions = possible_actions

        state = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.network(state)

            # Mask to only valid action indices (positions 0..len-1)
            if possible_actions is not None and len(possible_actions) > 0:
                mask = torch.full(logits.shape, float('-inf'))
                num_valid = min(len(possible_actions), logits.shape[1])
                for a in range(num_valid):
                    mask[0][a] = 0.0
                logits = logits + mask

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        self._last_action = action.item()
        self._last_log_prob = log_prob
        self._last_value = value.squeeze()

        
        # ChefsHat expects integer action index returned
        action_idx = min(self._last_action, len(possible_actions) - 1)
        return action_idx

    def update_match_over(self, info):
        """Called by ChefsHat when a match ends — store reward and learn."""
        finishing_order = info.get("finishing_order", [])
        if self.name in finishing_order:
            position = finishing_order.index(self.name)
            reward = 1.0 - (position / max(len(finishing_order) - 1, 1))
        else:
            reward = 0.0

        if self._last_obs is not None and self._last_log_prob is not None:
            self.states.append(self._last_obs)
            self.actions.append(self._last_action)
            self.rewards.append(reward)
            self.log_probs.append(self._last_log_prob)
            self.values.append(self._last_value)
            self.dones.append(True)

        self.training_stats["episode_scores"].append(reward)

        if len(self.states) >= 20:
            self._update()

    def _update(self):
        """Perform PPO update on collected experience."""
        if len(self.states) == 0:
            return

        # Compute discounted returns
        returns = []
        R = 0.0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                R = 0.0
            R = r + self.gamma * R
            returns.insert(0, R)

        states_t = torch.FloatTensor(np.array(self.states))
        actions_t = torch.LongTensor(np.array(self.actions))
        old_lp_t = torch.stack(self.log_probs).detach()
        returns_t = torch.FloatTensor(returns)
        old_v_t = torch.stack(self.values).detach()

        advantages = returns_t - old_v_t
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = torch.tensor(0.0)
        critic_loss = torch.tensor(0.0)
        entropy = torch.tensor(0.0)

        for _ in range(self.epochs):
            logits, values = self.network(states_t)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns_t)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.training_stats["actor_losses"].append(actor_loss.item())
        self.training_stats["critic_losses"].append(critic_loss.item())
        self.training_stats["entropies"].append(entropy.item())
        self._reset_buffers()

    def save(self, path):
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "training_stats" in ckpt:
            self.training_stats = ckpt["training_stats"]