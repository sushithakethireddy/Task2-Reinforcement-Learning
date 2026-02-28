"""

Partial Observability Wrapper for Chef's Hat Gym Environment.

Variant: ID mod 7 = 4 - Partial Observability Variant
Student ID: 16387858

This wrapper restricts what the agent can observe from the full game state,
simulating real-world partial information scenarios. Four observability levels
are implemented to systematically study the effect of information restriction.

Observability Levels:
    Level 1 - Full observation (baseline, no masking)
    Level 2 - Hide opponent hands only
    Level 3 - Hide opponent hands and discard history
    Level 4 - Minimal observation (own hand and current play only)
"""

import numpy as np


class PartialObservabilityWrapper:
    """
    Wraps the Chef's Hat environment to restrict agent observations.

    This is the core contribution of the Partial Observability variant.
    By masking portions of the observation vector, we simulate limited
    information access and force the agent to rely on memory-based
    approaches (LSTM) to compensate for hidden information.

    Parameters
    ----------
    env : ChefsHatEnv
        The base Chef's Hat environment instance.
    level : int
        Observability level from 1 (full) to 4 (minimal).
    verbose : bool
        If True, prints masking details on first observation.
    """

    LEVEL_DESCRIPTIONS = {
        1: "Full Observation - Agent sees all game state information",
        2: "Level 2 - Opponent hands are hidden from agent",
        3: "Level 3 - Opponent hands and discard history are hidden",
        4: "Minimal Observation - Agent sees only own hand and current play"
    }

    def __init__(self, env, level=1, verbose=False):
        assert level in [1, 2, 3, 4], "Observability level must be 1, 2, 3, or 4"
        self.env = env
        self.level = level
        self.verbose = verbose
        self.full_obs_size = None
        self.obs_size = None
        self._mask = None
        self._initialized = False

    def reset(self):
        """
        Reset the environment and apply observability mask.

        Returns
        -------
        numpy.ndarray
            Masked observation vector.
        """
        out = self.env.reset()
        if isinstance(out, tuple):
            full_obs = out[0]
        else:
            full_obs = out
        full_obs = np.array(full_obs, dtype=np.float32)

        if not self._initialized:
            self.full_obs_size = len(full_obs)
            self._build_mask(self.full_obs_size)
            self.obs_size = self.full_obs_size
            self._initialized = True

            if self.verbose:
                print("Partial Observability Wrapper Initialized")
                print("Full observation size: {}".format(self.full_obs_size))
                print("Level: {} - {}".format(
                    self.level, self.LEVEL_DESCRIPTIONS[self.level]))
                print("Masked indices count: {}".format(
                    int(np.sum(self._mask == 0))))

        return self._apply_mask(full_obs)

    def step(self, action):
        """
        Take a step in the environment and apply observability mask.

        Parameters
        ----------
        action : int
            Action selected by the agent.

        Returns
        -------
        tuple
            (masked_observation, reward, done, info)
        """
        full_obs, reward, done, info = self.env.step(action)
        full_obs = np.array(full_obs, dtype=np.float32)
        partial_obs = self._apply_mask(full_obs)
        return partial_obs, reward, done, info

    def _build_mask(self, obs_size):
        """
        Build the binary mask for the chosen observability level.

        The observation vector structure in Chef's Hat is divided into
        segments representing different aspects of the game state.
        We partition the vector equally across 4 regions for masking.

        Parameters
        ----------
        obs_size : int
            Total size of the full observation vector.
        """
        self._mask = np.ones(obs_size, dtype=np.float32)

        quarter = obs_size // 4
        half = obs_size // 2
        three_quarter = (obs_size * 3) // 4

        if self.level == 1:
            # No masking - full observation kept
            pass

        elif self.level == 2:
            # Mask opponent hand region (second quarter of observation)
            self._mask[quarter:half] = 0.0

        elif self.level == 3:
            # Mask opponent hands AND action/discard history
            self._mask[quarter:three_quarter] = 0.0

        elif self.level == 4:
            # Keep only own hand region (first quarter)
            self._mask[quarter:] = 0.0

    def _apply_mask(self, obs):
        """
        Apply the binary mask to an observation.

        Parameters
        ----------
        obs : numpy.ndarray
            Full observation from the environment.

        Returns
        -------
        numpy.ndarray
            Masked observation with hidden regions zeroed out.
        """
        if self._mask is None:
            return obs
        return obs * self._mask

    def get_mask_info(self):
        """
        Return information about the current mask configuration.

        Returns
        -------
        dict
            Dictionary with mask statistics.
        """
        if self._mask is None:
            return {"status": "not initialized"}

        visible = int(np.sum(self._mask == 1))
        hidden = int(np.sum(self._mask == 0))
        ratio = visible / len(self._mask) if len(self._mask) > 0 else 0

        return {
            "level": self.level,
            "description": self.LEVEL_DESCRIPTIONS[self.level],
            "total_obs_size": len(self._mask),
            "visible_features": visible,
            "hidden_features": hidden,
            "visibility_ratio": round(ratio, 3)
        }

    def render(self):
        """Delegate render to base environment if available."""
        if hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        """Close the base environment."""
        if hasattr(self.env, 'close'):
            self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space