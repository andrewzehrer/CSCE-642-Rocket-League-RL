import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rocket_league_rl'))

import numpy as np
import torch

from rocket_league_rl.rlgym_ppo.learner import Learner
from rocket_league_rl.rlgym_ppo.util import torch_functions
from plr_utils import PLRBuffer

class PLRLearner(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plr_buffer = PLRBuffer()
        print("PLR Learner Initialized")

    @torch.no_grad()
    def add_new_experience(self, experience):
        # Unpack the data
        states, actions, log_probs, rewards, next_states, dones, truncated = experience
        value_net = self.ppo_learner.value_net

        # Extract the Scenario Index from the last column of observation
        # This tells us which scenario generated this data
        if isinstance(states, torch.Tensor):
            seed_indices = states[:, -1].cpu().numpy()
        else:
            seed_indices = states[:, -1]

        # Calculate Difficulty (Value Loss / Advantage)
        val_inp = np.zeros(shape=(states.shape[0] + 1, states.shape[1]))
        val_inp[:-1] = states
        val_inp[-1] = next_states[-1]
        
        val_preds = value_net(val_inp).cpu().flatten().tolist()
        torch.cuda.empty_cache()
        
        ret_std = self.return_stats.std[0] if self.standardize_returns else None
        value_targets, advantages, returns = torch_functions.compute_gae(
            rewards, dones, truncated, val_preds,
            gamma=self.gae_gamma, lmbda=self.gae_lambda, return_std=ret_std
        )
        
        # Update the Buffer
        # abs(advantage) represents how surprised the agent was.
        # High Surprise = High Learning Potential.
        errors = torch.abs(advantages).cpu().numpy()
        self.plr_buffer.update(seed_indices, errors)

        # Submit to standard PPO buffer
        if self.standardize_returns:
            n = min(self.max_returns_per_stats_increment, len(returns))
            self.return_stats.increment(returns[:n], n)

        self.experience_buffer.submit_experience(
            states, actions, log_probs, rewards, next_states,
            dones, truncated, value_targets, advantages
        )