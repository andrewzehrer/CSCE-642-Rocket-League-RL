import os
import sys
# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rocket_league_rl'))

import numpy as np
import torch
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.done_conditions import AnyCondition, TimeoutCondition, NoTouchTimeoutCondition, GoalCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_NET_Y, CEILING_Z, CAR_MAX_SPEED, CAR_MAX_ANG_VEL
from rocket_league_rl.rlgym_ppo.util import RLGymV2GymWrapper

# Import Custom Velocity Reward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.velocity_ball_to_goal_reward import VelocityBallToGoalReward

# PLR Imports
from plr_utils import PLRMutator, PLRObsBuilder
from plr_learner import PLRLearner

def build_phase_2_env():
    """
    Phase 2: THE GOAL SCORER 
    """
    # 1. SPAWN SETUP (1v0)
    # PLRMutator handles spawning internally
    plr_mutator = PLRMutator(replay_prob=0.6, blue_size=1, orange_size=0)

    # 2. OBSERVATION
    base_obs = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / SIDE_WALL_X, 1 / BACK_NET_Y, 1 / CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / CAR_MAX_SPEED,
        ang_vel_coef=1 / CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0
    )
    obs_builder = PLRObsBuilder(base_obs)

    # 3. REWARD FUNCTION
    reward_fn = CombinedReward(
        # A. Goal Reward (+100 for scoring, -100 for own goals)
        (GoalReward(), 100.0),

        # B. Velocity Guidance (+1.0) - Continuous directional feedback
        # This continuous feedback accelerates learning of directionality
        (VelocityBallToGoalReward(), 1.0),

        # C. Touch Reward (+1.0)
        (TouchReward(), 1.0)
    )

    # 4. TERMINATION
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=10),
        TimeoutCondition(timeout_seconds=20)
    )

    return RLGymV2GymWrapper(RLGym(
        state_mutator=plr_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=None  # No rendering during training
    ))

def load_weights_only(learner, checkpoint_path):
    print(f"\n{'='*70}\nLOADING WEIGHTS ONLY (Resetting Optimizer)\n{'='*70}")
    policy_path = os.path.join(checkpoint_path, "PPO_POLICY.pt")
    value_path = os.path.join(checkpoint_path, "PPO_VALUE_NET.pt")
    
    if not os.path.exists(policy_path): raise FileNotFoundError(f"Not found: {policy_path}")
    
    learner.agent.policy.load_state_dict(torch.load(policy_path, map_location='cpu'))
    learner.ppo_learner.value_net.load_state_dict(torch.load(value_path, map_location='cpu'))
    print("Only Weights Loaded.\n")

if __name__ == "__main__":
    phase1_checkpoint = "/Users/mrunmayd/Projects/DeepRL_Project/CSCE-642-Rocket-League-RL/scripts/data/checkpoints/rlgym-ppo-run-1764063044117131000/500030092"

    learner = PLRLearner(
        build_phase_2_env,
        wandb_run_name="Phase2_GoalScorer",
        n_proc=8,
        min_inference_size=7,
        ppo_batch_size=50000,
        ts_per_iteration=50000,
        exp_buffer_size=150000,
        ppo_minibatch_size=50000,
        ppo_ent_coef=0.01,
        ppo_epochs=1,
        standardize_returns=True,
        standardize_obs=False,

        # Network Architecture
        policy_layer_sizes=[512, 512],
        critic_layer_sizes=[512, 512],

        # Learning Rates
        policy_lr=1e-4,
        critic_lr=1e-4,

        # Saving
        save_every_ts=50_000_000,
        timestep_limit=500_000_000,
        log_to_wandb=True,
        checkpoint_load_folder=None
    )

    load_weights_only(learner, phase1_checkpoint)
    learner.learn()