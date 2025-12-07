"""
==================== PHASE 1: THE TOUCH AGENT ====================

GOAL: Teach the agent to touch the ball infront or at a angle.

SCENARIOS USED:
  - Touch scenarios where the ball is randomly placed near and far from the agent

REWARD:
  - AdvancedTouchReward (100.0)

KEY SETTINGS:
  - PLR with 60% prioritized sampling
  - NoTouchTimeout: 10 seconds
  - Episode timeout: 45 seconds
  - 73-dim observation (72 standard + 1 scenario index)

DURATION: ~500M timesteps
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.done_conditions import AnyCondition, TimeoutCondition, NoTouchTimeoutCondition, GoalCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_NET_Y, CEILING_Z, CAR_MAX_SPEED, CAR_MAX_ANG_VEL
from rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer
from rocket_league_rl.rlgym_ppo.util import RLGymV2GymWrapper

# Import Rewards
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward

# PLR Imports
from plr_utils import PLRMutator, PLRObsBuilder
from plr_learner import PLRLearner


def build_phase_1_env():
    """
    Phase 1 Environment
    """

    # 1. SPAWN SETUP
    # PLRMutator handles spawning internally (blue_size=1, orange_size=0 by default)
    # 60% prioritized sampling, 40% random - helps focus on harder scenarios
    plr_mutator = PLRMutator(replay_prob=0.6, blue_size=1, orange_size=0)

    # 2. OBSERVATION
    # PLRObsBuilder: 72 standard dims + 1 scenario index = 73 dims
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
        # A. Touch Reward (100.0)
        (AdvancedTouchReward(
            touch_reward=100.0,
            acceleration_reward=0.0,
            good_touch_reward=0.0
        ), 1.0),

        # B. Goal Reward - Not Used
        (GoalReward(), 0.0)
    )

    # 4. TERMINATION CONDITIONS
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        # No touch timeout of 10 sec
        NoTouchTimeoutCondition(timeout_seconds=10),
        # Episode timeout of 45 sec
        TimeoutCondition(timeout_seconds=45)
    )

    # 5. BUILD ENV
    rlgym_env = RLGym(
        state_mutator=plr_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1: THE TOUCH AGENT")
    print("=" * 70)
    print("Scenarios: Touch_Close, Touch_Mid, Touch_Far, Touch_Steering")
    print("Observation: 73 dims")
    print("Duration: ~500M timesteps")
    print("=" * 70)

    # Configuration
    n_proc = 8
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = PLRLearner(
        build_phase_1_env,
        wandb_run_name="Phase1_TouchAgent",
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,

        # Rendering
        render=False,
        render_last_only=True,

        # PPO Hyperparameters
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

        # Start from scratch
        checkpoint_load_folder=None
    )

    print("\n" + "=" * 70)
    print("Starting Phase 1 Training...")
    print("=" * 70 + "\n")

    learner.learn()
