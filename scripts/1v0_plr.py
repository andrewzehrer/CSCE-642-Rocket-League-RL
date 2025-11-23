import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from rocket_league_rl.rlgym.api import RLGym
from rocket_league_rl.rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rocket_league_rl.rlgym.rocket_league.done_conditions import GoalCondition, AnyCondition, TimeoutCondition, NoTouchTimeoutCondition
from rocket_league_rl.rlgym.rocket_league.obs_builders import DefaultObs
from rocket_league_rl.rlgym.rocket_league.reward_functions import CombinedReward
from rocket_league_rl.rlgym.rocket_league.sim import RocketSimEngine
from rocket_league_rl.rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_NET_Y, CEILING_Z, CAR_MAX_SPEED, CAR_MAX_ANG_VEL
from rocket_league_rl.rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator
from rocket_league_rl.rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer
from rocket_league_rl.rlgym_ppo.util import RLGymV2GymWrapper

# Rewards
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward

# PLR Imports
from scripts.plr_utils import PLRMutator, PLRObsBuilder
from scripts.plr_learner import PLRLearner

def build_rlgym_v2_env():
    spawn_cars = FixedTeamSizeMutator(blue_size=1, orange_size=0)
    # Replay_prob decides how frequently the hard scenarios are sampled
    apply_physics = PLRMutator(replay_prob=0.0)
    
    state_mutator = MutatorSequence(spawn_cars, apply_physics)

    # --- OBS BUILDER ---
    base_obs = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / SIDE_WALL_X, 1 / BACK_NET_Y, 1 / CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / CAR_MAX_SPEED,
        ang_vel_coef=1 / CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0
    )
    obs_builder = PLRObsBuilder(base_obs)
    
    # --- REWARDS ---
    reward_fn = CombinedReward(
        (AdvancedTouchReward(acceleration_reward=0.0), 15.0),
        (VelocityPlayerToBallReward(), 0.01) 
    )
    
    # --- TERMINATION ---
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=10), 
        TimeoutCondition(timeout_seconds=300)
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
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
    n_proc = 8
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = PLRLearner(
        build_rlgym_v2_env,
        wandb_run_name="Baseline_PLR",
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        render=True,
        render_last_only=False, 
        ppo_batch_size=50000,
        ts_per_iteration=50000,
        exp_buffer_size=150000,
        ppo_minibatch_size=50000,
        policy_layer_sizes=[512, 512],
        critic_layer_sizes=[512, 512],
        ppo_ent_coef=0.005,
        policy_lr=5e-5,
        critic_lr=5e-5,
        ppo_epochs=1,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=100_000,
        timestep_limit=2_000_000,
        log_to_wandb=True
    )

    learner.learn()