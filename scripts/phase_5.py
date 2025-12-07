"""
==================== PHASE 5: SELF-PLAY ====================

GOAL: Increase the speed and efficiency of phase 4 agent.

REWARD: Same as Phase 4.5

KEY SETTINGS:
  - orange_size=1 (enables 1v1 self-play)
  - Task Dropout always true with force_blind training
  - Lower learning rate (5e-5)
  - All 10 scenarios enabled (mirrored for orange team)

DURATION: Continue from Phase 4, ~100-200M additional timesteps
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.done_conditions import AnyCondition, TimeoutCondition, NoTouchTimeoutCondition, GoalCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_NET_Y, CEILING_Z, CAR_MAX_SPEED, CAR_MAX_ANG_VEL
from rocket_league_rl.rlgym_ppo.util import RLGymV2GymWrapper

# Import Custom Rewards
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.velocity_ball_to_goal_reward import VelocityBallToGoalReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.liu_distance_player_to_ball_reward import LiuDistancePlayerToBallReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.align_ball_to_goal_reward import AlignBallToGoalReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.face_ball_reward import FaceBallReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import AerialDistanceReward

# PLR Imports
from plr_utils import PLRMutator, PLRObsBuilder
from plr_learner import PLRLearner


def build_phase_5_env():
    """
    Phase 5 Environment: Self-Play
    """

    # 1. SPAWN SETUP (1v1 SELF-PLAY)
    # orange_size=1 enables self-play: both cars use the same policy
    # PLRMutator handles mirroring: orange car spawns at (-x, -y, z)
    plr_mutator = PLRMutator(replay_prob=0.6, blue_size=1, orange_size=1)

    # 2. OBSERVATION (73 dims per agent)
    # force_blind=True: Agent always sees scenario_idx=0.0
    base_obs = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / SIDE_WALL_X, 1 / BACK_NET_Y, 1 / CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / CAR_MAX_SPEED,
        ang_vel_coef=1 / CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0
    )
    obs_builder = PLRObsBuilder(base_obs, task_dropout=0.0, force_blind=True)  # Always blind

    # 3. REWARD FUNCTION
    reward_fn = CombinedReward(
        # A. GOAL (+500)
        (GoalReward(), 500.0),

        # B. VELOCITY TO BALL (+1.5)
        (VelocityPlayerToBallReward(), 1.5),

        # C. LIU DISTANCE (+0.5)
        (LiuDistancePlayerToBallReward(), 0.5),

        # D. TOUCH (+5.0)
        (TouchReward(), 5.0),

        # E. ALIGN BALL TO GOAL (+0.5)
        (AlignBallToGoalReward(), 0.5),

        # F. VELOCITY BALL TO GOAL (+1.0)
        (VelocityBallToGoalReward(), 1.0),

        # G. AERIAL DISTANCE (+5.0) - Reduced to 5.0
        (AerialDistanceReward(
            touch_height_weight=1.0,
            car_distance_weight=0.5,
            ball_distance_weight=0.5
        ), 5.0),

        # H. FACE BALL (+0.05)
        (FaceBallReward(), 0.05),
    )

    # 4. TERMINATION CONDITIONS
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=10),
        TimeoutCondition(timeout_seconds=45)
    )

    # 5. BUILD ENV
    return RLGymV2GymWrapper(RLGym(
        state_mutator=plr_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    ))


if __name__ == "__main__":
    phase_4_checkpoint = "/Users/mrunmayd/Projects/DeepRL_Project/CSCE-642-Rocket-League-RL/scripts/data/checkpoints/rlgym-ppo-run-1764565733513209000/400024369"

    print("=" * 70)
    print("PHASE 5: SELF-PLAY")
    print("=" * 70)
    print("Scenarios (1v1 Compatible):")
    print("  1. Kickoff")
    print("  2. Recovery_Defense")
    print("  3. Rebound_Chase")
    print("  4. Wall_Shot_Left")
    print("  5. Wall_Shot_Right")
    print("  6. Fast_Rolling")
    print("  7. Turn_Back_Easy")
    print("  8. Turn_Back_Hard")
    print("  9. Aerial_Hover")
    print(" 10. High_Pop")
    print("=" * 70)

    # Check checkpoint path
    if not os.path.exists(phase_4_checkpoint):
        print("\n" + "!" * 70)
        print("ERROR: Checkpoint not found!")
        print("!" * 70)
        print(f"Path: {phase_4_checkpoint}")
        print("")
        print("Please update 'phase_4_checkpoint' with your Phase 4 checkpoint.")
        print("!" * 70)
        sys.exit(1)

    # Configuration
    n_proc = 8
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = PLRLearner(
        build_phase_5_env,
        wandb_run_name="Phase5_SelfPlay_Duel",
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,

        # Rendering (disabled for training speed)
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

        # Learning Rates (Lower for stability)
        policy_lr=5e-5,
        critic_lr=5e-5,

        # Saving
        save_every_ts=50_000_000,
        timestep_limit=200_000_000,
        log_to_wandb=True,

        checkpoint_load_folder=phase_4_checkpoint,

        # Fresh WandB run
        load_wandb=False
    )

    # Reset timestep counter for fresh WandB graphs
    learner.agent.cumulative_timesteps = 0
    print("\n[INFO] Reset cumulative_timesteps to 0 for fresh WandB graphs")

    print("\n" + "=" * 70)
    print("Starting Phase 5 Training...")
    print("=" * 70 + "\n")

    learner.learn()
