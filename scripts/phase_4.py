"""
==================== PHASE 4: THE GENERALIST AGENT ====================

GOAL: Fix the scenario id dependence issue.

PROBLEMS SOLVED:
  1. Scenario Id Dependence: Agent decided its movements based on the scenario id

THE SOLUTION:
  - Task Dropout: Introduced Task Dropout of 0.5 so that the agent learns on the skills on kickoff (0.0)

SCENARIOS:
  1. Kickoff
  2. Recovery_Defense
  3. Rebound_Chase
  4. Wall_Shot_Left
  5. Wall_Shot_Right
  6. Fast_Rolling
  7. Turn_Back_Easy
  8. Turn_Back_Hard
  9. Aerial_Hover
  10. High_Pop

REWARD:
  - GoalReward: 500.0 (Increased the goal reward to prevent any reward hacking)
  - VelocityPlayerToBall: 1.5
  - LiuDistancePlayerToBall: 0.5
  - TouchReward: 5.0
  - AlignBallToGoal: 0.5
  - VelocityBallToGoal: 1.0
  - AerialDistanceReward: 5.0
  - FaceBallReward: 0.05

KEY SETTINGS:
  - 50% task dropout for generalization
  - NoTouchTimeout: 8 seconds
  - Episode timeout: 30 seconds

DURATION: Continue from Phase 3, ~300-400M additional timesteps
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rocket_league_rl'))

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


def build_phase_4_env():
    """
    Phase 4 Environment: The Generalist Agent
    """

    # 1. SPAWN SETUP (1v0)
    plr_mutator = PLRMutator(replay_prob=0.6, blue_size=1, orange_size=0)

    # 2. OBSERVATION (73 dims)
    # TASK DROPOUT: 50% blind training for generalization
    base_obs = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / SIDE_WALL_X, 1 / BACK_NET_Y, 1 / CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / CAR_MAX_SPEED,
        ang_vel_coef=1 / CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0
    )
    obs_builder = PLRObsBuilder(base_obs, task_dropout=0.5)

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

        # G. AERIAL DISTANCE (+5.0)
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
        NoTouchTimeoutCondition(timeout_seconds=8),   # Forces fast engagement
        TimeoutCondition(timeout_seconds=30) # Allows following the ball if agent misses first time
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
        renderer=None  # No rendering during training
    ))


if __name__ == "__main__":
    phase_3_checkpoint = "/Users/mrunmayd/Projects/DeepRL_Project/CSCE-642-Rocket-League-RL/scripts/data/checkpoints/rlgym-ppo-run-1764494333336973000/500030156"

    print("=" * 70)
    print("PHASE 4: THE GENERALIST AGENT")
    print("=" * 70)
    print("Scenarios (10 total):")
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
    if not os.path.exists(phase_3_checkpoint):
        print("\n" + "!" * 70)
        print("ERROR: Checkpoint not found!")
        print("!" * 70)
        print(f"Path: {phase_3_checkpoint}")
        print("")
        print("Please update 'phase_3_checkpoint' with your Phase 3 checkpoint.")
        print("!" * 70)
        sys.exit(1)

    # Configuration
    n_proc = 8 
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = PLRLearner(
        build_phase_4_env,
        wandb_run_name="Phase4_The_Generalist_Agent",
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
        timestep_limit=400_000_000,
        log_to_wandb=True,

        # Load from Phase 3 checkpoint
        checkpoint_load_folder=phase_3_checkpoint,

        # Fresh WandB run
        load_wandb=False
    )

    # Reset timestep counter for fresh WandB graphs
    learner.agent.cumulative_timesteps = 0
    print("\n[INFO] Reset cumulative_timesteps to 0 for fresh WandB graphs")

    print("\n" + "=" * 70)
    print("Starting Phase 4 Training...")
    print("=" * 70 + "\n")

    learner.learn()
