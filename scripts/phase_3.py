"""
==================== PHASE 3: THE AERIAL AGENT ====================

GOAL: Teach the agent to touch the ball in the air

SCENARIOS USED:
  - Previous ground scenarios so that the agent does not lose it ground abilities and Touch scenarios where ball is floating in the air infront of the agent

REWARD:
  - GoalReward: 100.0
  - AerialDistanceReward: 20.0 -> New Reward introduced
  - TouchReward: 2.0
  - VelocityBallToGoalReward: 1.0
  - AlignBallToGoalReward: 0.1
  - VelocityPlayerToBallReward: 0.1
  - LiuDistancePlayerToBallReward: 0.05
  - FaceBallReward: 0.05

DURATION: Continued from phase 2.5, ~500M additional timesteps
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
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator
from rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer
from rocket_league_rl.rlgym_ppo.util import RLGymV2GymWrapper

# Import Custom Rewards
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.velocity_ball_to_goal_reward import VelocityBallToGoalReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.liu_distance_player_to_ball_reward import LiuDistancePlayerToBallReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.align_ball_to_goal_reward import AlignBallToGoalReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.face_ball_reward import FaceBallReward

# Aerial Reward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.aerial_distance_reward import AerialDistanceReward

# PLR Imports
from plr_utils import PLRMutator, PLRObsBuilder
from plr_learner import PLRLearner


def build_phase_3_env():
    """
    Phase 3 Environment: The Aerial Agent
    """

    # 1. SPAWN SETUP
    spawn_cars = FixedTeamSizeMutator(blue_size=1, orange_size=0)
    plr_mutator = PLRMutator(replay_prob=0.6)
    state_mutator = MutatorSequence(spawn_cars, plr_mutator)

    # 2. OBSERVATION (73 dims)
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
        # A. GOAL (+100)
        (GoalReward(), 100.0),

        # B. AERIAL DISTANCE REWARD (+20)
        # Weight 20.0: Reward aerial touches
        # touch_height_weight=1.0: Reward first touch by height
        # car_distance_weight=0.5: Reward car movement for consecutive touches
        # ball_distance_weight=0.5: Reward ball movement for consecutive touches
        (AerialDistanceReward(
            touch_height_weight=1.0,
            car_distance_weight=0.5,
            ball_distance_weight=0.5
        ), 20.0),

        # C. GROUND TOUCH (+2)
        (TouchReward(), 2.0),

        # D. BALL VELOCITY TO GOAL (+1.0) - Shooting towards the goal reward
        (VelocityBallToGoalReward(), 1.0),

        # E. ALIGNMENT (+0.1) - Positioning hint
        (AlignBallToGoalReward(), 0.1),

        # F. LIU DISTANCE (+0.05) - Closeness reward
        (LiuDistancePlayerToBallReward(), 0.05),

        # G. VELOCITY TO BALL (+0.1) - Following the ball reward
        (VelocityPlayerToBallReward(), 0.1),

        # H. FACE BALL (+0.05) - Orientation signal
        (FaceBallReward(), 0.05),
    )

    # 4. TERMINATION CONDITIONS
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=10),
        TimeoutCondition(timeout_seconds=20)
    )

    # 5. BUILD ENV
    return RLGymV2GymWrapper(RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    ))


if __name__ == "__main__":
    phase_2_5_checkpoint = "/Users/mrunmayd/Projects/DeepRL_Project/CSCE-642-Rocket-League-RL/scripts/data/checkpoints/rlgym-ppo-run-1764480294799738000/150008963"

    print("=" * 70)
    print("PHASE 3: THE AERIAL AGENT")
    print("=" * 70)
    print("Scenarios (6 total):")
    print("  Ground Scenarios:")
    print("    1. Kickoff")
    print("    2. Stationary_Turn (Hard)")
    print("    3. Recovery_Offense")
    print("    4. Wall_Drill")
    print("  Aerial Scenarios:")
    print("    5. Aerial_Hover")
    print("    6. High_Pop")
    print("=" * 70)

    # Check checkpoint path
    if not os.path.exists(phase_2_5_checkpoint):
        print("\n" + "!" * 70)
        print("ERROR: Checkpoint path not found!")
        print("!" * 70)
        print(f"Path: {phase_2_5_checkpoint}")
        print("")
        print("Please update 'phase_2_5_checkpoint' with your Phase 2.5 checkpoint path.")
        sys.exit(1)

    # Configuration
    n_proc = 8
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = PLRLearner(
        build_phase_3_env,
        wandb_run_name="Phase3_TheAerialAgent",
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

        checkpoint_load_folder=phase_2_5_checkpoint,

        # Fresh WandB run
        load_wandb=True
    )

    # Reset timestep counter for fresh WandB graphs
    learner.agent.cumulative_timesteps = 0
    print("\n[INFO] Reset cumulative_timesteps to 0 for fresh WandB graphs")

    print("\n" + "=" * 70)
    print("Starting Phase 3 Training ...")
    print("=" * 70 + "\n")

    learner.learn()
