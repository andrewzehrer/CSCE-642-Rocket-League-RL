"""
==================== PHASE 2.5: THE GOAL SCORER 2.0 ====================

GOAL: Teach agent to score goals efficiently by positioning behind the ball.

HISTORY (Combined Phase 2.5 to Phase 2.8):
  - Phase 2.5: Tried fixing not following the ball after missing the first shot
  - Phase 2.6: Added turn scenarios to help learn the agent to follow the ball
  - Phase 2.7: Fixed the reward hacking in the previous scenario 
  - Phase 2.75: Tried fixing the alignment of the ball 
  - Phase 2.8: The Final Version: Fixed the alignment issue by introducing a new reward

PROBLEMS SOLVED:
  - Ball hugging: Agent stayed close to ball instead of shooting
  - Own goals: Agent didn't understand positioning relative to goal
  - Circling: Agent orbited ball instead of approaching directly

SOLUTION:
  1. AlignBallToGoalReward (0.1): Forces positioning behind ball
  2. VelocityBallToGoalReward (1.0): Prevents shooting toward own goal and reward shooting towards opponents goals
  3. LiuDistancePlayerToBall (0.05): Forces the agent to follow the ball
  4. Timeout reduced (20s): Removes reward hacking opportunity

SCENARIOS:
  - Goal-scoring scenarios with various angles and distances

REWARD:
  - GoalReward: 100.0
  - VelocityBallToGoalReward: 1.0
  - TouchReward: 2.0
  - AlignBallToGoalReward: 0.1
  - LiuDistancePlayerToBall: 0.05
  - VelocityPlayerToBallReward: 0.1
  - FaceBallReward: 0.05

DURATION: Continue from Phase 2, ~500M additional timesteps
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
from rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer
from rocket_league_rl.rlgym_ppo.util import RLGymV2GymWrapper

# Import Custom Rewards
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.velocity_ball_to_goal_reward import VelocityBallToGoalReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.liu_distance_player_to_ball_reward import LiuDistancePlayerToBallReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.align_ball_to_goal_reward import AlignBallToGoalReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.face_ball_reward import FaceBallReward

# PLR Imports
from plr_utils import PLRMutator, PLRObsBuilder
from plr_learner import PLRLearner


def build_phase_2_5_env():
    """
    Phase 2.5 Environment: The Goal Scorer 2.0
    """

    # 1. SPAWN SETUP
    # PLRMutator handles spawning internally
    plr_mutator = PLRMutator(replay_prob=0.6, blue_size=1, orange_size=0)

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

    # 3. REWARD FUNCTION - THE GOAL SCORER 2.0
    reward_fn = CombinedReward(
        # A. GOAL (+100)
        (GoalReward(), 100.0),

        # B. ALIGNMENT (+0.1) - Positioning reward
        # +reward when behind ball
        # -reward when in front of ball
        (AlignBallToGoalReward(), 0.1),

        # C. BALL VELOCITY TO GOAL (+1.0) - Shooting towards the goal reward
        (VelocityBallToGoalReward(), 1.0),

        # D. TOUCH (+2.0) - Contact reward
        (TouchReward(), 2.0),

        # E. LIU DISTANCE (+0.05) - Closeness reward
        (LiuDistancePlayerToBallReward(), 0.05),

        # F. VELOCITY TO BALL (+0.1) - Following the ball reward
        (VelocityPlayerToBallReward(), 0.1),

        # G. FACE BALL (+0.05) - Orientation signal
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
        state_mutator=plr_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    ))


if __name__ == "__main__":
    phase_2_checkpoint = "/Users/mrunmayd/Projects/DeepRL_Project/CSCE-642-Rocket-League-RL/scripts/data/checkpoints/rlgym-ppo-run-1764280509226264000/400024067"

    print("=" * 70)
    print("PHASE 2.5: THE GOAL SCORER 2.0")
    print("=" * 70)

    # Check checkpoint path
    if not os.path.exists(phase_2_checkpoint):
        print("\n" + "!" * 70)
        print("ERROR: Checkpoint path not found!")
        print("!" * 70)
        print(f"Path: {phase_2_checkpoint}")
        print("")
        print("Please update 'phase_2_checkpoint' with your Phase 2 checkpoint path.")
        print("!" * 70)
        sys.exit(1)

    # Configuration
    n_proc = 8
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = PLRLearner(
        build_phase_2_5_env,
        wandb_run_name="Phase2.5_TheGoalScorer2.0",
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

        # Load from Phase 2 checkpoint
        checkpoint_load_folder=phase_2_checkpoint,

        # Fresh WandB run
        load_wandb=False
    )

    # Reset timestep counter for fresh WandB graphs
    learner.agent.cumulative_timesteps = 0
    print("\n[INFO] Reset cumulative_timesteps to 0 for fresh WandB graphs")

    print("\n" + "=" * 70)
    print("Starting Phase 2.5 Training (The Goal Scorer 2.0)...")
    print("=" * 70 + "\n")

    learner.learn()
