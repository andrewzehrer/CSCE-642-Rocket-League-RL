# ==============================================================================
# ROCKET LEAGUE RL AGENT SIMULATOR
# ==============================================================================
# Visualize trained RL agents (trained on 1v0) using RLViser renderer.
#
# TWO SIMULATION MODES:
#   1. "scenarios" - Test agent in 1v0 PLR scenarios
#   2. "1v1"       - Watch agent play against itself in 1v1 self-play
#
# Usage:
#   python3 simulate.py                                # Default (scenarios mode)
#   python3 simulate.py --mode 1v1                     # 1v1 self-play
#   python3 simulate.py --mode scenarios               # Cycle through scenarios
#   python3 simulate.py --speed 2                      # 2x speed
#   python3 simulate.py --speed 0                      # Max speed (no delay)
#   python3 simulate.py --checkpoint /path/to/ckpt     # Specific checkpoint
# ==============================================================================

import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rocket_league_rl'))

import time
import numpy as np
import torch
from rlgym.api import RLGym
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_NET_Y, CEILING_Z, CAR_MAX_SPEED, CAR_MAX_ANG_VEL
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.done_conditions import GoalCondition
from rocket_league_rl.rlgym_ppo.util import RLGymV2GymWrapper

from plr_utils import PLRMutator, PLRObsBuilder
from plr_learner import PLRLearner

# ==============================================================================
# ENVIRONMENT BUILDERS
# ==============================================================================

def build_scenario_env(force_blind=True, render=True):
    """
    1v0 Environment with PLR scenarios.

    Used for:
      - Loading agents (render=False)
      - Scenario review mode (render=True)

    Args:
        force_blind: Always send ID=0.0 (test generalization)
        render: Enable RLViser renderer
    """
    plr_mutator = PLRMutator(replay_prob=1.0, blue_size=1, orange_size=0)

    base_obs = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1/SIDE_WALL_X, 1/BACK_NET_Y, 1/CEILING_Z]),
        ang_coef=1/np.pi,
        lin_vel_coef=1/CAR_MAX_SPEED,
        ang_vel_coef=1/CAR_MAX_ANG_VEL,
        boost_coef=1/100.0
    )
    obs_builder = PLRObsBuilder(base_obs, force_blind=force_blind)

    env = RLGym(
        state_mutator=plr_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=GoalReward(),
        termination_cond=GoalCondition(),
        truncation_cond=None,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer() if render else None
    )

    if render:
        return env, plr_mutator
    return RLGymV2GymWrapper(env)

def build_1v1_env():
    """
    1v1 Environment for self-play.

    Both Blue and Orange use the same policy.
    Observations are 93 dims but sliced to 73 for agents trained on 1v0.
    """
    spawn = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    kickoff = KickoffMutator()

    base_obs = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1/SIDE_WALL_X, 1/BACK_NET_Y, 1/CEILING_Z]),
        ang_coef=1/np.pi,
        lin_vel_coef=1/CAR_MAX_SPEED,
        ang_vel_coef=1/CAR_MAX_ANG_VEL,
        boost_coef=1/100.0
    )
    obs_builder = PLRObsBuilder(base_obs)

    return RLGym(
        state_mutator=MutatorSequence(spawn, kickoff),
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=GoalReward(),
        termination_cond=GoalCondition(),
        truncation_cond=None,
        transition_engine=RocketSimEngine(),
        renderer=RLViserRenderer()
    )

# ==============================================================================
# AGENT LOADER
# ==============================================================================
def _build_loader_env():
    """Module-level function for loading agents."""
    return build_scenario_env(force_blind=False, render=False)

def load_agent(path):
    """Load agent from checkpoint (73-dim observations)."""
    print(f"Loading agent from: {path}")

    learner = PLRLearner(
        env_create_function=_build_loader_env,
        n_proc=1, min_inference_size=1,
        policy_layer_sizes=[512, 512], critic_layer_sizes=[512, 512],
        ppo_ent_coef=0.01, policy_lr=1e-4, critic_lr=1e-4,
        ppo_batch_size=1000, ts_per_iteration=1000,
        exp_buffer_size=1000, ppo_minibatch_size=1000,
        log_to_wandb=False, checkpoint_load_folder=None
    )
    learner.load(path, load_wandb=False)
    agent = learner.agent
    agent.policy.eval()
    return agent

# ==============================================================================
# SIMULATION LOOPS
# ==============================================================================
def run_1v1(agent, speed=1.0):
    """
    1v1 self-play: Both Blue and Orange use the same policy.
    Observations sliced to 73 dims (agents trained on 1v0 are blind to opponent).
    """
    env = build_1v1_env()
    sleep_time = 0.05 / speed if speed > 0 else 0

    print("\n" + "="*70)
    print("1v1 SELF-PLAY")
    print("="*70)
    print("Both cars use the SAME policy")
    print(f"Speed: {speed}x" + (" (max)" if speed == 0 else ""))
    print("="*70 + "\n")

    obs_dict = env.reset()
    blue_score, orange_score = 0, 0
    episode = 1

    try:
        while True:
            action_dict = {}
            for agent_id, obs in obs_dict.items():
                # Slice to 73 dims: [ball+car (72)] + [scenario_idx=0.0 (1)]
                if len(obs) == 93:
                    obs_final = np.concatenate([obs[:72], np.array([0.0], dtype=np.float32)])
                else:
                    obs_final = obs

                with torch.no_grad():
                    action, _ = agent.policy.get_action(obs_final, deterministic=True)
                action_dict[agent_id] = np.array([int(action)])

            step_ret = env.step(action_dict)
            obs_dict = step_ret[0]
            term = step_ret[2]

            env.render()
            if sleep_time > 0:
                time.sleep(sleep_time)

            if any(term.values()):
                if env.state.scoring_team == 0:
                    blue_score += 1
                    print(f"Episode {episode}: BLUE SCORES! ({blue_score} - {orange_score})")
                elif env.state.scoring_team == 1:
                    orange_score += 1
                    print(f"Episode {episode}: ORANGE SCORES! ({blue_score} - {orange_score})")
                else:
                    print(f"Episode {episode}: RESET")
                episode += 1
                obs_dict = env.reset()

    except KeyboardInterrupt:
        print(f"\nFINAL: Blue {blue_score} - {orange_score} Orange ({episode-1} episodes)")
    finally:
        env.close()

def run_scenarios(agent, speed=1.0, force_blind=True):
    """
    Cycle through all scenarios in 1v0 mode.
    """
    env, mutator = build_scenario_env(force_blind=force_blind, render=True)
    scenarios = mutator.scenarios
    sleep_time = 0.05 / speed if speed > 0 else 0

    print("\n" + "="*70)
    print(f"SCENARIO REVIEW ({len(scenarios)} Scenarios)")
    print("="*70)
    print("Cycling through PLR scenarios in 1v0 mode")
    print(f"Speed: {speed}x" + (" (max)" if speed == 0 else ""))
    print("="*70 + "\n")

    current_idx = 0

    try:
        while True:
            mutator.forced_idx = current_idx
            scenario_name = scenarios[current_idx].name
            print(f"\n>>> SCENARIO {current_idx + 1}/{len(scenarios)}: {scenario_name} <<<")

            obs_dict = env.reset()
            steps = 0
            done = False

            while not done and steps < 450:  # 30 seconds max
                action_dict = {}
                for agent_id, obs in obs_dict.items():
                    with torch.no_grad():
                        action, _ = agent.policy.get_action(obs, deterministic=True)
                    action_dict[agent_id] = np.array([int(action)])

                step_ret = env.step(action_dict)
                obs_dict = step_ret[0]
                term = step_ret[2]

                env.render()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                steps += 1

                if any(term.values()):
                    print(f"GOAL in {steps} steps!")
                    done = True

            if not done:
                print(f"Timeout ({steps} steps)")

            current_idx = (current_idx + 1) % len(scenarios)
            if speed > 0:
                time.sleep(0.5 / speed)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rocket League RL Agent Simulator')

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "data/checkpoints/rlgym-ppo-run-1764611773436919000/150009064")

    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--mode', type=str, default='scenarios', choices=['1v1', 'scenarios'])
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--blind', action='store_true', default=True)

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ROCKET LEAGUE RL SIMULATOR")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    print(f"Speed: {args.speed}x")
    print("="*70)

    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    agent = load_agent(args.checkpoint)

    if args.mode == "1v1":
        run_1v1(agent, speed=args.speed)
    else:
        run_scenarios(agent, speed=args.speed, force_blind=args.blind)
