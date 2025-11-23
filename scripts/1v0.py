from rocket_league_rl.rlgym.api import RLGym
from rocket_league_rl.rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rocket_league_rl.rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
from rocket_league_rl.rlgym.rocket_league.obs_builders import DefaultObs
from rocket_league_rl.rlgym.rocket_league.reward_functions import CombinedReward
from rocket_league_rl.rlgym.rocket_league.sim import RocketSimEngine
from rocket_league_rl.rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rocket_league_rl.rlgym.rocket_league import common_values
from rocket_league_rl.rlgym.rocket_league.rlviser.rlviser_renderer import RLViserRenderer
import numpy as np

# Modified classes
from rocket_league_rl.rlgym_ppo import Learner
from rocket_league_rl.rlgym_ppo.util import RLGymV2GymWrapper
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.ball_travel_reward import BallTravelReward

def build_rlgym_v2_env():
    spawn_opponents = False
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds), 
        TimeoutCondition(timeout_seconds=game_timeout_seconds)
    )

    reward_fn = CombinedReward(
        (AdvancedTouchReward(), 10.0),
        (BallTravelReward(), 1.0)
    )

    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator()
    )

    renderer = RLViserRenderer()
    
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=renderer
    )
    
    return RLGymV2GymWrapper(rlgym_env)

if __name__ == "__main__":
    
    n_proc = 8

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rlgym_v2_env,
        wandb_run_name="learning_to_hit_ball",
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=None,
        render=True,
        render_last_only=True,
        ppo_batch_size=50000, # batch size - set this number to as large as your GPU can handle
        policy_layer_sizes=[512, 512], # policy network
        critic_layer_sizes=[512, 512], # value network
        ts_per_iteration=50000, # timesteps per training iteration - set this equal to the batch size
        exp_buffer_size=150000, # size of experience buffer - keep this 2 - 3x the batch size
        ppo_minibatch_size=50000, # minibatch size - set this less than or equal to the batch size
        ppo_ent_coef=0.01, # entropy coefficient - this determines the impact of exploration on the policy
        policy_lr=5e-5, # policy learning rate
        critic_lr=5e-5, # value function learning rate
        ppo_epochs=1,   # number of PPO epochs
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=50_000, # save every 50K steps
        timestep_limit=300_000, # Train for 100K steps
        log_to_wandb=False
    )

    learner.learn()