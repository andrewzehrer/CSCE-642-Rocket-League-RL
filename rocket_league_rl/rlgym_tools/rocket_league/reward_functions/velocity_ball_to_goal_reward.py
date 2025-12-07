from typing import Any, Dict, List
import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_MAX_SPEED, BACK_WALL_Y


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards ball velocity toward the opponent's goal.
    This provides a continuous reward signal BEFORE goals are scored,
    guiding the agent to hit the ball in the right direction.
    """
    def __init__(self, own_goal_penalty: bool = True):
        """
        Args:
            own_goal_penalty: If True, penalizes ball moving toward own goal (can be negative).
                            If False, only gives positive rewards for correct direction.
        """
        self.own_goal_penalty = own_goal_penalty

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        ball = state.ball

        for agent in agents:
            car = state.cars[agent]

            # Determine which goal is the target
            # Blue team (team_num=0) shoots at orange goal (+Y direction)
            # Orange team (team_num=1) shoots at blue goal (-Y direction)
            if car.team_num == 0:  # Blue team
                # Target is orange goal at +BACK_WALL_Y
                vel_toward_goal = ball.linear_velocity[1]  # Positive Y velocity is good
            else:  # Orange team
                # Target is blue goal at -BACK_WALL_Y
                vel_toward_goal = -ball.linear_velocity[1]  # Negative Y velocity is good

            # Normalize by max ball speed to keep reward in [-1, 1] range
            norm_vel = vel_toward_goal / BALL_MAX_SPEED

            if self.own_goal_penalty:
                # Allow negative rewards for shooting wrong direction
                rewards[agent] = norm_vel
            else:
                # Only positive rewards
                rewards[agent] = max(0, norm_vel)

        return rewards
