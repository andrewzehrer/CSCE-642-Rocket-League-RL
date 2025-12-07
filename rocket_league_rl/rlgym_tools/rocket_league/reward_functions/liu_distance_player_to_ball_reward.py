from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RADIUS, CAR_MAX_SPEED


class LiuDistancePlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Distance-based reward that incentivizes the agent to GET CLOSER to the ball.

    Formula: exp(-0.5 * dist / CAR_MAX_SPEED)

    Returns:
        1.0: Car touching ball (dist = 0)
        ~0.61: Car 1 car-length away (~2300 uu)
        ~0.37: Car 2 car-lengths away (~4600 uu)
        ~0.08: Car at half-field (~11500 uu)

    Inspired by Liu et al. research on RL in Rocket League environments.
    Used successfully in Necto and other top RL bots.
    """

    def __init__(self, steepness: float = 0.5):
        """
        Args:
            steepness: Controls how quickly reward decays with distance.
                       Higher = steeper decay = more urgency to close distance.
                       Default 0.5 matches original Liu implementation.
                       Try 1.0 for more aggressive "ram the ball" behavior.
        """
        self.steepness = steepness

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        ball = state.ball
        car = state.cars[agent].physics

        # Calculate distance from car to ball surface (not center)
        # Subtract BALL_RADIUS because the inside of the ball is unreachable
        dist = np.linalg.norm(car.position - ball.position) - BALL_RADIUS

        # Clamp to 0 in case car clips inside ball
        dist = max(0.0, dist)

        # Exponential decay: closer = higher reward
        # CAR_MAX_SPEED (~2300) normalizes the distance
        reward = np.exp(-self.steepness * dist / CAR_MAX_SPEED)

        return float(reward)
