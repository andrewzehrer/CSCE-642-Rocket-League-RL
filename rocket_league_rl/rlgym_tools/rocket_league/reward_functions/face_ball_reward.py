from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState


class FaceBallReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for facing toward the ball.

    Returns:
        +1.0: Car facing directly at ball
         0.0: Ball is perpendicular to car (90 degrees to side)
        -1.0: Car facing directly away from ball
    """

    def __init__(self, include_negative_values: bool = True):
        """
        Args:
            include_negative_values: If True, returns [-1, 1]. If False, returns [0, 1].
                                    Set to True to penalize facing away from ball.
        """
        self.include_negative_values = include_negative_values

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        ball = state.ball
        car = state.cars[agent].physics

        # Vector from car to ball
        car_to_ball = ball.position - car.position

        # Normalize (avoid division by zero if car is exactly on ball)
        dist = np.linalg.norm(car_to_ball)
        if dist < 1e-8:
            return 1.0  # If on top of ball, consider it "facing"

        car_to_ball_normalized = car_to_ball / dist

        # Get car's forward direction vector
        car_forward = car.forward

        # Dot product gives cosine of angle between vectors
        # +1.0 = facing directly at ball (0 degrees)
        #  0.0 = ball is to the side (90 degrees)
        # -1.0 = facing directly away (180 degrees)
        dot = float(np.dot(car_forward, car_to_ball_normalized))

        if self.include_negative_values:
            return dot
        return max(0.0, dot)
