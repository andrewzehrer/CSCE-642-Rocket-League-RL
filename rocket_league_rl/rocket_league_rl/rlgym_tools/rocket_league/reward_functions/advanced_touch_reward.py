from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_MAX_SPEED

class AdvancedTouchReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, touch_reward: float = 1.0, acceleration_reward: float = 0.0, use_touch_count: bool = True):
        self.touch_reward = touch_reward
        self.acceleration_reward = acceleration_reward
        self.use_touch_count = use_touch_count

        self.prev_ball = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball = initial_state.ball

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {agent: 0 for agent in agents}
        ball = state.ball
        for agent in agents:
            touches = state.cars[agent].ball_touches

            if touches > 0:
                if not self.use_touch_count:
                    touches = 1
                acceleration = np.linalg.norm(ball.linear_velocity - self.prev_ball.linear_velocity) / BALL_MAX_SPEED
                rewards[agent] += self.touch_reward * touches
                rewards[agent] += acceleration * self.acceleration_reward

        self.prev_ball = ball

        return rewards
    
    
class TouchScaler(RewardFunction[AgentID, GameState, float]):
    def __init__(self, base_reward_fn):
        self.base = base_reward_fn
        self.last_touches = {}

    def reset(self, agents, initial_state, shared_info):
        self.base.reset(agents, initial_state, shared_info)
        self.last_touches = {agent: 0 for agent in agents}

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        base_rewards = self.base.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        scaled = {}

        for agent in agents:
            touches = state.cars[agent].ball_touches
            last = self.last_touches[agent]

            if touches > last:
                if touches == 1:
                    scaled[agent] = 8.0  # first touch
                else:
                    scaled[agent] = 4.0  # subsequent touches
            else:
                scaled[agent] = 0.0

            self.last_touches[agent] = touches

        return scaled
