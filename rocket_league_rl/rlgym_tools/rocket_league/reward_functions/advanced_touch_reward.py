from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_MAX_SPEED

class AdvancedTouchReward(RewardFunction[AgentID, GameState, float]):
    """
    Advanced reward combining:
        - Touch count reward: +8 for first touch, +4 for subsequent touches
        - Good touch reward: +5 if ball accelerated toward opponent goal
        - Optional acceleration reward
    """

    def __init__(
            self, 
            touch_reward: float = 4.0, 
            acceleration_reward: float = 0.0, 
            good_touch_reward: float = 5.0, 
            use_touch_count: bool = True, 
            tick_rate: float = 120.0,
            cooldown_seconds: float = 0.5
        ):

        self.use_touch_count = use_touch_count
        self.touch_reward = touch_reward
        self.acceleration_reward = acceleration_reward
        self.good_touch_reward = good_touch_reward
        self.tick_rate = tick_rate  # RLgym sim steps per second
        self.cooldown_seconds = cooldown_seconds

        self.prev_ball = None
        self.last_touches = {}      # last touch count per agent
        self.last_touch_frame = {}  # last rewarded frame per agent
        self.frame = 0              # simulation step counter

    def reset(
            self, 
            agents: List[AgentID], 
            initial_state: GameState, 
            shared_info: Dict[str, Any]
        ) -> None:
        
        self.prev_ball = initial_state.ball
        self.last_touches = {agent: 0 for agent in agents}
        self.last_touch_frame = {agent: -9999 for agent in agents}
        self.frame = 0

    def get_rewards(
            self,
            agents: List[AgentID],
            state: GameState,
            is_terminated: Dict[AgentID, bool],
            is_truncated: Dict[AgentID, bool],
            shared_info: Dict[str, Any]
        ) -> Dict[AgentID, float]:

        rewards = {agent: 0.0 for agent in agents}
        ball = state.ball

        for agent in agents:
            car = state.cars[agent]
            touches = car.ball_touches
            last_touch_count = self.last_touches.get(agent, 0)
            last_touch_frame = self.last_touch_frame.get(agent, -9999)

            # --- Check if this touch is new and outside cooldown ---
            if touches > last_touch_count and (self.frame - last_touch_frame) >= self.cooldown_seconds * self.tick_rate:
                # Touch reward
                if touches == 1:
                    rewards[agent] += (2 * self.touch_reward)
                else:
                    rewards[agent] += self.touch_reward

                # Good touch reward
                if car.team_num == 0:
                    goal_vector = np.array([0, 5120, 0]) - ball.position
                else:
                    goal_vector = np.array([0, -5120, 0]) - ball.position

                goal_dir_norm = goal_vector / np.linalg.norm(goal_vector)
                ball_vel_change = ball.linear_velocity - self.prev_ball.linear_velocity
                contribution = np.dot(ball_vel_change, goal_dir_norm)
                # print(f"Contribution: {contribution}")

                if contribution > 0:
                    rewards[agent] += self.good_touch_reward

                # Update last touch frame
                self.last_touch_frame[agent] = self.frame

            # --- Acceleration reward ---
            ball_acc_change = np.linalg.norm(ball.linear_velocity - self.prev_ball.linear_velocity)
            rewards[agent] += (ball_acc_change / BALL_MAX_SPEED) * self.acceleration_reward

            # --- Update last touch count ---
            self.last_touches[agent] = touches

        # Update previous ball state
        self.prev_ball = ball
        self.frame += 1

        return rewards