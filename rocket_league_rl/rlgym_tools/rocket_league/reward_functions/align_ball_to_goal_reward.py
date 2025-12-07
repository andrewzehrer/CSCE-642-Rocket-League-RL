from typing import Any, Dict, List

import numpy as np
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y


class AlignBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    """
    Rewards the agent for being positioned BEHIND the ball relative to the goal.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        ball = state.ball

        # 1. Determine target goal (opponent's net)
        # Blue team (team_num=0) shoots at orange goal (+BACK_WALL_Y)
        # Orange team (team_num=1) shoots at blue goal (-BACK_WALL_Y)
        target_y = BACK_WALL_Y if car.team_num == 0 else -BACK_WALL_Y

        # Goal position (center of the net)
        goal_pos = np.array([0.0, target_y, 0.0])

        # 2. Vector from Ball to Goal
        ball_to_goal = goal_pos - ball.position
        ball_to_goal_dist = np.linalg.norm(ball_to_goal)
        if ball_to_goal_dist < 1e-8:
            # Ball is in goal, max reward
            return 1.0
        ball_to_goal_dir = ball_to_goal / ball_to_goal_dist

        # 3. Vector from Car to Ball
        car_to_ball = ball.position - car.physics.position
        car_to_ball_dist = np.linalg.norm(car_to_ball)
        if car_to_ball_dist < 1e-8:
            # Car is on ball, consider aligned
            return 1.0
        car_to_ball_dir = car_to_ball / car_to_ball_dist

        # 4. Alignment = Dot Product
        # +1.0 = Car -> Ball -> Goal (perfect shooting position)
        #  0.0 = Car is perpendicular to ball-goal line
        # -1.0 = Goal -> Ball -> Car (blocking/own goal position)
        alignment = float(np.dot(car_to_ball_dir, ball_to_goal_dir))

        return alignment
