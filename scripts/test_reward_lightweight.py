import numpy as np
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward

# --- Dummy classes to simulate RLGym objects ---
class DummyBall:
    def __init__(self):
        self.position = np.zeros(3)
        self.linear_velocity = np.zeros(3)

    def copy(self):
        new_ball = DummyBall()
        new_ball.position = self.position.copy()
        new_ball.linear_velocity = self.linear_velocity.copy()
        return new_ball

class DummyCar:
    def __init__(self, team_num=0):
        self.team_num = team_num
        self.ball_touches = 0

class DummyState:
    def __init__(self):
        self.ball = DummyBall()
        self.cars = {0: DummyCar(team_num=0)}

# --- Setup reward function ---
reward_fn = AdvancedTouchReward(
    touch_reward=1.0,
    acceleration_reward=0.0,
    good_touch_reward=5.0,
    tick_rate=10.0  # low tick_rate for quick testing
)

agents = [0]
state = DummyState()
reward_fn.reset(agents, state, {})

# --- Step through frames and simulate touches ---
for frame in range(10):
    # Simulate touches
    if frame == 3 or frame == 6:
        car = state.cars[0]
        car.ball_touches += 1

        # Simulate ball velocity toward opponent goal
        state.ball.linear_velocity = np.array([0, 1000, 0])

        # Make sure prev_ball is a separate copy
        reward_fn.prev_ball = state.ball.copy()
        reward_fn.prev_ball.linear_velocity[:] = 0  # previous frame velocity = 0

    rewards = reward_fn.get_rewards(agents, state, {}, {}, {})
    print(f"Frame {frame}:  num_touches={state.cars[0].ball_touches}, reward={rewards[0]}")
