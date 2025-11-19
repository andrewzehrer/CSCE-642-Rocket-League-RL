import copy
import numpy as np
from typing import Dict

from rlgym.rocket_league.api.car import Car
from rlgym.rocket_league.api.game_state import GameState
from rlgym.rocket_league.api.physics_object import PhysicsObject
from rocket_league_rl.rlgym_tools.rocket_league.reward_functions.advanced_touch_reward import AdvancedTouchReward

# Constants
BALL_MAX_SPEED = 6000  # Rocket League ball max speed
DT = 1/120.0
TICKS = 120 * 60 # 60 seconds
SEED = 42
np.random.seed(SEED)

# --- Initialize PhysicsObject (Ball) ---
ball = PhysicsObject()
ball.position = np.zeros(3)
ball.linear_velocity = np.zeros(3)
ball.angular_velocity = np.zeros(3)

# --- Initialize a Car ---
car = Car()
car.team_num = 0
car.ball_touches = 0
car.physics = PhysicsObject()
car.physics.position = np.zeros(3)
car.physics.linear_velocity = np.zeros(3)
car.physics.angular_velocity = np.zeros(3)

# --- GameState ---
state = GameState()
state.tick_count = 0
state.goal_scored = False
state.cars = {0: car}
state.ball = ball

# --- Reward function ---
reward_fn = AdvancedTouchReward(
    touch_reward=1.0,
    acceleration_reward=0.1,
    good_touch_reward=5.0,
    tick_rate=120.0
)
reward_fn.reset([0], state, {})

# --- Track no-touch timeout ---
timeout_counter = 0
timeout_frames = 30 * 120  # 30 seconds

# 0.1% chance to touch ball every frame
touch_chance = 0.01  # 0.1%

# --- Simulation loop ---
for frame in range(TICKS):
    state.tick_count = frame
    
    # Check if this frame triggers a touch
    if np.random.rand() < touch_chance:
        # Apply impulse toward opponent goal
        goal_vector = np.array([0, 5120, 0]) - ball.position
        goal_dir = goal_vector / np.linalg.norm(goal_vector)
        ball.linear_velocity += goal_dir * 1000  # impulse
        car.ball_touches += 1

        # Reset prev_ball for reward calculation
        reward_fn.prev_ball = copy.deepcopy(ball)

        timeout_counter = 0
    else:
        timeout_counter += 1

    # Step ball physics (simplified: linear only)
    ball.position += ball.linear_velocity * DT
    speed = np.linalg.norm(ball.linear_velocity)
    if speed > BALL_MAX_SPEED:
        ball.linear_velocity = ball.linear_velocity / speed * BALL_MAX_SPEED

    # Compute rewards
    rewards: Dict[int, float] = reward_fn.get_rewards([0], state, {}, {}, {})
    if rewards[0] > 0:
        print(f"Frame {frame}:  touches={car.ball_touches}, reward={rewards[0]:.2f}, ball_vel={ball.linear_velocity}")

    # Termination conditions
    if timeout_counter >= timeout_frames:
        print("No-touch timeout triggered")
        break