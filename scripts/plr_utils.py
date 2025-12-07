"""
PLR (Prioritized Level Replay) Setup for Rocket League RL Training
=======================================================================

This module provides setup for curriculum learning and prioritized level replay for RL Training

Components:
  - Scenario: Defines training setups (ball/car positions)
  - PLRBuffer: Tracks scenario difficulty scores
  - PLRMutator: Selects scenarios using Prioritized Level Replay
  - PLRObsBuilder: Appends scenario ID to observations (with task dropout)


================================================================================
                        TRAINING PROGRESSION OVERVIEW
================================================================================

PHASE 1: THE TOUCH AGENT
---------------------------------
Goal: Learn basic ball chasing and steering
Scenarios:
  - Touch_Close     (touch_distance, 300-800 units)
  - Touch_Mid       (touch_distance, 800-2000 units)
  - Touch_Far       (touch_distance, 2000-4000 units)
  - Steering_Easy   (touch_steering, offset 100-500)
  - Steering_Hard   (touch_steering, offset 500-1500)


PHASE 2: THE GOAL SCORER
---------------------------------
Goal: Learn to score on stationary balls
Scenarios:
  - Basic_Kickoff   (kickoff)
  - Tap_In          (tap_in)
  - Open_Net        (open_net)
  - Center_Shot     (center_shot)


PHASE 2.5: THE GOAL SCORER 2.0
----------------------------------
Goal: Handle moving balls, rebounds and ball behind the car
Scenarios:
  - Fast_Rolling        (fast_rolling)
  - Angled_Shot_Left    (angled_shot)
  - Angled_Shot_Right   (angled_shot)
  - Rebound_Chase       (rebound_chase)
  - Recovery_Defense    (recovery_drill, mode: defense)
  - Recovery_Offense    (recovery_drill, mode: offense)
  - Stationary_Turn     (stationary_turn) 


PHASE 3: THE AERIAL AGENT
-----------------------------------------------------
Goal: Jump to hit elevated balls
Scenarios:
  - Kickoff             (kickoff)
  - Stationary_Turn     (stationary_turn)
  - Recovery_Offense    (recovery_offense)
  - Wall_Drill          (wall_drill)
  - Aerial_Hover        (aerial_hover)
  - High_Pop            (high_pop)


PHASE 4: THE GENERALIST AGENT
---------------------------------------------
Goal: Remove Scenario Id Dependence (50% task dropout)
Scenarios:
  1. Kickoff
  2. Recovery_Defense
  3. Rebound_Chase
  4. Wall_Shot_Left
  5. Wall_Shot_Right
  6. Fast_Rolling
  7. Turn_Back_Easy
  8. Turn_Back_Hard
  9. Aerial_Hover
  10. High_Pop

  
PHASE 5: SELF-PLAY 1v1
----------------------
Goal: Competitive play against self
Mode: 1v1 (orange car mirrored across field)
Same 10 scenarios as Phase 4.5


================================================================================
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rocket_league_rl'))

import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from rlgym.api import StateMutator, ObsBuilder
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_NET_Y, CEILING_Z, BALL_RADIUS
from rlgym.rocket_league.state_mutators import KickoffMutator, FixedTeamSizeMutator

PLR_FILE = "plr_state.json"

# =============================================================================
# SCENARIO DEFINITION
# =============================================================================
@dataclass
class Scenario:
    """Defines a training scenario with name, type, and configuration."""
    name: str
    type: str
    config: Dict[str, Any]

# =============================================================================
# CURRENT SCENARIOS
# =============================================================================
def get_scenarios() -> List[Scenario]:
    """
    Returns the current scenarios used.
    """
    return [
        # 1. KICKOFF -  For 1v1 skill and Scenario id 0.0 (default)
        Scenario("Kickoff", "kickoff", {}),

        # 2. RECOVERY_DEFENSE - Ball toward own goal
        Scenario("Recovery_Defense", "recovery_defense", {}),

        # 3. REBOUND_CHASE - Ball bouncing off wall
        Scenario("Rebound_Chase", "rebound_chase", {}),

        # 4-5. WALL SHOTS - Ball near wall at moderate height
        Scenario("Wall_Shot_Left", "wall_shot", {"side": "left"}),
        Scenario("Wall_Shot_Right", "wall_shot", {"side": "right"}),

        # 6. FAST_ROLLING - Fast moving ball
        Scenario("Fast_Rolling", "fast_rolling", {}),

        # 7-8. TURN-BACK - Ball behind car, 180-degree turns
        Scenario("Turn_Back_Easy", "turn_back_easy", {}),
        Scenario("Turn_Back_Hard", "turn_back_hard", {}),

        # 9-10. AERIALS - Ball low, car close
        Scenario("Aerial_Hover", "aerial_hover", {}),
        Scenario("High_Pop", "high_pop", {}),
    ]

# =============================================================================
# SCENARIOS USED PER PHASE
# =============================================================================
def get_phase_1_scenarios() -> List[Scenario]:
    return [
        # Distance-based touch scenarios
        Scenario("Touch_Close", "touch_distance", {"min_dist": 300, "max_dist": 800}),
        Scenario("Touch_Mid", "touch_distance", {"min_dist": 800, "max_dist": 2000}),
        Scenario("Touch_Far", "touch_distance", {"min_dist": 2000, "max_dist": 4000}),

        # Steering scenarios (ball offset from car's forward direction)
        Scenario("Steering_Easy", "touch_steering", {"min_off": 100, "max_off": 500}),
        Scenario("Steering_Hard", "touch_steering", {"min_off": 500, "max_off": 1500}),
    ]

def get_phase_2_scenarios() -> List[Scenario]:
    return [
        Scenario("Basic_Kickoff", "kickoff", {}),
        Scenario("Tap_In", "tap_in", {}),
        Scenario("Open_Net", "open_net", {}),
        Scenario("Center_Shot", "center_shot", {}),
    ]

def get_phase_2_5_scenarios() -> List[Scenario]:
    return [
        Scenario("Fast_Rolling", "fast_rolling", {}),
        Scenario("Angled_Shot_Left", "angled_shot", {"side": "left"}),
        Scenario("Angled_Shot_Right", "angled_shot", {"side": "right"}),
        Scenario("Rebound_Chase", "rebound_chase", {}),
        Scenario("Recovery_Defense", "recovery_drill", {"mode": "defense"}),
        Scenario("Recovery_Offense", "recovery_drill", {"mode": "offense"}),
    ]

def get_phase_3_scenarios() -> List[Scenario]:
    return [
        Scenario("Kickoff", "kickoff", {}),
        Scenario("Stationary_Turn", "stationary_turn", {"min_angle": 135, "max_angle": 180}),
        Scenario("Recovery_Offense", "recovery_offense", {}),
        Scenario("Wall_Drill", "wall_drill", {}),
        Scenario("Aerial_Hover", "aerial_hover", {}),
        Scenario("High_Pop", "high_pop", {}),
    ]

# =============================================================================
# PLR BUFFER - Tracks scenario difficulty scores
# =============================================================================
class PLRBuffer:
    """
    Tracks TD error (difficulty) scores for each scenario.
    Higher error = harder scenario = more likely to be sampled.
    """

    def __init__(self, capacity=2000, num_scenarios=None):
        self.capacity = capacity
        self.scores = {}
        self.num_scenarios = num_scenarios or len(get_scenarios())

    def update(self, seed_indices, errors):
        """Update scores with exponential moving average."""
        dirty = False
        for idx, error in zip(seed_indices, errors):
            idx = int(idx)
            if idx >= self.num_scenarios:
                continue
            current = self.scores.get(idx, 1.0)
            self.scores[idx] = 0.9 * current + 0.1 * float(error)
            dirty = True
        if dirty:
            self._save()

    def _save(self):
        """Save scores to file for cross-process sharing."""
        try:
            temp = PLR_FILE + ".tmp"
            with open(temp, 'w') as f:
                json.dump(self.scores, f)
            os.replace(temp, PLR_FILE)
        except Exception:
            pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_yaw_facing_position(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """Calculate yaw angle for a car at from_pos to face to_pos."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return np.arctan2(dy, dx)


def mirror_position(pos: np.ndarray) -> np.ndarray:
    """Mirror position across field center for orange team: (x, y, z) -> (-x, -y, z)"""
    return np.array([-pos[0], -pos[1], pos[2]], dtype=np.float32)


def mirror_velocity(vel: np.ndarray) -> np.ndarray:
    """Mirror velocity for orange team: (vx, vy, vz) -> (-vx, -vy, vz)"""
    return np.array([-vel[0], -vel[1], vel[2]], dtype=np.float32)


# =============================================================================
# PLR MUTATOR - Selects and applies scenarios
# =============================================================================
class PLRMutator(StateMutator):
    """
    State mutator that selects scenarios using Prioritized Level Replay.

    Supports both 1v0 and 1v1 modes:
      - 1v0: orange_size=0, single agent training
      - 1v1: orange_size=1, self-play (orange mirrored)
    """

    def __init__(self, replay_prob=0.6, blue_size=1, orange_size=0):
        self.replay_prob = replay_prob
        self.scenarios = get_scenarios()
        self.scores = {i: 1.0 for i in range(len(self.scenarios))}
        self.current_idx = 0
        self.last_sync = 0
        self.kickoff_mutator = KickoffMutator()
        self.forced_idx = None
        self.blue_size = blue_size
        self.orange_size = orange_size
        self.spawn_mutator = FixedTeamSizeMutator(blue_size=blue_size, orange_size=orange_size)

    def _sync(self):
        """Sync scores from file every 5 seconds."""
        if time.time() - self.last_sync > 5 and os.path.exists(PLR_FILE):
            try:
                with open(PLR_FILE, 'r') as f:
                    data = json.load(f)
                    num_scenarios = len(self.scenarios)
                    self.scores = {int(k): v for k, v in data.items() if int(k) < num_scenarios}
                self.last_sync = time.time()
            except Exception:
                pass

    def _select_scenario_idx(self) -> int:
        """
        Select scenario index using PLR logic:
          - 20% forced aerials 
          - 60% PLR sampling
          - 20% random
        """
        aerial_indices = [i for i, s in enumerate(self.scenarios)
                         if s.type in ("aerial_hover", "high_pop")]

        if aerial_indices and np.random.random() < 0.20:
            return np.random.choice(aerial_indices)
        elif np.random.random() < self.replay_prob:
            total = sum(self.scores.get(i, 1.0) for i in range(len(self.scenarios)))
            probs = [self.scores.get(i, 1.0) / total for i in range(len(self.scenarios))]
            return np.random.choice(len(self.scenarios), p=probs)
        else:
            return np.random.randint(0, len(self.scenarios))

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        """Apply selected scenario to game state."""
        self.spawn_mutator.apply(state, shared_info)
        self._sync()

        if self.forced_idx is not None:
            self.current_idx = self.forced_idx
        else:
            self.current_idx = self._select_scenario_idx()

        scenario = self.scenarios[self.current_idx]
        shared_info['scenario_idx'] = float(self.current_idx)

        # Kickoff uses built-in mutator
        if scenario.type == "kickoff":
            self.kickoff_mutator.apply(state, {})
            return

        # Generate ball and car states
        ball_state, car_state = self._generate_scenario_states(scenario)

        # Apply ball state
        state.ball.position = ball_state['position'].copy()
        state.ball.linear_velocity = ball_state['linear_velocity'].copy()
        state.ball.angular_velocity = ball_state['angular_velocity'].copy()

        # Apply car states (orange is mirrored in 1v1)
        for car in state.cars.values():
            if car.team_num == 0:  # Blue team
                car.physics.position = car_state['position'].copy()
                car.physics.linear_velocity = car_state['linear_velocity'].copy()
                car.physics.angular_velocity = car_state['angular_velocity'].copy()
                # Use scenario-specific euler_angles if provided, otherwise face ball
                if 'euler_angles' in car_state:
                    car.physics.euler_angles = car_state['euler_angles'].copy()
                else:
                    yaw = get_yaw_facing_position(car.physics.position, state.ball.position)
                    car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)
            else:  # Orange team (mirrored)
                car.physics.position = mirror_position(car_state['position'])
                car.physics.linear_velocity = mirror_velocity(car_state['linear_velocity'])
                car.physics.angular_velocity = mirror_velocity(car_state['angular_velocity'])
                # Orange always faces ball (for 1v1 fairness)
                yaw = get_yaw_facing_position(car.physics.position, state.ball.position)
                car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)

            car.boost_amount = car_state['boost']

    def _generate_scenario_states(self, scenario: Scenario) -> Tuple[Dict, Dict]:
        """
        Generate ball and car states for a scenario type.
        All scenarios organized by training phase.
        """
        rng = np.random

        # Default states
        ball_state = {
            'position': np.array([0, 0, BALL_RADIUS], dtype=np.float32),
            'linear_velocity': np.zeros(3, dtype=np.float32),
            'angular_velocity': np.zeros(3, dtype=np.float32),
        }
        car_state = {
            'position': np.array([0, -2000, 17], dtype=np.float32),
            'linear_velocity': np.zeros(3, dtype=np.float32),
            'angular_velocity': np.zeros(3, dtype=np.float32),
            'boost': 100.0,
        }

        # =====================================================================
        # PHASE 1: THE TOUCH AGENT SCENARIOS
        # =====================================================================

        # TOUCH_DISTANCE - Ball at various distances from car
        if scenario.type == "touch_distance":
            min_d = scenario.config.get('min_dist', 300)
            max_d = scenario.config.get('max_dist', 800)
            distance = rng.uniform(min_d, max_d)
            offset_x = rng.uniform(-50, 50)

            ball_state['position'] = np.array([0, 0, BALL_RADIUS], dtype=np.float32)
            car_state['position'] = np.array([offset_x, -distance, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 100.0

        # TOUCH_STEERING - Ball offset from car's forward direction
        elif scenario.type == "touch_steering":
            distance = rng.uniform(1000, 3500)
            min_off = scenario.config.get('min_off', 100)
            max_off = scenario.config.get('max_off', 500)
            side = rng.choice([-1, 1])
            offset_x = rng.uniform(min_off, max_off) * side

            ball_state['position'] = np.array([0, 0, BALL_RADIUS], dtype=np.float32)
            car_state['position'] = np.array([offset_x, -distance, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 100.0

        # =====================================================================
        # PHASE 2: THE GOAL SCORER SCENARIOS
        # =====================================================================

        # TAP_IN - Ball very close to goal, easy tap
        elif scenario.type == "tap_in":
            ball_state['position'] = np.array([
                rng.uniform(-500, 500),
                rng.uniform(3500, 4500),  # Very close to orange goal
                BALL_RADIUS
            ], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([
                rng.uniform(-50, 50),
                rng.uniform(50, 150),  # Slow roll toward goal
                0
            ], dtype=np.float32)

            car_state['position'] = np.array([
                ball_state['position'][0] + rng.uniform(-200, 200),
                ball_state['position'][1] - rng.uniform(300, 600),
                17
            ], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 100.0

        # OPEN_NET - Clear path to goal
        elif scenario.type == "open_net":
            ball_state['position'] = np.array([
                rng.uniform(-800, 800),
                rng.uniform(1000, 2000),
                BALL_RADIUS
            ], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([
                rng.uniform(-100, 100),
                rng.uniform(0, 200),
                0
            ], dtype=np.float32)

            car_state['position'] = np.array([0, -1000, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 100.0

        # CENTER_SHOT - Ball at midfield, rolling toward goal
        elif scenario.type == "center_shot":
            ball_state['position'] = np.array([
                rng.uniform(-1000, 1000),
                rng.uniform(-500, 500),
                BALL_RADIUS
            ], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([
                rng.uniform(-300, 300),
                rng.uniform(200, 600),
                0
            ], dtype=np.float32)

            car_state['position'] = np.array([rng.uniform(-500, 500), -2500, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)  # Face +Y (goal)
            car_state['boost'] = 50.0

        # =====================================================================
        # PHASE 2.5: THE GOAL SCORER 2.0 SCENARIOS
        # =====================================================================

        # ANGLED_SHOT - Off-center shots requiring aim
        elif scenario.type == "angled_shot":
            side = -1 if scenario.config.get('side', 'left') == 'left' else 1

            ball_state['position'] = np.array([
                rng.uniform(500, 1500) * side,
                rng.uniform(1500, 2500),
                BALL_RADIUS
            ], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([
                rng.uniform(-200, 200),
                rng.uniform(100, 400),
                0
            ], dtype=np.float32)

            car_state['position'] = np.array([rng.uniform(500, 1500) * side, -1500, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 70.0

        # FAST_ROLLING - Ball moving fast toward goal
        elif scenario.type == "fast_rolling":
            ball_x = rng.uniform(-800, 800)
            ball_y = rng.uniform(-500, 1000)

            ball_state['position'] = np.array([ball_x, ball_y, BALL_RADIUS], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([
                rng.uniform(-300, 300),
                rng.uniform(800, 1500),
                0
            ], dtype=np.float32)

            car_state['position'] = np.array([
                rng.uniform(-500, 500),
                ball_y - rng.uniform(800, 1500),
                17
            ], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 100.0

        # REBOUND_CHASE - Ball bouncing off wall
        elif scenario.type == "rebound_chase":
            side = rng.choice([-1, 1])
            safe_max_x = SIDE_WALL_X - BALL_RADIUS - 200
            wall_x = rng.uniform(2000, min(3000, safe_max_x))

            ball_state['position'] = np.array([
                wall_x * side,
                rng.uniform(500, 2000),
                BALL_RADIUS
            ], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([
                rng.uniform(300, 800) * -side,
                rng.uniform(400, 900),
                0
            ], dtype=np.float32)

            car_state['position'] = np.array([
                rng.uniform(-500, 500),
                rng.uniform(-1000, 0),
                17
            ], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = rng.uniform(50, 100)

        # RECOVERY_DRILL - Ball behind car (stationary)
        elif scenario.type == "recovery_drill":
            mode = scenario.config.get("mode", "offense")
            ball_x = rng.uniform(-1500, 1500)

            if mode == "defense":
                ball_y = rng.uniform(-3500, -1000)
            else:
                ball_y = rng.uniform(-1000, 2000)

            ball_state['position'] = np.array([ball_x, ball_y, BALL_RADIUS], dtype=np.float32)

            offset_y = rng.uniform(800, 2000)
            car_state['position'] = np.array([ball_x, ball_y + offset_y, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 80.0
        
        # STATIONARY_TURN - Ball behind/to side of car
        elif scenario.type == "stationary_turn":
            min_angle_deg = scenario.config.get("min_angle", 90)
            max_angle_deg = scenario.config.get("max_angle", 180)

            side = rng.choice([-1, 1])
            angle_rad = np.deg2rad(rng.uniform(min_angle_deg, max_angle_deg)) * side
            dist = rng.uniform(800, 1500)

            ball_x = dist * np.sin(angle_rad)
            ball_y = dist * np.cos(angle_rad)

            ball_state['position'] = np.array([ball_x, ball_y, BALL_RADIUS], dtype=np.float32)
            car_state['position'] = np.array([0, 0, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 33.0

        # =====================================================================
        # PHASE 3: THE AERIAL AGENT SCENARIOS
        # =====================================================================

        # RECOVERY_OFFENSE - Ball behind car and MOVING
        elif scenario.type == "recovery_offense":
            car_x = rng.uniform(-1000, 1000)
            car_y = rng.uniform(500, 2000)

            ball_x = car_x + rng.uniform(-800, 800)
            ball_y = car_y - rng.uniform(1200, 2500)

            ball_vel_x = rng.uniform(-500, 500)
            ball_vel_y = rng.uniform(-500, 0)

            ball_state['position'] = np.array([ball_x, ball_y, BALL_RADIUS], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([ball_vel_x, ball_vel_y, 0], dtype=np.float32)

            car_state['position'] = np.array([car_x, car_y, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 80.0

        # WALL_DRILL - Ball rolling along wall
        elif scenario.type == "wall_drill":
            side = rng.choice([-1, 1])
            x_pos = (SIDE_WALL_X - 200) * side

            car_y = rng.uniform(-2000, 0)
            ball_y = car_y + rng.uniform(400, 700)

            ball_state['position'] = np.array([x_pos, ball_y, BALL_RADIUS], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([0, rng.uniform(800, 1200), 0], dtype=np.float32)

            car_state['position'] = np.array([x_pos, car_y, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 50.0

        # AERIAL_HOVER - Ball at jump height
        elif scenario.type == "aerial_hover":
            ball_z = rng.uniform(250, 450)
            ball_x = rng.uniform(-2000, 2000)
            ball_y = rng.uniform(-2000, 2000)

            ball_state['position'] = np.array([ball_x, ball_y, ball_z], dtype=np.float32)

            offset_dist = rng.uniform(400, 600)
            car_state['position'] = np.array([ball_x, ball_y - offset_dist, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 100.0

        # HIGH_POP - Ball popped up with vertical velocity
        elif scenario.type == "high_pop":
            ball_x = rng.uniform(-2000, 2000)
            ball_y = rng.uniform(-1500, 1500)
            ball_vel_z = rng.uniform(400, 700)

            ball_state['position'] = np.array([ball_x, ball_y, 100], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([
                rng.uniform(-100, 100),
                rng.uniform(-100, 100),
                ball_vel_z
            ], dtype=np.float32)

            car_state['position'] = np.array([
                ball_x + rng.uniform(-200, 200),
                ball_y - rng.uniform(300, 600),
                17
            ], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 100.0

        # =====================================================================
        # PHASE 4: THE GENERALIST AGENT SCENARIOS
        # =====================================================================

        # RECOVERY_DEFENSE - Ball shooting toward own goal
        elif scenario.type == "recovery_defense":
            ball_x = rng.uniform(-1500, 1500)
            ball_y = rng.uniform(-1000, 1500)
            ball_vel_y = rng.uniform(-1500, -800)
            ball_vel_x = rng.uniform(-300, 300)

            ball_state['position'] = np.array([ball_x, ball_y, BALL_RADIUS], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([ball_vel_x, ball_vel_y, 0], dtype=np.float32)

            car_state['position'] = np.array([
                rng.uniform(-1000, 1000),
                rng.uniform(-4000, -2500),
                17
            ], dtype=np.float32)
            # No euler_angles - let it face ball (defensive awareness)
            car_state['boost'] = 100.0

        # WALL_SHOT - Ball near wall at moderate height
        elif scenario.type == "wall_shot":
            side = -1 if scenario.config.get('side', 'left') == 'left' else 1
            wall_x = (SIDE_WALL_X - 300) * side
            ball_y = rng.uniform(1000, 3000)
            ball_z = rng.uniform(200, 500)

            ball_state['position'] = np.array([wall_x, ball_y, ball_z], dtype=np.float32)
            ball_state['linear_velocity'] = np.array([
                rng.uniform(-200, 200) * side,
                rng.uniform(200, 600),
                rng.uniform(-100, 100)
            ], dtype=np.float32)

            car_state['position'] = np.array([
                wall_x * 0.3,
                ball_y - rng.uniform(800, 1500),
                17
            ], dtype=np.float32)
            # No euler_angles - let it face ball
            car_state['boost'] = 80.0

        # TURN_BACK_EASY - Ball stationary behind car, close
        elif scenario.type == "turn_back_easy":
            car_x = rng.uniform(-1500, 1500)
            car_y = rng.uniform(500, 2500)

            ball_distance = rng.uniform(800, 1500)
            ball_x = car_x + rng.uniform(-300, 300)
            ball_y = car_y - ball_distance

            ball_state['position'] = np.array([ball_x, ball_y, BALL_RADIUS], dtype=np.float32)
            car_state['position'] = np.array([car_x, car_y, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 50.0

        # TURN_BACK_HARD - Ball moving OR far behind car
        elif scenario.type == "turn_back_hard":
            car_x = rng.uniform(-1500, 1500)
            car_y = rng.uniform(1000, 3000)

            ball_distance = rng.uniform(1500, 3000)
            ball_x = car_x + rng.uniform(-800, 800)
            ball_y = car_y - ball_distance

            ball_state['position'] = np.array([ball_x, ball_y, BALL_RADIUS], dtype=np.float32)

            # 50% chance ball is moving
            if rng.random() < 0.5:
                ball_state['linear_velocity'] = np.array([
                    rng.uniform(-400, 400),
                    rng.uniform(-600, -100),
                    0
                ], dtype=np.float32)

            car_state['position'] = np.array([car_x, car_y, 17], dtype=np.float32)
            car_state['euler_angles'] = np.array([0, np.pi/2, 0], dtype=np.float32)
            car_state['boost'] = 80.0

        return ball_state, car_state

# =============================================================================
# PLR OBS BUILDER - Appends scenario ID to observations
# =============================================================================
class PLRObsBuilder(ObsBuilder):
    """
    Observation builder that appends scenario ID to base observations.

    Features:
      - task_dropout: Probability of replacing ID with 0.0 (for generalization)
      - force_blind: Always use ID=0.0 (for testing)
      - obs_size: Fixed observation size (73 = 72 base + 1 scenario ID)
    """

    def __init__(self, base_obs, task_dropout=0.0, force_blind=False, obs_size=72):
        self.base_obs = base_obs
        self.current_idx = 0.0
        self.task_dropout = task_dropout
        self.force_blind = force_blind
        self.obs_size = obs_size

    def reset(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                self.current_idx = arg.get('scenario_idx', 0.0)
                break
        try:
            self.base_obs.reset(*args, **kwargs)
        except TypeError:
            pass

    def get_obs_space(self, agent):
        base_space = self.base_obs.get_obs_space(agent)
        fixed_size = self.obs_size + 1

        if isinstance(base_space, tuple):
            return (base_space[0], fixed_size)
        return fixed_size

    def build_obs(self, agents, state: GameState, shared_info: dict):
        base_output = self.base_obs.build_obs(agents, state, shared_info)

        real_idx = float(self.current_idx) if not isinstance(self.current_idx, dict) else 0.0

        # Apply task dropout or force_blind
        if self.force_blind:
            seed = 0.0
        elif self.task_dropout > 0 and np.random.random() < self.task_dropout:
            seed = 0.0
        else:
            seed = real_idx

        def process_obs(obs):
            sliced = obs[:self.obs_size]
            return np.append(sliced, [seed]).astype(np.float32)

        if isinstance(base_output, dict):
            return {k: process_obs(v) for k, v in base_output.items()}

        if isinstance(base_output, list):
            return [process_obs(x) for x in base_output]

        return process_obs(base_output)
