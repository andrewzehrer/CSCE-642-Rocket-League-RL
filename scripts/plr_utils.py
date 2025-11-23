import numpy as np
import os
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List

from rocket_league_rl.rlgym.api import StateMutator, ObsBuilder
from rocket_league_rl.rlgym.rocket_league.api import GameState
from rocket_league_rl.rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_NET_Y, CEILING_Z, BALL_RADIUS
from rocket_league_rl.rlgym.rocket_league.state_mutators import KickoffMutator

# To keep track of the scores
PLR_FILE = "plr_state.json"

# Define the Scenarios
@dataclass
class Scenario:
    name: str
    type: str
    config: Dict[str, Any]

# Create a fixed list of scenarios to train
def get_scenarios() -> List[Scenario]:
    scenarios = []
    scenarios.append(Scenario("Kickoff_Center", "kickoff", {"var": 0}))
    scenarios.append(Scenario("Rolling_Shot_Mid", "rolling", {"x_range": 1000, "y_dist": 2000}))
    scenarios.append(Scenario("Rolling_Shot_Far", "rolling", {"x_range": 2000, "y_dist": 3500}))
    scenarios.append(Scenario("Wall_Left", "wall", {"side": "left"}))
    scenarios.append(Scenario("Wall_Right", "wall", {"side": "right"}))
    scenarios.append(Scenario("Defense_Save", "defense", {}))
    return scenarios

# This Class tracks and updates the scores for each scenarios based on the acquired results
class PLRBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.scores = {} 
        
    def update(self, seed_indices, errors):
        dirty = False
        for idx, error in zip(seed_indices, errors):
            idx = int(idx)
            current = self.scores.get(idx, 1.0)
            self.scores[idx] = 0.9 * current + 0.1 * float(error)
            dirty = True
        if dirty: self._save()

    def _save(self):
        try:
            temp = PLR_FILE + ".tmp"
            with open(temp, 'w') as f: json.dump(self.scores, f)
            os.replace(temp, PLR_FILE)
        except: pass

# This Class decides the scenario to train the model based on the scores.
class PLRMutator(StateMutator):
    def __init__(self, replay_prob=0.6):
        self.replay_prob = replay_prob
        self.scenarios = get_scenarios()
        self.scores = {i: 1.0 for i in range(len(self.scenarios))}
        self.current_idx = 0
        self.last_sync = 0
        self.kickoff_mutator = KickoffMutator()

    def _sync(self):
        if time.time() - self.last_sync > 5 and os.path.exists(PLR_FILE):
            try:
                with open(PLR_FILE, 'r') as f:
                    data = json.load(f)
                    self.scores = {int(k): v for k,v in data.items()}
                self.last_sync = time.time()
            except: pass

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        self._sync()
        if np.random.random() < self.replay_prob:
            total_score = sum(self.scores.get(i, 1.0) for i in range(len(self.scenarios)))
            probs = [self.scores.get(i, 1.0) / total_score for i in range(len(self.scenarios))]
            self.current_idx = np.random.choice(len(self.scenarios), p=probs)
        else:
            self.current_idx = np.random.randint(0, len(self.scenarios))

        scenario = self.scenarios[self.current_idx]
        self._apply_scenario_physics(state, scenario)
        shared_info['scenario_idx'] = float(self.current_idx)

    def _apply_scenario_physics(self, state: GameState, scenario: Scenario):
        rng = np.random
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)

        for car in state.cars.values():
            car.physics.position = np.array([0, -3000, 17], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.zeros(3, dtype=np.float32)
            car.boost_amount = 33.0

        if scenario.type == "kickoff":
            self.kickoff_mutator.apply(state, {})
        elif scenario.type == "rolling":
            cfg = scenario.config
            state.ball.position = np.array([rng.uniform(-cfg['x_range'], cfg['x_range']), rng.uniform(-cfg['y_dist'], cfg['y_dist']), BALL_RADIUS], dtype=np.float32)
            state.ball.linear_velocity = np.array([rng.uniform(-500, 500), rng.uniform(-500, 500), 0], dtype=np.float32)
        elif scenario.type == "wall":
            side = -1 if scenario.config['side'] == 'left' else 1
            wall_x = (SIDE_WALL_X - 50) * side
            state.ball.position = np.array([wall_x, 0, 500], dtype=np.float32)
            state.ball.linear_velocity = np.array([0, 1000, 500], dtype=np.float32)
            for car in state.cars.values():
                car.physics.position = np.array([wall_x * 0.5, -1000, 17], dtype=np.float32)
                car.physics.euler_angles[1] = 0.5 * np.pi * side
        elif scenario.type == "defense":
            state.ball.position = np.array([rng.uniform(-1000, 1000), -1000, BALL_RADIUS], dtype=np.float32)
            state.ball.linear_velocity = np.array([rng.uniform(-500, 500), -1500, 0], dtype=np.float32)

# OBS Builder that appends the scenario id to the observation
class PLRObsBuilder(ObsBuilder):
    def __init__(self, base_obs):
        self.base_obs = base_obs
        self.current_idx = 0.0

    def reset(self, *args, **kwargs):
        # Find shared_info to get the seed
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
        if isinstance(base_space, tuple):
            return (base_space[0], base_space[1] + 1)
        return base_space + 1

    def build_obs(self, agents, state: GameState, shared_info: dict):

        base_output = self.base_obs.build_obs(agents, state, shared_info)
        
        seed = float(self.current_idx) if not isinstance(self.current_idx, dict) else 0.0

        if isinstance(base_output, dict):
            for k, v in base_output.items():
                base_output[k] = np.append(v, [seed]).astype(np.float32)
            return base_output
            
        if isinstance(base_output, list): 
            return [np.append(x, [seed]).astype(np.float32) for x in base_output]
            
        return np.append(base_output, [seed]).astype(np.float32)