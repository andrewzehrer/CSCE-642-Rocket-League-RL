"""
CSCE 642 Rocket League RL Bot
Uses trained PPO policy with RLBot v4 API
"""

import math
import numpy as np
import torch
from pathlib import Path

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from discrete import DiscreteFF
from act import LookupTableAction


# Constants for observation normalization (must match training)
SIDE_WALL_X = 4096
BACK_NET_Y = 5120
CEILING_Z = 2044
CAR_MAX_SPEED = 2300
CAR_MAX_ANG_VEL = 5.5


class CSCE642Bot(BaseAgent):
    """
    RL Bot trained with PPO and PLR curriculum.
    """

    def initialize_agent(self):
        """Called once when the bot is loaded."""
        self.controller = SimpleControllerState()
        self.tick_counter = 0
        self.tick_skip = 8
        self.last_action = np.zeros(8)
        self.initialized = False

        # State tracking for partial observables (RLBot doesn't expose these directly)
        self.is_holding_jump = False
        self.jump_start_tick = 0
        self.is_jumping = False
        self.has_flipped = False
        self.is_flipping = False
        self.flip_start_tick = 0
        self.air_time_since_jump = 0.0
        self.last_on_ground = True
        self.left_ground_tick = 0
        self.last_jumped = False
        self.last_double_jumped = False
        self.last_handbrake = False
        self.last_car_pos = (0, 0, 0)
        self._was_jumping = False

        # Load the policy
        self.device = torch.device("cpu")
        model_path = Path(__file__).parent / "PPO_POLICY.pt"

        if not model_path.exists():
            self.logger.error(f"Policy file not found at {model_path}")
            return

        try:
            # Load model and get architecture from weights
            try:
                model_file = torch.load(str(model_path), map_location=self.device, weights_only=True)
            except TypeError:
                # Older PyTorch versions don't support weights_only
                model_file = torch.load(str(model_path), map_location=self.device)

            # Extract architecture from weights
            from collections import OrderedDict
            state_dict = OrderedDict(model_file)
            bias_counts = []
            weight_counts = []
            for key, value in state_dict.items():
                if ".weight" in key:
                    weight_counts.append(value.numel())
                if ".bias" in key:
                    bias_counts.append(value.size(0))

            input_amount = int(weight_counts[0] / bias_counts[0])
            action_amount = bias_counts[-1]
            layer_sizes = bias_counts[:-1]

            self.logger.info(f"Model: inputs={input_amount}, actions={action_amount}, layers={layer_sizes}")

            # Create policy
            self.policy = DiscreteFF(input_amount, action_amount, layer_sizes, self.device)
            self.policy.load_state_dict(model_file)
            self.policy.eval()

            # Action parser
            self.action_parser = LookupTableAction()

            self.initialized = True
            self.logger.info("CSCE642Bot initialized successfully!")

        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """Called every tick to get the bot's output."""

        # If not initialized, do nothing
        if not self.initialized:
            return self.controller

        self.tick_counter += 1

        # Update state tracking EVERY tick (not just action ticks)
        try:
            car = packet.game_cars[self.index]
            self.update_state_tracking(car)
        except:
            pass

        # Only compute new action every tick_skip frames
        if self.tick_counter % self.tick_skip != 0:
            return self.controller

        try:
            # Build observation
            obs = self.build_obs(packet)
            if obs is None:
                self.logger.warning("build_obs returned None")
                return self.controller

            # Debug: log actual game values vs observation values
            if self.tick_counter % 120 == 0:  # Log every ~1 second
                # Raw game values
                ball = packet.game_ball
                raw_ball = (ball.physics.location.x, ball.physics.location.y, ball.physics.location.z)
                raw_car = (car.physics.location.x, car.physics.location.y, car.physics.location.z)
                # What we're sending to the network
                obs_ball = obs[0:3]  # Normalized ball pos
                obs_car = obs[52:55]  # Normalized car pos
                # Distance to ball
                dist = ((raw_ball[0]-raw_car[0])**2 + (raw_ball[1]-raw_car[1])**2)**0.5
                self.logger.info(f"RAW ball={raw_ball}, car={raw_car}, dist={dist:.0f}")
                self.logger.info(f"OBS ball={obs_ball}, car={obs_car}")

            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action_idx, _ = self.policy.get_action(obs_tensor, deterministic=True)
                if hasattr(action_idx, 'item'):
                    action_idx = action_idx.item()
                elif hasattr(action_idx, '__len__'):
                    action_idx = int(action_idx[0]) if len(action_idx) > 0 else 0
                else:
                    action_idx = int(action_idx)

            # Debug: log action
            if self.tick_counter <= 16:
                self.logger.info(f"Action idx: {action_idx}")

            # Parse action
            action = self.action_parser._lookup_table[action_idx]
            self.last_action = action

            # Update controller
            self.controller.throttle = float(action[0])
            self.controller.steer = float(action[1])
            self.controller.pitch = float(action[2])
            self.controller.yaw = float(action[3])
            self.controller.roll = float(action[4])
            self.controller.jump = bool(action[5] > 0.5)
            self.controller.boost = bool(action[6] > 0.5)
            self.controller.handbrake = bool(action[7] > 0.5)

            # Debug: log controller
            if self.tick_counter <= 16:
                self.logger.info(f"Controller: throttle={self.controller.throttle}, steer={self.controller.steer}")

        except Exception as e:
            self.logger.error(f"Error in get_output: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return self.controller

    def reset_state_tracking(self):
        """Reset all state tracking variables to defaults."""
        self.is_holding_jump = False
        self.jump_start_tick = self.tick_counter
        self.is_jumping = False
        self.has_flipped = False
        self.is_flipping = False
        self.flip_start_tick = self.tick_counter
        self.air_time_since_jump = 0.0
        self.last_on_ground = True
        self.left_ground_tick = self.tick_counter
        self.last_jumped = False
        self.last_double_jumped = False
        self.last_handbrake = False
        self._was_jumping = False

    def update_state_tracking(self, car):
        """
        Update state tracking variables to approximate partial observables.
        """
        on_ground = car.has_wheel_contact
        jumped = car.jumped
        double_jumped = car.double_jumped

        # Detect kickoff/respawn: car on ground + jumped/double_jumped reset to False
        # This happens when game resets after goal
        if on_ground and not jumped and not double_jumped:
            if self.last_jumped or self.last_double_jumped or not self.last_on_ground:
                # State just reset (goal scored, respawn, etc.)
                self.reset_state_tracking()
                self.last_on_ground = on_ground
                self.last_jumped = jumped
                self.last_double_jumped = double_jumped
                return

        # Track when we leave the ground
        if self.last_on_ground and not on_ground:
            self.left_ground_tick = self.tick_counter

        # is_holding_jump: We're outputting jump
        self.is_holding_jump = self.controller.jump
        if self.controller.jump and not getattr(self, '_was_jumping', False):
            self.jump_start_tick = self.tick_counter
        self._was_jumping = self.controller.jump

        # is_jumping: First jump active (in air, jumped, not double jumped)
        if jumped and not double_jumped and not on_ground:
            ticks_since_jump = self.tick_counter - self.jump_start_tick
            self.is_jumping = ticks_since_jump < 24
        else:
            self.is_jumping = False

        # has_flipped & is_flipping
        if double_jumped and not self.last_double_jumped:
            self.has_flipped = True
            self.is_flipping = True
            self.flip_start_tick = self.tick_counter
        elif self.has_flipped:
            ticks_since_flip = self.tick_counter - self.flip_start_tick
            self.is_flipping = ticks_since_flip < 78

        # Reset flip state when on ground
        if on_ground:
            self.has_flipped = False
            self.is_flipping = False

        # air_time_since_jump
        if not on_ground:
            ticks_in_air = self.tick_counter - self.left_ground_tick
            self.air_time_since_jump = ticks_in_air / 120.0  # Convert to seconds
        else:
            self.air_time_since_jump = 0.0

        # Update last states
        self.last_on_ground = on_ground
        self.last_jumped = jumped
        self.last_double_jumped = double_jumped
        self.last_handbrake = self.controller.handbrake

    def build_obs(self, packet: GameTickPacket) -> np.ndarray:
        """
        Build observation matching training format.
        Returns 73 dims: 72 (ball + pads + partial + car) + 1 (scenario_idx=0.0)
        """
        try:
            car = packet.game_cars[self.index]
            ball = packet.game_ball

            # Team inversion for orange (invert X and Y, keep Z)
            inv = -1 if self.team == 1 else 1

            obs = []

            # === BALL (9 dims) ===
            pos_coef = np.array([1/SIDE_WALL_X, 1/BACK_NET_Y, 1/CEILING_Z])

            # For orange team: invert X and Y positions/velocities
            ball_pos = np.array([
                ball.physics.location.x * inv,
                ball.physics.location.y * inv,
                ball.physics.location.z
            ])
            ball_vel = np.array([
                ball.physics.velocity.x * inv,
                ball.physics.velocity.y * inv,
                ball.physics.velocity.z
            ])
            ball_ang = np.array([
                ball.physics.angular_velocity.x * inv,
                ball.physics.angular_velocity.y * inv,
                ball.physics.angular_velocity.z
            ])

            obs.extend(ball_pos * pos_coef)
            obs.extend(ball_vel / CAR_MAX_SPEED)
            obs.extend(ball_ang / CAR_MAX_ANG_VEL)

            # === BOOST PADS (34 dims) ===
            # Note: RLBot boost pad order may differ from RocketSim
            # Using zeros for now - the agent should work without perfect pad info
            boost_timers = np.zeros(34)
            obs.extend(boost_timers)

            # === PARTIAL OBSERVABLES (9 dims) ===
            # Use tracked state variables for accurate body awareness
            can_flip = (not car.double_jumped and
                       not car.has_wheel_contact and
                       self.air_time_since_jump < 1.5)

            obs.extend([
                float(self.is_holding_jump),
                float(self.last_handbrake),
                float(car.jumped),
                float(self.is_jumping),
                float(self.has_flipped),
                float(self.is_flipping),
                float(car.double_jumped),
                float(can_flip),
                self.air_time_since_jump
            ])

            # === OWN CAR (20 dims) ===
            car_pos = np.array([
                car.physics.location.x * inv,
                car.physics.location.y * inv,
                car.physics.location.z
            ])
            car_vel = np.array([
                car.physics.velocity.x * inv,
                car.physics.velocity.y * inv,
                car.physics.velocity.z
            ])
            car_ang_vel = np.array([
                car.physics.angular_velocity.x * inv,
                car.physics.angular_velocity.y * inv,
                car.physics.angular_velocity.z
            ])

            # Calculate forward and up vectors from rotation
            pitch = car.physics.rotation.pitch
            yaw = car.physics.rotation.yaw
            roll = car.physics.rotation.roll

            # For orange team, add 180 degrees to yaw (rotate view)
            if inv == -1:
                yaw = yaw + math.pi

            cp = math.cos(pitch)
            sp = math.sin(pitch)
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            cr = math.cos(roll)
            sr = math.sin(roll)

            # Forward vector (direction car is facing)
            # Matches rlgym: rotation_mtx[:, 0]
            forward = np.array([cp*cy, cp*sy, sp])

            # Up vector
            # Matches rlgym: rotation_mtx[:, 2] from euler_to_rotation()
            up = np.array([
                -cr*cy*sp - sr*sy,
                -cr*sy*sp + sr*cy,
                cp*cr
            ])

            obs.extend(car_pos * pos_coef)
            obs.extend(forward)
            obs.extend(up)
            obs.extend(car_vel / CAR_MAX_SPEED)
            obs.extend(car_ang_vel / CAR_MAX_ANG_VEL)

            # is_boosting: We're outputting boost AND have boost available
            is_boosting = self.controller.boost and car.boost > 0

            obs.extend([
                car.boost / 100.0,
                0.0,  # demo_respawn_timer (not available in RLBot)
                float(car.has_wheel_contact),
                float(is_boosting),
                float(car.is_super_sonic)
            ])

            # === SCENARIO INDEX (1 dim) - ALWAYS 0.0 ===
            obs.append(0.0)

            result = np.array(obs, dtype=np.float32)

            # Log the actual shape for debugging
            if self.tick_counter <= 16:
                self.logger.info(f"Built obs with {len(result)} dims")

            return result

        except Exception as e:
            self.logger.error(f"Error building obs: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
