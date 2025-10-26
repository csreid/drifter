import numpy as np
from enum import Enum


class ManeuverType(Enum):
	"""Types of exploration maneuvers"""

	RANDOM = "random"
	HIGH_SPEED_TURN = "high_speed_turn"
	ZIGZAG = "zigzag"
	EMERGENCY_CORRECTION = "emergency_correction"
	ACCELERATION_BURST = "acceleration_burst"
	DRIFT_ATTEMPT = "drift_attempt"


class ExplorationPolicy:
	"""
	Stateful exploration policy for discovering dangerous driving behaviors.

	Manages multi-step maneuvers that explore high-risk state-action combinations
	like high-speed turns that can cause flipping.
	"""

	def __init__(
		self,
		action_space,
		maneuver_duration=10,
		aggressive_prob=0.35,
		velocity_threshold=0.3,
		random_seed=None,
	):
		"""
		Args:
		    action_space: Gym action space (assumes [steering, throttle])
		    maneuver_duration: Number of steps each dangerous maneuver lasts
		    aggressive_prob: Probability of starting aggressive maneuver
		    velocity_threshold: Speed threshold for velocity-aware behaviors
		    random_seed: Random seed for reproducibility
		"""
		self._action_space = action_space
		self._maneuver_duration = maneuver_duration
		self._aggressive_prob = aggressive_prob
		self._velocity_threshold = velocity_threshold

		if random_seed is not None:
			np.random.seed(random_seed)

		# State tracking
		self._current_maneuver = ManeuverType.RANDOM
		self._maneuver_steps_remaining = 0
		self._maneuver_params = {}
		self._total_steps = 0

		# Statistics
		self.stats = {
			"total_actions": 0,
			"maneuvers_started": {m: 0 for m in ManeuverType},
			"maneuvers_completed": {m: 0 for m in ManeuverType},
		}

	def get_action(self, observation):
		"""
		Get next action based on current exploration policy state.

		Args:
		    observation: Current state observation from environment

		Returns:
		    action: numpy array of actions to take
		"""
		self._total_steps += 1
		self.stats["total_actions"] += 1

		# Check if we need to start a new maneuver
		if self._maneuver_steps_remaining <= 0:
			self._select_new_maneuver(observation)

		# Execute current maneuver
		action = self._execute_maneuver(observation)
		self._maneuver_steps_remaining -= 1

		# Track completed maneuvers
		if self._maneuver_steps_remaining == 0:
			self.stats["maneuvers_completed"][self._current_maneuver] += 1

		return action

	def _select_new_maneuver(self, observation):
		"""Select and initialize a new maneuver"""
		if np.random.rand() < self._aggressive_prob:
			# Choose a dangerous maneuver
			maneuver = np.random.choice(
				[
					ManeuverType.HIGH_SPEED_TURN,
					ManeuverType.ZIGZAG,
					ManeuverType.EMERGENCY_CORRECTION,
					ManeuverType.ACCELERATION_BURST,
					ManeuverType.DRIFT_ATTEMPT,
				]
			)
		else:
			maneuver = ManeuverType.RANDOM

		self._current_maneuver = maneuver
		self._maneuver_steps_remaining = self._maneuver_duration
		self._maneuver_params = self._initialize_maneuver_params(maneuver)
		self.stats["maneuvers_started"][maneuver] += 1

	def _initialize_maneuver_params(self, maneuver):
		"""Initialize parameters for a specific maneuver type"""
		params = {}

		if maneuver == ManeuverType.HIGH_SPEED_TURN:
			params["steering_direction"] = np.random.choice([-1, 1])
			params["steering_magnitude"] = np.random.uniform(0.7, 1.0)
			params["throttle"] = np.random.uniform(0.6, 1.0) * np.random.choice([-1, 1])
			params["phase"] = "accelerate"  # Start by accelerating
			params["accel_steps"] = self._maneuver_duration // 2

		elif maneuver == ManeuverType.ZIGZAG:
			params["frequency"] = np.random.uniform(0.2, 0.5)
			params["amplitude"] = np.random.uniform(0.6, 0.9)
			params["throttle"] = np.random.uniform(0.5, 0.8) * np.random.choice([-1, 1])
			params["start_step"] = self._total_steps

		elif maneuver == ManeuverType.EMERGENCY_CORRECTION:
			params["initial_direction"] = np.random.choice([-1, 1])
			params["correction_step"] = self._maneuver_duration // 2
			params["throttle_change"] = np.random.choice(
				["brake", "neutral", "accel"]
			)

		elif maneuver == ManeuverType.ACCELERATION_BURST:
			params["throttle"] = np.random.uniform(0.8, 1.0) * np.random.choice([-1, 1])
			params["steering_noise"] = np.random.uniform(0.1, 0.3)

		elif maneuver == ManeuverType.DRIFT_ATTEMPT:
			params["entry_speed_steps"] = self._maneuver_duration // 3
			params["turn_direction"] = np.random.choice([-1, 1])
			params["turn_sharpness"] = np.random.uniform(0.8, 1.0)
			params["drift_throttle"] = np.random.uniform(0.4, 0.7) * np.random.choice([-1, 1])
			params["phase"] = "accelerate"

		return params

	def _execute_maneuver(self, observation):
		"""Execute the current maneuver and return action"""
		velocity = self._estimate_velocity(observation)

		if self._current_maneuver == ManeuverType.RANDOM:
			return self._action_space.sample()

		elif self._current_maneuver == ManeuverType.HIGH_SPEED_TURN:
			return self._execute_high_speed_turn(velocity)

		elif self._current_maneuver == ManeuverType.ZIGZAG:
			return self._execute_zigzag()

		elif self._current_maneuver == ManeuverType.EMERGENCY_CORRECTION:
			return self._execute_emergency_correction()

		elif self._current_maneuver == ManeuverType.ACCELERATION_BURST:
			return self._execute_acceleration_burst()

		elif self._current_maneuver == ManeuverType.DRIFT_ATTEMPT:
			return self._execute_drift_attempt(velocity)

		return self._action_space.sample()

	def _execute_high_speed_turn(self, velocity):
		"""Build speed then execute sharp turn"""
		p = self._maneuver_params

		if p["phase"] == "accelerate":
			# Accelerate in straight line
			steering = np.random.uniform(-0.2, 0.2)
			throttle = p["throttle"]

			# Switch to turning phase
			if self._maneuver_steps_remaining <= p["accel_steps"]:
				p["phase"] = "turn"
		else:
			# Execute the turn
			steering = p["steering_direction"] * p["steering_magnitude"]
			throttle = p["throttle"] * 0.8  # Slightly reduce throttle

		action = np.array([steering, throttle])
		return np.clip(action, self._action_space.low, self._action_space.high)

	def _execute_zigzag(self):
		"""Rapid steering changes at moderate speed"""
		p = self._maneuver_params
		steps_elapsed = self._total_steps - p["start_step"]

		steering = p["amplitude"] * np.sin(
			steps_elapsed * p["frequency"] * 2 * np.pi
		)
		throttle = p["throttle"]

		action = np.array([steering, throttle])
		return np.clip(action, self._action_space.low, self._action_space.high)

	def _execute_emergency_correction(self):
		"""Sudden steering reversal (simulates overcorrection)"""
		p = self._maneuver_params

		if self._maneuver_steps_remaining > p["correction_step"]:
			# Initial direction
			steering = p["initial_direction"] * np.random.uniform(0.7, 0.9)
		else:
			# Sudden correction in opposite direction
			steering = -p["initial_direction"] * np.random.uniform(0.8, 1.0)

		# Handle throttle
		if p["throttle_change"] == "brake":
			throttle = np.random.uniform(-1.0, -0.5)
		elif p["throttle_change"] == "neutral":
			throttle = np.random.uniform(-0.2, 0.2)
		else:
			throttle = np.random.uniform(0.5, 0.8)

		action = np.array([steering, throttle])
		return np.clip(action, self._action_space.low, self._action_space.high)

	def _execute_acceleration_burst(self):
		"""Aggressive acceleration with slight steering instability"""
		p = self._maneuver_params

		steering = np.random.uniform(-p["steering_noise"], p["steering_noise"])
		throttle = p["throttle"]

		action = np.array([steering, throttle])
		return np.clip(action, self._action_space.low, self._action_space.high)

	def _execute_drift_attempt(self, velocity):
		"""Attempt a drift maneuver: speed up then sharp turn"""
		p = self._maneuver_params

		if p["phase"] == "accelerate":
			# Build up speed
			steering = np.random.uniform(-0.15, 0.15)
			throttle = 0.9

			if self._maneuver_steps_remaining <= (
				self._maneuver_duration - p["entry_speed_steps"]
			):
				p["phase"] = "drift"
		else:
			# Execute drift: sharp turn with controlled throttle
			steering = p["turn_direction"] * p["turn_sharpness"]
			throttle = p["drift_throttle"]

		action = np.array([steering, throttle])
		return np.clip(action, self._action_space.low, self._action_space.high)

	def _estimate_velocity(self, observation):
		"""Estimate velocity from observation"""
		# Assumes velocity is at indices 6:9, adjust based on your env
		if len(observation) > 8:
			return np.linalg.norm(observation[6:9])
		return 0.0

	def reset(self):
		"""Reset policy state (call when episode ends)"""
		self._current_maneuver = ManeuverType.RANDOM
		self._maneuver_steps_remaining = 0
		self._maneuver_params = {}

	def get_statistics(self):
		"""Get exploration statistics"""
		stats = self.stats.copy()
		stats["aggressive_ratio"] = (
			stats["total_actions"]
			- stats["maneuvers_started"][ManeuverType.RANDOM]
		) / max(stats["total_actions"], 1)
		return stats

	def print_statistics(self):
		"""Print formatted statistics"""
		stats = self.get_statistics()
		print("\n=== Exploration Policy Statistics ===")
		print(f"Total actions: {stats['total_actions']}")
		print(f"Aggressive action ratio: {stats['aggressive_ratio']:.2%}")
		print("\nManeuvers started:")
		for maneuver, count in stats["maneuvers_started"].items():
			completed = stats["maneuvers_completed"][maneuver]
			print(
				f"  {maneuver.value:25s}: {count:4d} started, {completed:4d} completed"
			)
