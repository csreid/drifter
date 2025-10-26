import time
import numpy as np
from simulation import RCCarSimulation
from gymnasium import spaces
import gymnasium as gym
import time

observation_space = gym.spaces.Box(
	low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
)
action_space = spaces.Box(
	low=np.array([-1.0, -1.0]),
	high=np.array([1.0, 1.0]),
	shape=(2,),
	dtype=np.float32,
)


class DrifterEnv(gym.Env):
	def __init__(self, action_duration=0.1, gui: bool = False):
		self.gui = gui

		self.max_episode_steps = 1000
		self.current_step = 0

		# self.observation_space = gym.spaces.Dict(
		# {
		# "state": gym.spaces.Box(
		# low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
		# ),
		# "camera": gym.spaces.Box(
		# low=0, high=255, shape=(240, 320, 3), dtype=np.uint8
		# ),
		# }
		# )

		self.observation_space = observation_space

		# self.action_space = spaces.Box(
		# low=1., high=1., shape=(2,), dtype=np.float32
		# )

		self.action_space = action_space

		self.sim = RCCarSimulation(gui=self.gui, generated_terrain=True)
		self.n_substeps = int(240 * action_duration)
		self.prev_timestamp = time.time()
		self._realtime = False

	def _realtime_sleep(self):
		cur_timestamp = time.time()
		time_between_steps = 1.0 / 24.0
		time_since_last_step = cur_timestamp - self.prev_timestamp
		time_to_sleep = time_between_steps - time_since_last_step

		if time_to_sleep > 0:
			time.sleep(time_to_sleep)
		else:
			print(f"WARNING: running slower than real time")

		self.prev_timestamp = time.time()

	def _get_obs(self):
		simstate = self.sim.get_state()

		pos = np.array(simstate["position"])
		local_goal_pos = np.array(simstate["local_goal_pos"])
		goal_pos = np.array(simstate["goal_pos"])
		orn = np.array(simstate["orientation"])
		# omega = np.array(simstate["angular_velocity_local"])
		vel = np.array(simstate["velocity_local"])
		is_flipped = np.array([simstate["is_flipped"]]).astype("float")

		# oned_obs = [pos, local_goal_pos, vel, is_flipped, goal_pos, orn,]
		oned_obs = [
			pos,
			local_goal_pos,
			vel,
			is_flipped,
			goal_pos,
			orn,
		]
		# oned_obs = [pos, orn, vel]
		obs = np.concatenate(oned_obs)

		# camera = self.sim.capture_front_camera(
		# image_width=320, image_height=240
		# )

		# return {"state": obs, "camera": camera}
		return obs

	def _current_distance(self):
		simstate = self.sim.get_state()
		goal_position = np.array(simstate["goal_pos"])
		pos = np.array(simstate["position"])

		distance = np.linalg.norm(goal_position - pos)

		return distance

	def _compute_reward(self):
		simstate = self.sim.get_state()

		pos = np.array(simstate["position"])
		goal_position = np.array(simstate["goal_pos"])
		velocity = np.array(simstate["velocity_world"])

		# Calculate distance to goal
		distance = self._current_distance()

		# Direction from car to goal (normalized)
		if distance > 0:
			goal_direction = (goal_position - pos) / distance
		else:
			goal_direction = np.array([0, 0, 0])

		# Velocity magnitude
		speed = np.linalg.norm(velocity)

		# Velocity direction (normalized)
		if speed > 0:
			velocity_direction = velocity / speed
		else:
			velocity_direction = np.array([0, 0, 0])

		# --- Reward Components ---

		# 1. Distance reward (negative, encourages moving closer)
		distance_reward = -distance / 10.0

		# 2. Speed reward (encourages moving fast)
		speed_reward = speed * 0.1

		# 3. Direction alignment reward
		# Dot product gives cosine of angle between velocity and goal direction
		direction_alignment = np.dot(
			velocity_direction[:2], goal_direction[:2]
		)  # Only x,y components
		direction_reward = direction_alignment * speed * 0.3  # Scale by speed

		# 4. Goal reached bonus
		goal_bonus = 1000 if (distance < 2.0) else 0

		# 5. Progress reward (reward for getting closer than before)
		if not hasattr(self, "previous_distance"):
			self.previous_distance = distance

		progress = self.previous_distance - distance
		progress_reward = progress  # Scale factor
		self.previous_distance = distance

		# 6. Flip penalty (severe)
		flip_penalty = -1000 if self.sim.is_flipped() else 0

		# 7. Time penalty (small, encourages faster completion)
		time_penalty = -0.1

		# Combine all rewards
		total_reward = (
			distance_reward
			+
			# speed_reward +
			# direction_reward +
			goal_bonus
			+ flip_penalty
			# + time_penalty
		)

		return total_reward

	def _is_done(self):
		simstate = self.sim.get_state()

		pos = np.array(simstate["position"])
		goal_position = np.array(simstate["goal_pos"])
		distance = np.linalg.norm(goal_position - pos)

		if distance < 2.0 or self.sim.is_flipped():
			return True

		return False

	def _is_truncated(self):
		return self.current_step > 100

	def step(self, action):
		self.current_step += 1
		self.sim.update_camera()
		# self.sim.render_camera_image()

		steering, throttle = action
		self.sim.set_controls(steering, throttle)
		for _ in range(self.n_substeps):
			self.sim.step_simulation()

		obs = self._get_obs()
		reward = self._compute_reward()
		done = self._is_done()
		truncated = self._is_truncated()
		info = {}

		if self._realtime:
			self._realtime_sleep()

		return obs, reward, done, truncated, info

	def set_realtime(self, val: bool):
		self._realtime = val

	def reset(self, *args, **kwargs):
		self.current_step = 0
		self.sim.reset_simulation()
		self.initial_distance = self._current_distance()

		obs = self._get_obs()

		return obs, {}


if __name__ == "__main__":
	# Create environment
	env = DrifterEnv(gui=True)

	# Test random policy
	obs, info = env.reset()
	print(f"Observation shape: {obs.shape}")
	print(f"Action space: {env.action_space}")

	for step in range(100000):
		action = env.action_space.sample()  # Random action
		obs, reward, done, truncated, info = env.step(action)
		time.sleep(1 / 240.0)

		if done or truncated:
			print(f"Episode ended at step {step}")
			print(f"Success: {info['success']}")
			obs, info = env.reset()
