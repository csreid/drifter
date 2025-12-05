import mujoco
import numpy as np
from collections import deque
import mediapy as media
from scipy.spatial.transform import Rotation
import cv2


class RCCarSimulation:
	def __init__(self, model_path="robot.xml", render=True, generated_terrain=False):
		"""
		Initialize the RC car simulation using MuJoCo

		Args:
		    model_path (str): Path to the MuJoCo XML model file
		    render (bool): Whether to enable rendering
		"""
		# Load model
		self.model = mujoco.MjModel.from_xml_path(model_path)
		self.data = mujoco.MjData(self.model)

		# Set up renderer if needed
		self.render_enabled = render

		# Control parameters
		self.max_torque = 2.0
		self.max_steering_angle = 0.6
		self.wheelbase = 0.535
		self.track_width = 0.281

		# State variables
		self.steering_angle = 0.0
		self.throttle = 0.0
		self.ebrake = False

		# Get actuator and sensor IDs
		self._get_ids()

		# Initialize goal position
		self.goal_pos = np.array([10.0, 10.0, 0.0])
		self._update_goal_position()

		# Velocity tracking for acceleration estimation
		self._prev_velocities = deque([np.array([0, 0, 0])], maxlen=10)
		self.flipped_frames = 0

		self._camera_renderer = mujoco.Renderer(self.model, 480, 480)

	def _get_ids(self):
		"""Get IDs for actuators, sensors, bodies, and geoms"""
		# Actuator IDs
		self.actuators = {
			"front_left_steering": mujoco.mj_name2id(
				self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_left_steering"
			),
			"front_right_steering": mujoco.mj_name2id(
				self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "front_right_steering"
			),
			"rear_left_drive": mujoco.mj_name2id(
				self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_drive"
			),
			"rear_right_drive": mujoco.mj_name2id(
				self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_drive"
			),
		}

		# Body IDs
		self.bodies = {
			"base_link": mujoco.mj_name2id(
				self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
			),
			"goal": mujoco.mj_name2id(
				self.model, mujoco.mjtObj.mjOBJ_BODY, "goal"
			),
		}

		# Camera ID
		self.camera_id = mujoco.mj_name2id(
			self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera_l"
		)

	def reset_simulation(self):
		"""Reset the simulation to a random starting state"""
		# Reset to default state
		mujoco.mj_resetData(self.model, self.data)

		# Set random car position
		new_x = np.random.uniform(-15, 15)
		new_y = np.random.uniform(-15, 15)
		new_z = 0.5

		car_qpos_idx = 0  # First 7 elements are free joint: pos (3) + quat (4)
		self.data.qpos[car_qpos_idx : car_qpos_idx + 3] = [new_x, new_y, new_z]
		self.data.qpos[car_qpos_idx + 3 : car_qpos_idx + 7] = [
			1,
			0,
			0,
			0,
		]  # Identity quat

		# Set random goal position
		goal_x = np.random.uniform(-15, 15)
		goal_y = np.random.uniform(-15, 15)
		self.goal_pos = np.array([goal_x, goal_y, 2.5])
		self._update_goal_position()

		# Reset velocities
		self.data.qvel[:] = 0

		# Reset control state
		self.steering_angle = 0.0
		self.throttle = 0.0
		self.ebrake = False

		# Clear velocity history
		self._prev_velocities.clear()
		self._prev_velocities.append(np.array([0, 0, 0]))

		self.flipped_frames = 0

		# Forward the simulation to settle
		mujoco.mj_forward(self.model, self.data)

	def _update_goal_position(self):
		"""Update the goal marker position in the simulation"""
		goal_body_id = self.bodies["goal"]
		self.model.body_pos[goal_body_id] = self.goal_pos

	def set_controls(self, steering_input, throttle_input, ebrake=False):
		"""
		Set control inputs for the car

		Args:
		    steering_input (float): Steering input from -1 (full left) to 1 (full right)
		    throttle_input (float): Throttle input from -1 (full reverse) to 1 (full forward)
		    ebrake (bool): Emergency brake activation
		"""
		steering_input = np.clip(steering_input, -1.0, 1.0)
		throttle_input = np.clip(throttle_input, -1.0, 1.0)

		self.steering_angle = steering_input * self.max_steering_angle
		self.throttle = throttle_input * self.max_torque
		self.ebrake = ebrake

	def step_simulation(self):
		"""
		Step the simulation forward by one frame

		Returns:
		    dict: Current state information
		"""
		# Calculate Ackermann steering angles
		left_angle, right_angle = self._ackermann_angles(self.steering_angle)

		# Apply steering
		self.data.ctrl[self.actuators["front_left_steering"]] = left_angle
		self.data.ctrl[self.actuators["front_right_steering"]] = right_angle

		# Apply drive torque or brake
		if self.ebrake:
			# Simple brake implementation: apply damping
			self.data.ctrl[self.actuators["rear_left_drive"]] = 0
			self.data.ctrl[self.actuators["rear_right_drive"]] = 0
			# Add additional damping through qfrc_applied
			rl_joint_id = mujoco.mj_name2id(
				self.model, mujoco.mjtObj.mjOBJ_JOINT, "rear_left_wheel_joint"
			)
			rr_joint_id = mujoco.mj_name2id(
				self.model, mujoco.mjtObj.mjOBJ_JOINT, "rear_right_wheel_joint"
			)
			rl_qvel_idx = self.model.jnt_dofadr[rl_joint_id]
			rr_qvel_idx = self.model.jnt_dofadr[rr_joint_id]
			self.data.qfrc_applied[rl_qvel_idx] = (
				-100.0 * self.data.qvel[rl_qvel_idx]
			)
			self.data.qfrc_applied[rr_qvel_idx] = (
				-100.0 * self.data.qvel[rr_qvel_idx]
			)
		else:
			self.data.ctrl[self.actuators["rear_left_drive"]] = self.throttle
			self.data.ctrl[self.actuators["rear_right_drive"]] = self.throttle

		# Step simulation
		mujoco.mj_step(self.model, self.data)

		return self.get_state()

	def get_state(self):
		"""
		Get current car state without stepping simulation

		Returns:
		    dict: Current state information
		"""
		# Get car position and orientation
		car_body_id = self.bodies["base_link"]
		car_pos = self.data.xpos[car_body_id].copy()
		car_quat = self.data.xquat[car_body_id].copy()  # [w, x, y, z] in MuJoCo

		# Convert quaternion to rotation matrix
		R = Rotation.from_quat(
			[car_quat[1], car_quat[2], car_quat[3], car_quat[0]]
		).as_matrix()

		# Get velocities (in world frame from sensor)
		world_vel = self.data.sensordata[7:10].copy()
		world_angvel = self.data.sensordata[10:13].copy()

		# Transform to local frame
		local_vel = R.T @ world_vel
		local_angvel = R.T @ world_angvel

		# Update velocity history
		self._prev_velocities.append(local_vel)

		# Calculate local goal position
		local_goal_pos = R.T @ (self.goal_pos - car_pos)

		return {
			"position": tuple(car_pos),
			"orientation": tuple(
				[car_quat[1], car_quat[2], car_quat[3], car_quat[0]]
			),
			"goal_pos": tuple(self.goal_pos),
			"velocity_world": tuple(world_vel),
			"velocity_local": tuple(local_vel),
			"angular_velocity_world": tuple(world_angvel),
			"angular_velocity_local": tuple(local_angvel),
			"speed": np.linalg.norm(local_vel),
			"steering_angle": self.steering_angle,
			"throttle": self.throttle,
			"ebrake": self.ebrake,
			"is_flipped": self.is_flipped(),
			"local_goal_pos": tuple(local_goal_pos),
			"camera_img": self.capture_front_camera(),
		}

	def is_flipped(self, flip_threshold=0.3):
		"""Check if the car is flipped over"""
		car_body_id = self.bodies["base_link"]
		car_quat = self.data.xquat[car_body_id]  # [w, x, y, z]

		# Convert to rotation matrix
		R = Rotation.from_quat(
			[car_quat[1], car_quat[2], car_quat[3], car_quat[0]]
		).as_matrix()

		# Car's up vector in world coordinates
		car_up_vector = R[:, 2]
		world_up_vector = np.array([0, 0, 1])

		# Calculate dot product
		dot_product = np.dot(car_up_vector, world_up_vector)

		if dot_product < flip_threshold:
			self.flipped_frames += 1
		else:
			self.flipped_frames = 0

		return self.flipped_frames >= 10

	def capture_front_camera(self):
		"""
		Capture an RGB image from the front-facing camera

		Returns:
		    numpy.ndarray: RGB image with shape (height, width, 3)
		"""
		# Update the renderer with current state
		self._camera_renderer.update_scene(self.data, camera=self.camera_id)

		# Render and return RGB image
		rgb_array = self._camera_renderer.render()

		return rgb_array

	def render_camera_image(self):
		rgb_image = self.capture_front_camera()
		bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
		cv2.imshow("Camera Image", bgr_image)
		cv2.waitKey(1)  # Non-blocking wait

	def close(self):
		"""Clean up resources"""
		if hasattr(self, "renderer"):
			del self.renderer

	def get_car_dimensions(self):
		"""Get car dimensions (wheelbase, track_width)"""
		return (self.wheelbase, self.track_width)

	def _ackermann_angles(self, angle):
		"""Calculate individual wheel steering angles using Ackermann geometry"""
		if abs(angle) < 1e-6:
			return 0.0, 0.0

		wheelbase, track_width = self.get_car_dimensions()
		turning_radius = wheelbase / np.tan(abs(angle))

		if angle > 0:  # Left turn
			left_angle = np.arctan(
				wheelbase / (turning_radius - track_width / 2)
			)
			right_angle = np.arctan(
				wheelbase / (turning_radius + track_width / 2)
			)
			return left_angle, right_angle
		else:  # Right turn
			left_angle = -np.arctan(
				wheelbase / (turning_radius + track_width / 2)
			)
			right_angle = -np.arctan(
				wheelbase / (turning_radius - track_width / 2)
			)
			return left_angle, right_angle


# Example usage
if __name__ == "__main__":
	# Create simulation
	sim = RCCarSimulation(render=True)

	try:
		frames = []
		# Run for 1000 steps with simple control pattern
		for i in range(1000):
			# Simple test: go forward and turn left
			steering = 0.5 if i > 100 else 0.0
			throttle = 0.3

			# Set controls
			sim.set_controls(steering, throttle)

			# Step simulation and get state
			state = sim.step_simulation()
			# sim.render_camera_image()

			# Render and collect frames
			frame = sim.render()
			if frame is not None:
				cv2.imshow("Simulation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
				cv2.waitKey(1)

			# Print state every 100 steps
			if i % 100 == 0:
				print(
					f"Step {i}: Speed = {state['speed']:.2f}, Position = {state['position']}"
				)

		# Save video
		if frames:
			media.write_video("simulation.mp4", frames, fps=30)
			print("Video saved as simulation.mp4")

	finally:
		sim.close()
