import pybullet as p
import pybullet_data
import numpy as np
import xacro
import tempfile
from terrain import generate_terrain
from collections import deque
import cv2


def _grid_sampling(
	spawn_radius, tree_spacing=3.0, min_distance_from_origin=5.0
):
	"""Much faster tree placement using grid with random noise"""
	points = []

	# Create approximate grid
	for x in np.arange(-spawn_radius, spawn_radius, tree_spacing):
		for y in np.arange(-spawn_radius, spawn_radius, tree_spacing):
			# Add random offset
			actual_x = x + np.random.uniform(
				-tree_spacing * 0.3, tree_spacing * 0.3
			)
			actual_y = y + np.random.uniform(
				-tree_spacing * 0.3, tree_spacing * 0.3
			)

			distance = np.sqrt(actual_x**2 + actual_y**2)
			if min_distance_from_origin <= distance <= spawn_radius:
				# Randomly skip some trees for more natural look
				if np.random.random() > 0.2:  # Skip 20% of positions
					points.append((actual_x, actual_y))

	return points


def _poisson_disk_sampling(
	spawn_radius, min_distance, min_distance_from_origin=5.0
):
	"""Generate points with minimum distance using Poisson disk sampling"""
	points = []
	active_list = []

	# Start with random point in circular area, respecting min distance from origin
	while True:
		angle = np.random.uniform(0, 2 * np.pi)
		r = np.random.uniform(min_distance_from_origin, spawn_radius)
		first_point = (r * np.cos(angle), r * np.sin(angle))
		break

	points.append(first_point)
	active_list.append(first_point)

	while active_list:
		# Pick random active point
		idx = np.random.randint(len(active_list))
		current_point = active_list[idx]

		found_valid = False
		# Try to generate new points around current point
		for _ in range(30):  # Max attempts per point
			angle = np.random.uniform(0, 2 * np.pi)
			distance = np.random.uniform(min_distance, 2 * min_distance)

			new_x = current_point[0] + distance * np.cos(angle)
			new_y = current_point[1] + distance * np.sin(angle)

			# Check if within spawn radius
			distance_from_origin = np.sqrt(new_x**2 + new_y**2)
			if (
				distance_from_origin > spawn_radius
				or distance_from_origin < min_distance_from_origin
			):
				continue

			# Check distance from existing points
			valid = True
			for px, py in points:
				if (
					np.sqrt((new_x - px) ** 2 + (new_y - py) ** 2)
					< min_distance
				):
					valid = False
					break

			if valid:
				new_point = (new_x, new_y)
				points.append(new_point)
				active_list.append(new_point)
				found_valid = True
				break

		if not found_valid:
			active_list.pop(idx)

	return points


class RCCarSimulation:
	def __init__(self, generated_terrain=True, gui=True):
		"""
		Initialize the RC car simulation core

		Args:
		    gui (bool): Whether to show the PyBullet GUI
		"""
		self.generated_terrain = generated_terrain

		# Physics setup
		self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
		self.frame_time = 1.0 / 240.0
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)
		p.setPhysicsEngineParameter(
			fixedTimeStep=self.frame_time,
			numSubSteps=4,
			numSolverIterations=50,
			enableConeFriction=1,
			contactBreakingThreshold=0.001,
			erp=0.1,
			contactERP=0.1,
			frictionERP=0.2,
			maxNumCmdPer1ms=1000,
		)

		if gui:
			p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
			p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

		# Control parameters
		self.max_torque = 2.0
		self.max_steering_force = 50.0
		self.max_wheel_velocity = 10000.0
		self.max_steering_angle = 0.6

		self.spring_stiffness = 1000.0
		self.spring_damping = 50.0
		self.spring_rest_length = 0.04

		self.suspend = False

		# State variables
		self.steering_angle = 0.0
		self.throttle = 0.0
		self.ebrake = False

		# Joint indices (will be populated after loading URDF)
		self.joints = {}
		self.links = {}

		# Load world and car
		self._setup_world()
		self._load_car()
		self._setup_physics()
		if gui:
			self._setup_camera()

		self._prev_velocities = deque([np.array([0, 0, 0])], maxlen=10)

		self.flipped_frames = 0

	def reset_simulation(self):
		new_x = np.random.uniform(-32, 32)
		new_y = np.random.uniform(-32, 32)
		new_z = self.get_height_at(new_x, new_y) + 0.5
		new_pos = [new_x, new_y, new_z]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])

		# Reset car position and orientation
		p.resetBasePositionAndOrientation(
			self.car_id, new_pos, start_orientation
		)

		goal_x = np.random.uniform(-64, 64)
		goal_y = np.random.uniform(-64, 64)
		goal_z = self.get_height_at(goal_x, goal_y)
		new_goal_pos = [goal_x, goal_y, goal_z]
		p.resetBasePositionAndOrientation(
			self.goal_id, new_goal_pos, start_orientation
		)

		# Reset car velocity
		p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0])

		# Reset all joint states
		for joint_id in self.joints.values():
			p.resetJointState(
				self.car_id, joint_id, targetValue=0, targetVelocity=0
			)

		# Reset control state variables
		self.steering_angle = 0.0
		self.throttle = 0.0
		self.ebrake = False

		# Clear velocity history for acceleration estimation
		self._prev_velocities.clear()
		self._prev_velocities.append(np.array([0, 0, 0]))

		self.flipped_frames = 0

	def get_height_at(self, x, y):
		ray_start = [x, y, 100]  # Start ray high above
		ray_end = [x, y, -100]  # End ray below ground
		ray_result = p.rayTest(ray_start, ray_end)
		if ray_result[0][0] != -1:  # Hit something
			ground_z = ray_result[0][3][2]  # Hit position Z coordinate
		else:
			ground_z = 0  # Fallback to zero if no hit

		return ground_z

	def _spawn_goal(self):
		x = np.random.uniform(-64, 64)
		y = np.random.uniform(-64, 64)
		z = self.get_height_at(x, y)

		goal_height = 5.0
		goal_shape = p.createCollisionShape(
			p.GEOM_CYLINDER, radius=0.2, height=goal_height
		)
		goal_visual = p.createVisualShape(
			p.GEOM_CYLINDER,
			radius=1.0,
			length=goal_height,
			rgbaColor=[0.9, 0.0, 0.3, 0.8],
		)

		goal_id = p.createMultiBody(
			baseMass=0,  # Static object
			baseCollisionShapeIndex=goal_shape,
			baseVisualShapeIndex=goal_visual,
			basePosition=[x, y, z + goal_height * 0.5],
		)

		return goal_id

	def _spawn_random_trees(
		self, spawn_radius=50.0, min_distance_from_origin=1.0
	):
		tree_points = _grid_sampling(
			spawn_radius,
			2.0,
			min_distance_from_origin,
		)

		tree_ids = []

		tree_templates = []
		for i in range(3):  # 3 different tree types
			trunk_radius = 0.1 + i * 0.1
			crown_radius = 0.8 + i * 0.3

			trunk_shape = p.createCollisionShape(
				p.GEOM_CYLINDER, radius=trunk_radius, height=2.0
			)
			trunk_visual = p.createVisualShape(
				p.GEOM_CYLINDER,
				radius=trunk_radius,
				length=3.0,
				rgbaColor=[0.4, 0.2, 0.1, 1.0],
			)
			crown_shape = p.createCollisionShape(
				p.GEOM_SPHERE, radius=crown_radius
			)
			crown_visual = p.createVisualShape(
				p.GEOM_SPHERE,
				radius=crown_radius,
				rgbaColor=[0.1, 0.6, 0.1, 1.0],
			)

			tree_templates.append(
				(trunk_shape, trunk_visual, crown_shape, crown_visual)
			)

		tree_points = _grid_sampling(spawn_radius, 3, min_distance_from_origin)
		tree_ids = []

		for x, y in tree_points:
			ray_start = [x, y, 100]  # Start ray high above
			ray_end = [x, y, -100]  # End ray below ground
			ray_result = p.rayTest(ray_start, ray_end)

			hit_trees = [rr[0] in tree_ids for rr in ray_result]

			if any(hit_trees):
				continue

			tmpl_idx = np.random.randint(len(tree_templates))

			(
				trunk_shape,
				trunk_visual,
				crown_shape,
				crown_visual,
			) = tree_templates[tmpl_idx]

			if ray_result[0][0] != -1:  # Hit something
				ground_z = ray_result[0][3][2]  # Hit position Z coordinate
			else:
				ground_z = 0  # Fallback to zero if no hit

			tree_height = 2.0
			# Create multi-body tree
			tree_id = p.createMultiBody(
				baseMass=0,  # Static object
				baseCollisionShapeIndex=trunk_shape,
				baseVisualShapeIndex=trunk_visual,
				basePosition=[
					x,
					y,
					ground_z + tree_height * 0.5,
				],  # Half trunk height above ground
				linkMasses=[0],
				linkCollisionShapeIndices=[crown_shape],
				linkVisualShapeIndices=[crown_visual],
				linkPositions=[
					[0, 0, tree_height * 0.4]
				],  # Crown offset above trunk
				linkOrientations=[[0, 0, 0, 1]],
				linkInertialFramePositions=[[0, 0, 0]],
				linkInertialFrameOrientations=[[0, 0, 0, 1]],
				linkParentIndices=[0],
				linkJointTypes=[p.JOINT_FIXED],
				linkJointAxis=[[0, 0, 0]],
			)

			tree_ids.append(tree_id)

		return tree_ids

	def set_controls(self, steering_input, throttle_input, ebrake=False):
		"""
		Set control inputs for the car

		Args:
		    steering_input (float): Steering input from -1 (full left) to 1 (full right)
		    throttle_input (float): Throttle input from -1 (full reverse) to 1 (full forward)
		    ebrake (bool): Emergency brake activation
		"""
		# Clamp inputs to valid range
		steering_input = np.clip(steering_input, -1.0, 1.0)
		throttle_input = np.clip(throttle_input, -1.0, 1.0)

		# Convert normalized inputs to actual control values
		self.steering_angle = steering_input * self.max_steering_angle
		self.throttle = throttle_input * self.max_torque
		self.ebrake = ebrake

	def step_simulation(self):
		"""
		Step the simulation forward by one frame

		Returns:
		    dict: Current state information including position, velocity, acceleration
		"""
		# Apply suspension forces if enabled
		if self.suspend:
			self._apply_suspension_forces()

		# Apply controls to car
		self._control_car()

		# Step physics simulation
		p.stepSimulation()

		# Update camera to follow car (if GUI is enabled)
		# if self.physics_client == p.GUI:
		# self.update_camera()

		# Return current state
		return self.get_state()

	def capture_front_camera(
		self,
		image_width=256,
		image_height=256,
		fov=60,
		near_plane=0.1,
		far_plane=200.0,
	):
		"""
		Capture a 256x256 RGB image from a camera mounted on the front of the car

		Args:
				image_width (int): Width of captured image in pixels (default: 256)
				image_height (int): Height of captured image in pixels (default: 256)
				fov (float): Field of view in degrees (default: 60)
				near_plane (float): Near clipping plane distance (default: 0.1)
				far_plane (float): Far clipping plane distance (default: 50.0)

		Returns:
				numpy.ndarray: RGB image as numpy array with shape (height, width, 3) and values 0-255
		"""
		# Get car's current position and orientation
		car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id)

		# Convert quaternion to rotation matrix
		rotation_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(
			3, 3
		)

		# Define camera offset from car center (mounted on front bumper)
		# Adjust these values based on your car's dimensions and desired camera placement
		camera_offset_local = np.array(
			[0.3, 0.0, 0.1]
		)  # Forward, right, up in car's local frame

		# Transform camera offset to world coordinates
		camera_offset_world = rotation_matrix @ camera_offset_local
		camera_pos = np.array(car_pos) + camera_offset_world

		# Calculate camera target (where the camera is looking)
		# Look ahead in the car's forward direction
		look_ahead_distance = 10.0
		forward_direction = rotation_matrix @ np.array(
			[1.0, 0.0, 0.0]
		)  # Car's forward direction
		camera_target = camera_pos + forward_direction * look_ahead_distance

		# Define camera up vector (usually just world up, but could be car's up vector)
		up_vector = rotation_matrix @ np.array(
			[0.0, 0.0, 1.0]
		)  # Car's up direction

		# Compute view matrix
		view_matrix = p.computeViewMatrix(
			cameraEyePosition=camera_pos.tolist(),
			cameraTargetPosition=camera_target.tolist(),
			cameraUpVector=up_vector.tolist(),
		)

		# Compute projection matrix
		aspect_ratio = image_width / image_height
		projection_matrix = p.computeProjectionMatrixFOV(
			fov=fov, aspect=aspect_ratio, nearVal=near_plane, farVal=far_plane
		)

		# Render the image
		width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
			width=image_width,
			height=image_height,
			viewMatrix=view_matrix,
			projectionMatrix=projection_matrix,
			# renderer=p.ER_BULLET_HARDWARE_OPENGL,  # Use hardware acceleration if available
			renderer=p.ER_TINY_RENDERER,
		)

		# Convert RGBA to RGB and reshape
		rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(
			(image_height, image_width, 4)
		)
		rgb_array = rgb_array[:, :, :3]  # Remove alpha channel

		return rgb_array

	def get_state(self):
		"""
		Get current car state without stepping simulation

		Returns:
		    dict: Current state information
		"""
		world_vel, world_angularvel = p.getBaseVelocity(self.car_id)
		pos, orn = p.getBasePositionAndOrientation(self.car_id)
		R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
		goal_pos, _ = p.getBasePositionAndOrientation(self.goal_id)

		local_vel = R.T @ np.array(np.array(world_vel))
		local_angularvel = R.T @ np.array(world_angularvel)

		return {
			"position": pos,
			"orientation": orn,
			"goal_pos": goal_pos,
			"velocity_world": world_vel,
			"velocity_local": local_vel,
			"angular_velocity_world": world_angularvel,
			"angular_velocity_local": local_angularvel,
			"speed": np.linalg.norm(local_vel),
			"steering_angle": self.steering_angle,
			"throttle": self.throttle,
			"ebrake": self.ebrake,
			"is_flipped": self.is_flipped(),
			"local_goal_pos": R.T @ (np.array(goal_pos) - np.array(pos)),
		}

	def close(self):
		"""Clean up and close the simulation"""
		p.disconnect()

	def get_car_dimensions(self):
		return (0.25, 0.25)  # wheelbase, track_width

	def _estimate_acc(self):
		estimates = []
		for i in range(0, len(self._prev_velocities) - 1):
			vi = self._prev_velocities[i]
			vj = self._prev_velocities[i + 1]
			estimates.append((vj - vi) / (self.frame_time))

		if len(estimates) > 1:
			return np.mean(estimates, axis=0)
		return estimates[0] if estimates else np.array([0, 0, 0])

	def _ackermann_angles(self, angle):
		if abs(angle) < 1e-6:
			return 0.0, 0.0

		wheelbase, track_width = self.get_car_dimensions()
		turning_radius = wheelbase / np.tan(abs(angle))

		if angle > 0:  # Left turn
			left_angle = np.atan(wheelbase / (turning_radius - track_width / 2))
			right_angle = np.atan(
				wheelbase / (turning_radius + track_width / 2)
			)
			return left_angle, right_angle
		else:  # Right turn
			left_angle = -np.atan(
				wheelbase / (turning_radius + track_width / 2)
			)
			right_angle = -np.atan(
				wheelbase / (turning_radius - track_width / 2)
			)
			return left_angle, right_angle

	def _setup_world(self):
		"""Load the ground plane and obstacles"""
		if self.generated_terrain:
			hmap = generate_terrain(128, 128, scale=15)

			ground_shape = p.createCollisionShape(
				shapeType=p.GEOM_HEIGHTFIELD,
				meshScale=[1, 1, 1],
				heightfieldData=hmap.flatten(),
				heightfieldTextureScaling=128,
				numHeightfieldRows=128,
				numHeightfieldColumns=128,
			)

			self.plane_id = p.createMultiBody(
				baseMass=0,
				baseCollisionShapeIndex=ground_shape,
				basePosition=[0, 0, 0],
			)
		else:
			self.plane_id = p.loadURDF("plane.urdf")

		self.goal_id = self._spawn_goal()

		# self.tree_ids = self._spawn_random_trees()

	def _load_car(self):
		"""Load the RC car URDF and get joint information"""
		start_pos = [0, 0, 2.0]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])

		with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf") as f:
			urdf_str = xacro.process_file(
				"robot.xacro",
				mappings={"suspend": "true" if self.suspend else "false"},
			).toxml()
			f.write(urdf_str)
			f.flush()

			self.car_id = p.loadURDF(f.name, start_pos, start_orientation)

		# Get joint information
		num_joints = p.getNumJoints(self.car_id)
		p.changeDynamics(
			self.car_id, -1, linearDamping=0.005, angularDamping=0.005
		)

		for i in range(num_joints):
			joint_info = p.getJointInfo(self.car_id, i)
			joint_name = joint_info[1].decode("utf-8")
			link_name = joint_info[12].decode("utf-8")
			self.joints[joint_name] = i
			self.links[link_name] = i

	def _setup_physics(self):
		"""Configure physics properties and disable default motors"""
		joints = [
			"front_left_wheel_joint",
			"front_right_wheel_joint",
			"rear_left_wheel_joint",
			"rear_right_wheel_joint",
		]

		if self.suspend:
			joints += [
				"front_left_suspension",
				"front_right_suspension",
				"rear_left_suspension",
				"rear_right_suspension",
			]

		front_wheels = ["front_left_wheel", "front_right_wheel"]
		rear_wheels = ["rear_left_wheel", "rear_right_wheel"]

		for joint_name in joints:
			if joint_name in self.joints:
				joint_id = self.joints[joint_name]
				p.setJointMotorControl2(
					self.car_id,
					joint_id,
					p.VELOCITY_CONTROL,
					targetVelocity=0,
					force=0,
				)
				p.changeDynamics(
					self.car_id,
					joint_id,
					jointLimitForce=1000,
					maxJointVelocity=10000,
				)

		for wheel_name in front_wheels + rear_wheels:
			lid = self.links[wheel_name]
			p.changeDynamics(
				self.car_id,
				lid,
				lateralFriction=0.9,
				rollingFriction=0.001,
				spinningFriction=0.001,
			)

		# Set ground friction
		p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)

	def _setup_camera(self):
		"""Set initial camera position"""
		p.resetDebugVisualizerCamera(
			cameraDistance=2.0,
			cameraYaw=30,
			cameraPitch=-30,
			cameraTargetPosition=[0, 0, 0],
		)

	def _get_joint_id(self, joint_name):
		"""Get joint ID by name, with error checking"""
		return self.joints.get(joint_name)

	def _apply_velocity_limiting(self, joint_name, requested_torque):
		"""Apply velocity limiting to prevent wheels from spinning too fast"""
		joint_id = self._get_joint_id(joint_name)
		if joint_id is None:
			return 0

		current_velocity = p.getJointState(self.car_id, joint_id)[1]

		if abs(current_velocity) > self.max_wheel_velocity:
			return 0
		elif abs(current_velocity) > self.max_wheel_velocity * 0.8:
			reduction_factor = 1.0 - (
				(abs(current_velocity) - self.max_wheel_velocity * 0.8)
				/ (self.max_wheel_velocity * 0.2)
			)
			return requested_torque * reduction_factor
		else:
			return requested_torque

	def _control_car(self):
		"""Apply steering and throttle controls to the car"""
		l_angle, r_angle = self._ackermann_angles(self.steering_angle)

		fl = self._get_joint_id("front_left_steering_joint")
		fr = self._get_joint_id("front_right_steering_joint")

		for j, a in zip([fl, fr], [l_angle, r_angle]):
			if j is not None:
				p.setJointMotorControl2(
					self.car_id,
					j,
					p.POSITION_CONTROL,
					targetPosition=a,
					force=self.max_steering_force,
				)

		# Apply drive torque to rear wheels with velocity limiting
		for joint_name in ["rear_left_wheel_joint", "rear_right_wheel_joint"]:
			joint_id = self._get_joint_id(joint_name)
			if joint_id is not None:
				if self.ebrake:
					p.setJointMotorControl2(
						self.car_id,
						joint_id,
						p.VELOCITY_CONTROL,
						targetVelocity=0.0,
						force=100.0,
					)
				else:
					p.setJointMotorControl2(
						self.car_id, joint_id, p.VELOCITY_CONTROL, force=0.0
					)
					limited_torque = self._apply_velocity_limiting(
						joint_name, self.throttle
					)
					p.setJointMotorControl2(
						self.car_id,
						joint_id,
						p.TORQUE_CONTROL,
						force=limited_torque,
					)

	def _apply_suspension_forces(self):
		"""Manually apply spring-damper forces to suspension joints"""
		suspension_joints = [
			"front_left_suspension",
			"front_right_suspension",
			"rear_left_suspension",
			"rear_right_suspension",
		]

		for joint_name in suspension_joints:
			joint_id = self._get_joint_id(joint_name)
			if joint_id is not None:
				joint_state = p.getJointState(self.car_id, joint_id)
				position = joint_state[0]
				velocity = joint_state[1]

				# Rear springs 2x as stiff as front
				scale = 2 if joint_name.startswith("rear") else 1

				# Calculate spring-damper force
				spring_force = (
					-self.spring_stiffness
					* (position - self.spring_rest_length)
					* scale
				)
				damper_force = -self.spring_damping * velocity
				total_force = spring_force + damper_force

				# Apply force to suspension joint
				p.setJointMotorControl2(
					self.car_id, joint_id, p.TORQUE_CONTROL, force=total_force
				)

	def render_camera_image(self):
		try:
			# Capture camera image
			rgb_image = self.capture_front_camera(
				image_width=320, image_height=240
			)

			# Convert RGB to BGR for OpenCV
			bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

			# Add some overlay information
			state = self.get_state()
			speed_kmh = state["speed"] * 3.6

			# Display the image
			cv2.imshow("Capture Image", bgr_image)
			cv2.waitKey(1)  # Non-blocking wait

		except Exception as e:
			print(f"Error displaying camera feed: {e}")

	def is_flipped(self, flip_threshold=0.3):
		_, car_orn = p.getBasePositionAndOrientation(self.car_id)

		# Get rotation matrix from quaternion
		rotation_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape(
			3, 3
		)

		# Car's up vector in world coordinates (third column of rotation matrix)
		car_up_vector = rotation_matrix[:, 2]  # Z-axis in car's local frame
		world_up_vector = np.array([0, 0, 1])  # World up direction

		# Calculate dot product (cosine of angle between vectors)
		dot_product = np.dot(car_up_vector, world_up_vector)

		# If dot product is below threshold, car is flipped
		if dot_product < flip_threshold:
			self.flipped_frames += 1
		else:
			self.flipped_frames = 0

		# Return True if flipped for consecutive frames
		return self.flipped_frames >= 10

	def update_camera(self):
		"""Update camera to show car in foreground with goal in background"""
		car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id)
		goal_pos, _ = p.getBasePositionAndOrientation(self.goal_id)

		# Convert to numpy arrays for easier math
		car_pos = np.array(car_pos)

		car_euler = p.getEulerFromQuaternion(car_orn)
		car_yaw_radians = car_euler[2]

		camera_yaw = np.degrees(car_yaw_radians) - 90

		# Use original camera settings but with calculated yaw
		p.resetDebugVisualizerCamera(
			cameraDistance=2.0,
			cameraYaw=camera_yaw,
			cameraPitch=-30,
			cameraTargetPosition=car_pos.tolist(),
		)


# Example usage
if __name__ == "__main__":
	import time

	# Create simulation
	sim = RCCarSimulation(gui=True)

	try:
		# Run for 1000 steps with simple control pattern
		for i in range(1000):
			# Simple test: go forward and turn left
			steering = 0.5 if i > 100 else 0.0  # Start turning after 100 steps
			throttle = 0.3  # Constant forward throttle

			# Set controls
			sim.set_controls(steering, throttle)

			# Step simulation and get state
			state = sim.step_simulation()
			# time.sleep(1.0 / 240.0)

			# Print state every 100 steps
			if i % 100 == 0:
				print(
					f"Step {i}: Speed = {state['speed']:.2f}, Position = {state['position']}"
				)

	finally:
		sim.close()
