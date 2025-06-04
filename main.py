from io import StringIO
import pybullet as p
import pybullet_data
import time
import numpy as np
import xacro
import tempfile
from terrain import generate_terrain
import pygame
import os


class RCCarSimulation:
	def __init__(self):
		# Physics setup
		self.physics_client = p.connect(p.GUI)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)
		p.setPhysicsEngineParameter(
			fixedTimeStep=1.0 / 120.0,
			numSubSteps=4,
			numSolverIterations=50,
			enableConeFriction=1,
			contactBreakingThreshold=0.01,
			erp=0.2,
			contactERP=0.2,
			frictionERP=0.2,
			maxNumCmdPer1ms=1000,
		)

		p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

		# Control parameters
		self.max_torque = 4.0
		self.max_steering_force = 10.0
		self.max_wheel_velocity = 1000.0
		self.steering_return_rate = 0.01
		self.max_steering_angle = 0.9

		self.spring_stiffness = 500.0
		self.spring_damping = 50.0
		self.spring_rest_length = 0.0

		self.suspend = True

		# State variables
		self.steering_angle = 0.00
		self.throttle = 0.0

		# Joint indices (will be populated after loading URDF)
		self.joints = {}
		self.links = {}

		# Load world and car
		self._setup_world()
		self._load_car()
		self._setup_physics()
		self._setup_camera()

		pygame.init()
		pygame.joystick.init()

		if pygame.joystick.get_count() == 0:
			self.use_controller = False
		else:
			self.controller = pygame.joystick.Joystick(0)
			self.controller.init()
			self.use_controller = True

		self.com_marker_id = None

	def get_car_dimensions(self):
		fl = self.links["front_left_wheel"]
		fr = self.links["front_right_wheel"]
		rl = self.links["rear_left_wheel"]

		return (0.25, 0.25)

	def _ackermann_angles(self, angle):
		if abs(angle) < 1e-6:
			return 0.0, 0.0

		wheelbase, track_width = self.get_car_dimensions()

		turning_radius = wheelbase / np.tan(abs(angle))

		if angle > 0:  # Left turn
			# Left wheel is inner, right wheel is outer
			left_angle = np.atan(wheelbase / (turning_radius - track_width / 2))
			right_angle = np.atan(
				wheelbase / (turning_radius + track_width / 2)
			)
			return left_angle, right_angle
		else:  # Right turn
			# Right wheel is inner, left wheel is outer
			left_angle = -np.atan(
				wheelbase / (turning_radius + track_width / 2)
			)
			right_angle = -np.atan(
				wheelbase / (turning_radius - track_width / 2)
			)
			return left_angle, right_angle

	def _setup_world(self):
		"""Load the ground plane and obstacles"""
		self.plane_id = p.loadURDF("plane.urdf")

	# hmap = generate_terrain(256, 256, scale=100)
	#
	# ground_shape = p.createCollisionShape(
	# shapeType=p.GEOM_HEIGHTFIELD,
	# meshScale=[1, 1, 1],
	# heightfieldData=hmap.flatten(),
	# heightfieldTextureScaling=128,
	# numHeightfieldRows=256,
	# numHeightfieldColumns=256
	# )
	#
	# self.plane_id = p.createMultiBody(
	# baseMass=0,
	# baseCollisionShapeIndex=ground_shape,
	# basePosition=[0, 0, 0]
	# )

	def _load_car(self):
		"""Load the RC car URDF and get joint information"""
		start_pos = [0, 0, 2.0]  # Start slightly above ground
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
		print(f"Loaded car with {num_joints} joints:")

		for i in range(num_joints):
			joint_info = p.getJointInfo(self.car_id, i)
			joint_name = joint_info[1].decode("utf-8")
			link_name = joint_info[12].decode("utf-8")
			joint_type = joint_info[2]
			self.joints[joint_name] = i
			self.links[link_name] = i
			print(f"  Joint {i}: {joint_name} (type: {joint_type})")

	def _setup_physics(self):
		"""Configure physics properties and disable default motors"""
		# Disable default motors on wheel joints (not suspension joints)
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

		wheel_links = [
			"front_left_wheel_link",
			"front_right_wheel_link",
			"rear_left_wheel_link",
			"rear_right_wheel_link",
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
				print(f"Disabled motor on {joint_name}")
				p.changeDynamics(
					self.car_id,
					joint_id,
					jointLimitForce=1000,
					maxJointVelocity=2000,
				)
			else:
				print(f"Warning: {joint_name} not found!")

		for wheel_name in front_wheels:
			lid = self.links[wheel_name]
			p.changeDynamics(
				self.car_id,
				lid,
				lateralFriction=0.9,
				rollingFriction=0.01,
				spinningFriction=0.001,
			)
			print(f"Friction for {wheel_name} configured")

		for wheel_name in rear_wheels:
			lid = self.links[wheel_name]
			p.changeDynamics(
				self.car_id,
				lid,
				lateralFriction=0.4,
				rollingFriction=0.01,
				spinningFriction=0.001,
			)
			print(f"Friction for {wheel_name} configured")

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
		joint_id = self.joints.get(joint_name)
		if joint_id is None:
			print(f"Warning: Joint '{joint_name}' not found!")
		return joint_id

	def _apply_velocity_limiting(self, joint_name, requested_torque):
		"""Apply velocity limiting to prevent wheels from spinning too fast"""
		joint_id = self._get_joint_id(joint_name)
		if joint_id is None:
			return 0

		current_velocity = p.getJointState(self.car_id, joint_id)[1]

		if abs(current_velocity) > self.max_wheel_velocity:
			return 0  # Stop applying torque at max speed
		elif abs(current_velocity) > self.max_wheel_velocity * 0.8:
			# Gradually reduce torque near max speed
			reduction_factor = 1.0 - (
				(abs(current_velocity) - self.max_wheel_velocity * 0.8)
				/ (self.max_wheel_velocity * 0.2)
			)
			return requested_torque * reduction_factor
		else:
			return requested_torque

	def control_car(self, steering_input, throttle_input, ebrake):
		"""Apply steering and throttle controls to the car"""
		l_angle, r_angle = self._ackermann_angles(steering_input)

		fl = self._get_joint_id("front_left_steering_joint")
		fr = self._get_joint_id("front_right_steering_joint")

		for j, a in zip([fl, fr], [l_angle, r_angle]):
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
			if ebrake:
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
					joint_name, throttle_input
				)
				p.setJointMotorControl2(
					self.car_id,
					joint_id,
					p.TORQUE_CONTROL,
					force=limited_torque,
				)

	def update_controls(self):
		pygame.event.pump()

		steering_input = 0
		throttle_input = 0

		steering_input = -self.controller.get_axis(0)
		throttle_input = self.controller.get_axis(5) + 1
		throttle_input -= self.controller.get_axis(2) + 1
		throttle_input /= 2

		if self.controller.get_button(6):
			return False

		self.steering_angle = np.clip(
			steering_input / 2,
			-self.max_steering_angle,
			self.max_steering_angle,
		)
		self.throttle = np.clip(
			throttle_input * self.max_torque, -self.max_torque, self.max_torque
		)

		if self.controller.get_button(4):
			self.ebrake = True
		else:
			self.ebrake = False

		return True

	def _update_controls_kb(self, keys):
		"""Process keyboard input and update control state"""
		steering_input = 0
		throttle_input = 0

		# Process active keys
		for key, state in keys.items():
			if state & (p.KEY_WAS_TRIGGERED | p.KEY_IS_DOWN):
				if key == p.B3G_LEFT_ARROW:
					steering_input = 0.1
				elif key == p.B3G_RIGHT_ARROW:
					steering_input = -0.1
				elif key == p.B3G_UP_ARROW:
					throttle_input = 0.1
				elif key == p.B3G_DOWN_ARROW:
					throttle_input = -0.1
				elif key == 27:  # ESC
					return False  # Signal to exit

		# Update steering with return-to-center behavior
		if steering_input != 0:
			# Active steering input
			self.steering_angle = np.clip(
				self.steering_angle + steering_input, -0.5, 0.5
			)
		else:
			# Return to center when no input
			if abs(self.steering_angle) > 0.01:
				if self.steering_angle > 0:
					self.steering_angle = max(
						0, self.steering_angle - self.steering_return_rate
					)
				else:
					self.steering_angle = min(
						0, self.steering_angle + self.steering_return_rate
					)
			else:
				self.steering_angle = 0

		# Update throttle with decay when no input
		if throttle_input != 0:
			self.throttle = np.clip(
				self.throttle + throttle_input,
				-self.max_torque,
				self.max_torque,
			)
		else:
			# Gradual throttle decay
			if self.throttle > 0:
				self.throttle = max(0, self.throttle - 0.05)
			elif self.throttle < 0:
				self.throttle = min(0, self.throttle + 0.05)

		return True  # Continue simulation

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
			# Get current joint state
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

	def update_camera(self):
		"""Update camera to follow the car"""
		car_pos, _ = p.getBasePositionAndOrientation(self.car_id)
		p.resetDebugVisualizerCamera(
			cameraDistance=2.0,
			cameraYaw=30,
			cameraPitch=-30,
			cameraTargetPosition=car_pos,
		)

	def run(self):
		"""Main simulation loop"""
		print("RC Car Simulation Started")
		print("Controls: Arrow keys for steering and throttle")
		print("Press ESC to exit")

		try:
			while True:
				# Get keyboard input
				keys = p.getKeyboardEvents()

				# Update controls based on input
				if not self.update_controls():
					break  # Exit requested

				if self.suspend:
					self._apply_suspension_forces()

				# Apply controls to car
				self.control_car(
					self.steering_angle, self.throttle, self.ebrake
				)

				# Step physics simulation
				p.stepSimulation()

				# Update camera to follow car
				self.update_camera()

				# Control simulation speed (240 Hz)
				time.sleep(1.0 / 120.0)

		except KeyboardInterrupt:
			print("\nSimulation interrupted by user")
		finally:
			print("Shutting down simulation...")
			p.disconnect()


def main():
	"""Entry point for the RC car simulation"""
	sim = RCCarSimulation()
	sim.run()


if __name__ == "__main__":
	main()
