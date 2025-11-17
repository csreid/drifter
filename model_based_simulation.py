import torch
import numpy as np
import pybullet as p
import pybullet_data
from env_model import EnvModel
from drifter_env import observation_space, action_space
from memory import State, Action
from batched_memory import BatchedState, BatchedAction, BatchedStateDelta
import time
from collections import deque
import tempfile
import xacro


class ModelBasedSimulation:
	"""Interactive simulation using the learned environment model"""

	def _load_car(self):
		"""Load the RC car URDF and get joint information"""
		start_pos = [0, 0, 2.0]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])

		with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf") as f:
			urdf_str = xacro.process_file(
				"robot.xacro",
				mappings={"suspend": "true"},
			).toxml()
			f.write(urdf_str)
			f.flush()

			self.car_id = p.loadURDF(f.name, start_pos, start_orientation)

		# Get joint information
		num_joints = p.getNumJoints(self.car_id)
		p.changeDynamics(
			self.car_id, -1, linearDamping=0.005, angularDamping=0.005
		)

	def __init__(self, model_path="model.pt", gui=True):
		# Load the trained model
		self.model = EnvModel(
			action_space, observation_space, hidden_size=512, hidden_layers=2
		)
		self.model.load_state_dict(torch.load(model_path))
		self.model.eval()

		# Physics setup for visualization only
		self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.81)

		if gui:
			p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

		# Load world
		self.plane_id = p.loadURDF("plane.urdf")

		# Load car for visualization
		start_pos = [0, 0, 0.5]
		start_orientation = p.getQuaternionFromEuler([0, 0, 0])
		self._load_car()

		# Load goal marker
		self._spawn_goal([10, 10, 0])

		# Initialize state
		self.current_state = State.from_tensor(
			torch.tensor(
				[
					0.0,
					0.0,
					0.5,  # position
					10.0,
					10.0,
					0.0,  # local goal position
					0.0,
					0.0,
					0.0,  # velocity
					0.0,  # is_flipped
					10.0,
					10.0,
					0.0,  # absolute goal position
					0.0,
					0.0,
					0.0,
					1.0,  # orientation (quaternion)
				]
			).float()
		)

		# Control state
		self.steering = 0.0
		self.throttle = 0.0

		# Setup camera
		p.resetDebugVisualizerCamera(
			cameraDistance=5.0,
			cameraYaw=45,
			cameraPitch=-30,
			cameraTargetPosition=[0, 0, 0],
		)

		# Keyboard state tracking
		self.key_states = {
			ord("w"): False,
			ord("s"): False,
			ord("a"): False,
			ord("d"): False,
			ord("r"): False,  # Reset
		}

	def _spawn_goal(self, position):
		"""Create visual goal marker"""
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

		self.goal_id = p.createMultiBody(
			baseMass=0,
			baseCollisionShapeIndex=goal_shape,
			baseVisualShapeIndex=goal_visual,
			basePosition=[
				position[0],
				position[1],
				position[2] + goal_height * 0.5,
			],
		)

	def reset(self):
		"""Reset simulation to initial state"""
		# Random start position
		new_x = np.random.uniform(-10, 10)
		new_y = np.random.uniform(-10, 10)

		# Random goal position
		goal_x = np.random.uniform(-20, 20)
		goal_y = np.random.uniform(-20, 20)

		self.current_state = State.from_tensor(
			torch.tensor(
				[
					new_x,
					new_y,
					0.5,  # position
					goal_x - new_x,
					goal_y - new_y,
					-0.5,  # local goal position
					0.0,
					0.0,
					0.0,  # velocity
					0.0,  # is_flipped
					goal_x,
					goal_y,
					0.0,  # absolute goal position
					0.0,
					0.0,
					0.0,
					1.0,  # orientation (quaternion)
				]
			).float()
		)

		# Update visualization
		p.resetBasePositionAndOrientation(
			self.car_id, [new_x, new_y, 0.2], [0, 0, 0, 1]
		)
		p.resetBasePositionAndOrientation(
			self.goal_id, [goal_x, goal_y, 2.5], [0, 0, 0, 1]
		)

		self.steering = 0.0
		self.throttle = 0.0

	def update_controls(self):
		"""Update control inputs based on keyboard state"""
		keys = p.getKeyboardEvents()

		# Update key states
		for key, state in keys.items():
			if key in self.key_states:
				self.key_states[key] = (
					state == p.KEY_IS_DOWN or state == p.KEY_WAS_TRIGGERED
				)

		# Handle reset
		if self.key_states[ord("r")]:
			self.reset()
			self.key_states[ord("r")] = False
			return

		# Calculate steering (A/D keys)
		target_steering = 0.0
		if self.key_states[ord("a")]:
			target_steering = 1.0
		elif self.key_states[ord("d")]:
			target_steering = -1.0

		# Calculate throttle (W/S keys)
		target_throttle = 0.0
		if self.key_states[ord("w")]:
			target_throttle = 1.0
		elif self.key_states[ord("s")]:
			target_throttle = -1.0

		# Smooth control transitions
		self.steering = self.steering * 0.8 + target_steering * 0.2
		self.throttle = self.throttle * 0.8 + target_throttle * 0.2

	def step(self):
		"""Step the model-based simulation forward"""
		# Create action from current controls
		action = Action.from_tensor(
			torch.tensor([self.steering, self.throttle]).float()
		)

		# Get prediction from model
		with torch.no_grad():
			delta_tensor = self.model(self.current_state, action)
			delta = BatchedStateDelta.from_tensor(delta_tensor.unsqueeze(0))

		# Apply delta to current state (note: model predicts state - next_state)
		# So we need to subtract the delta
		new_state_tensor = self.current_state._tensor - delta_tensor
		self.current_state = State.from_tensor(new_state_tensor)

		# Update visualization
		pos = self.current_state.position._tensor.numpy()
		orn = self.current_state.orientation._tensor.numpy()

		p.resetBasePositionAndOrientation(
			self.car_id, pos.tolist(), orn.tolist()
		)

		# Update camera to follow car
		p.resetDebugVisualizerCamera(
			cameraDistance=5.0,
			cameraYaw=45,
			cameraPitch=-30,
			cameraTargetPosition=pos.tolist(),
		)

		# Display state info
		vel = self.current_state.velocity._tensor.numpy()
		speed = np.linalg.norm(vel)
		goal_pos = self.current_state.local_goal_position._tensor.numpy()
		goal_dist = np.linalg.norm(goal_pos)

		info_text = f"Speed: {speed:.2f} m/s | Goal Distance: {goal_dist:.2f} m | Steering: {self.steering:.2f} | Throttle: {self.throttle:.2f}"
		p.addUserDebugText(
			info_text,
			[pos[0], pos[1], pos[2] + 1],
			textColorRGB=[1, 1, 1],
			textSize=1.2,
			lifeTime=0.1,
		)

	def run(self):
		"""Main simulation loop"""
		print("Controls:")
		print("  W - Forward")
		print("  S - Backward")
		print("  A - Turn Left")
		print("  D - Turn Right")
		print("  R - Reset")
		print("\nRunning model-based simulation...")

		try:
			for _ in range(10000):
				p.stepSimulation()

			input()
			while True:
				self.update_controls()
				self.step()
				time.sleep(1.0 / 1.0)  # 60 FPS

		except KeyboardInterrupt:
			print("\nShutting down...")
		finally:
			p.disconnect()


if __name__ == "__main__":
	sim = ModelBasedSimulation(model_path="model.pt", gui=True)
	sim.run()
