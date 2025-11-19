import torch
from collections import deque
from torch.optim import Adam
import numpy as np
from exploration_policy import ExplorationPolicy
from env_model import EnvModel, loss_fn
from memory import MPPIMemory, Transition, Action, State
from batched_memory import (
	BatchedAction,
	BatchedState,
	BatchedStateDelta,
)
import mediapy as media


class MPPI:
	def __init__(self, env, lambda_=1.0):
		self._noise_sigma = 0.5
		self._action_space = env.action_space
		self._observation_space = env.observation_space
		self._lambda = lambda_  # Temperature parameter for MPPI

		self._env_model = EnvModel(
			self._action_space,
			self._observation_space,
			hidden_size=2048,
			hidden_layers=4,
		)
		self._env_model.load_state_dict(
			torch.load("model.pt", weights_only=True, map_location=torch.device('cpu'))
		)
		self._env_model.eval()

		# ---- Learning stuff ---#

		self._opt = Adam(self._env_model.parameters(), lr=0.0001)
		self._loss_fn = loss_fn

		# --- MPPI stuff --- #

		self._horizon = 2
		self._n_samples = 500
		self._action_dim = self._action_space.shape[0]

		# Initialize nominal action sequence
		self._U = np.zeros((self._horizon, self._action_dim))

	def _fit_batch(self, actions, observations, targets):
		pred_X = BatchedStateDelta.from_tensor(
			self._env_model(observations['state'], actions)
		)
		loss, per_output_loss = self._loss_fn(pred_X, targets)

		self._opt.zero_grad()
		loss.backward()
		self._opt.step()

		return loss, per_output_loss

	def _sample_noise(self):
		return np.random.normal(
			0,
			self._noise_sigma,
			size=(self._n_samples, self._horizon, self._action_dim),
		)

	def _compute_cost(self, states: BatchedState, actions: BatchedAction):
		"""Compute cost for a trajectory."""
		position = states.position
		# local_goal_pos = states.local_goal_position
		# is_flippeds = torch.sigmoid(states.is_flipped)
		absolute_goal_position = states.absolute_goal_position

		distances = torch.norm(
			(absolute_goal_position - position)._tensor, dim=-1
		)

		path_cost = distances.mean(dim=-1)
		velocity_cost = (
			torch.norm(states.velocity._tensor, dim=-1).mean(dim=-1)
		) / 50.0

		action_cost = 0.01 * (actions._tensor**2).sum(dim=-1).sum(dim=-1)

		return (
			path_cost + action_cost + velocity_cost
		)  # terminal_cost + path_cost  # + action_cost #+ velocity_cost

	def _rollout_batch(self, initial_state, action_sequences, horizon=None):
		"""Rollout a batch of action sequences"""
		if horizon is None:
			horizon = self._horizon

		batch_size = action_sequences.shape[0]
		states = BatchedState.from_tensor(
			initial_state.unsqueeze(0).repeat(batch_size, 1)
		)

		trajectory_states = []

		for t in range(horizon):
			# Concatenate state and action for batch forward pass
			batch_input = torch.cat(
				[action_sequences[:, t], states._tensor], dim=-1
			)
			state_deltas = BatchedStateDelta.from_tensor(
				self._env_model(batch_input)
			)
			states = states + state_deltas
			trajectory_states.append(states._tensor)

		return torch.stack(trajectory_states, dim=1)

	def get_action(self, observation: State):
		"""Get action using MPPI

		Args:
			observation: Current state/observation
			cost_fn: Optional custom cost function(states, actions) -> costs
		"""
		# Sample noise
		noise = self._sample_noise()

		# Generate perturbed action sequences
		action_sequences = self._U + noise  # (n_samples, horizon, action_dim)

		# Clip actions to valid range
		action_sequences = np.clip(
			action_sequences, self._action_space.low, self._action_space.high
		)

		# Convert to torch tensors
		obs_tensor = torch.tensor(observation['state'], dtype=torch.float32)
		actions_tensor = torch.tensor(action_sequences, dtype=torch.float32)

		# Rollout trajectories
		with torch.no_grad():
			states = self._rollout_batch(obs_tensor, actions_tensor)

		# Compute costs
		costs = self._compute_cost(
			BatchedState.from_tensor(states),
			BatchedAction.from_tensor(actions_tensor),
		)

		# Compute weights using softmax with temperature
		costs_np = costs.detach().numpy()
		beta = -1.0 / self._lambda
		weights = np.exp(beta * (costs_np - costs_np.min()))
		weights = weights / weights.sum()

		# Compute weighted average of action sequences
		weighted_actions = np.sum(
			weights[:, np.newaxis, np.newaxis] * noise, axis=0
		)

		# Update nominal trajectory
		self._U = self._U + weighted_actions

		# Shift trajectory and add zero at end
		action = self._U[0].copy()
		self._U = np.roll(self._U, -1, axis=0)
		self._U[-1] = 0

		# Clip final action
		action = np.clip(
			action, self._action_space.low, self._action_space.high
		)

		return action, costs

	def reset_trajectory(self):
		"""Reset the nominal action sequence"""
		self._U = np.zeros((self._horizon, self._action_dim))


def distance_delta(obs_1, obs_2):
	return np.linalg.norm(obs_2[:3] - obs_1[:3])


def velocity(obs_1, obs_2):
	v1 = np.linalg.norm(obs_1[3:6])
	v2 = np.linalg.norm(obs_2[3:6])

	return (v1 + v2) / 2


if __name__ == "__main__":
	from drifter_env import DrifterEnv
	from torch.utils.tensorboard import SummaryWriter

	env = DrifterEnv(generate_terrain=False)
	mppi_controller = MPPI(env, lambda_=0.01)

	test_a = torch.tensor(env.action_space.sample())
	test_obs = torch.tensor(env.observation_space.sample())
	expl_policy = ExplorationPolicy(
		env.action_space,
		maneuver_duration=15,
		aggressive_prob=0.35,
		velocity_threshold=0.3,
	)

	memory = MPPIMemory(100000)

	writer = SummaryWriter()

	# Phase 1: Collect data and train model
	print("Phase 1: Training environment model...")
	s, _ = env.reset()
	done = False
	new_episode = False
	i = 0

	# Phase 2: Use MPPI for control
	s, _ = env.reset()
	done = False
	mppi_controller.reset_trajectory()

	recent_rollout = deque([], 10)

	frames = []
	for j in range(1_000):
		a, costs = mppi_controller.get_action(s)
		camera_out = env._sim.capture_front_camera()
		frames.append(camera_out)

		sp, r, done, trunc, _ = env.step(a)
		transition = Transition(
			action=Action.from_tensor(torch.tensor(a)),
			state=State.from_tensor(torch.tensor(s['state'])),
			next_state=State.from_tensor(torch.tensor(sp['state'])),
		)

		recent_rollout.append((s, a, sp))
		if j > 10:
			pass

		memory.add(transition)

		writer.add_scalar("Minimum estimated cost", costs.min(), i + j)

		s = sp

		if done:
			s, _ = env.reset()
			mppi_controller.reset_trajectory()

	media.write_video('mppi_sim.mp4', frames, fps=10)
	print("Done!")
