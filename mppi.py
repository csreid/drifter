import torch
from tqdm import tqdm
from collections import deque
from torch.nn import Linear, Module, Sequential, LeakyReLU, MSELoss, Sigmoid
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np
from exploration_policy import ExplorationPolicy
from tabulate import tabulate
from IPython import embed

def loss_fn(inputs, targets):
	goal_pos_inp = inputs[..., :3]
	velocity_inp = inputs[..., 3:6]
	is_flipped_inp = inputs[..., 6:7]
	omega_inp = inputs[..., 7:10]
	orn_inp = inputs[..., 10:]

	goal_pos_target = targets[..., :3]
	velocity_target = targets[..., 3:6]
	is_flipped_target = targets[..., 6:7]
	omega_target = targets[..., 7:10]
	orn_target = targets[..., 10:]

	loss_components = {
		"position": F.mse_loss(goal_pos_inp, goal_pos_target),
		"velocity": F.mse_loss(velocity_inp, velocity_target),
		"is_flipped": F.binary_cross_entropy_with_logits(is_flipped_inp, is_flipped_target),
		"omega": F.mse_loss(omega_inp, omega_target),
		"orientation": F.mse_loss(orn_inp, orn_target),
	}

	return torch.sum(torch.stack([val for _, val in loss_components.items()])), loss_components

class EnvModel(Module):
	def __init__(
		self, action_space, observation_space, hidden_size=64, hidden_layers=4
	):
		super().__init__()

		# Assume the input spaces are 1-d, this will break otherwise
		input_len = action_space.shape[0] + observation_space.shape[0]
		self._input = Linear(input_len, hidden_size)

		hidden_layers = [
			[Linear(hidden_size, hidden_size), LeakyReLU()]
			for _ in range(hidden_layers)
		]
		self._hidden = Sequential(
			*[layer for tier in hidden_layers for layer in tier]
		)
		self._output = Linear(hidden_size, observation_space.shape[0])

	def forward(self, *args):
		"""If called with one argument, that argument should
		be an appropriately-shaped batch of action/observation
		tensors

		If called with two arguments, they should be a single action and state pair
		"""
		if len(args) == 1:
			return self._batch_forward(*args)
		elif len(args) == 2:
			return self._one_forward(*args)

	def _batch_forward(self, X):
		out = self._input(X)
		out = self._hidden(out)
		out = self._output(out)
		out

		return out

	def _one_forward(self, action, observation):
		inp = torch.cat([action, observation])

		return self._batch_forward(inp)


class MPPI:
	def __init__(self, env, lambda_=1.0, temperature=1.0):
		self._noise_sigma = 1.0
		self._action_space = env.action_space
		self._observation_space = env.observation_space
		self._lambda = lambda_  # Temperature parameter for MPPI
		self._temperature = temperature

		self._env_model = EnvModel(
			self._action_space,
			self._observation_space,
			hidden_size=256,
			hidden_layers=1,
		)

		# ---- Learning stuff ---#

		self._opt = Adam(self._env_model.parameters())
		self._loss_fn = loss_fn

		# --- MPPI stuff --- #

		self._horizon = 2
		self._n_samples = 1000
		self._action_dim = self._action_space.shape[0]

		# Initialize nominal action sequence
		self._U = np.zeros((self._horizon, self._action_dim))

	def add_delta(self, obs: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
		"""
		Apply a delta to an observation to get a new observation.

		Args:
				obs: Current observation, shape (..., 14)
				delta: Delta observation, shape (..., 14)

		Returns:
				New observation, shape (..., 14)
				- First 10 elements: standard addition (obs + delta)
				- Last 4 elements: quaternion multiplication (q_new = q_delta * q_obs)
				Quaternion format: [x, y, z, w] (PyBullet convention)
		"""
		# Split into non-quaternion and quaternion parts
		non_quat_obs = obs[..., :7]
		non_quat_delta = delta[..., :7]

		q_obs = obs[..., 7:]  # shape (..., 4) - [x, y, z, w]
		q_delta = delta[..., 7:]  # shape (..., 4)

		# Non-quaternion: simple addition
		new_non_quat = non_quat_obs + non_quat_delta

		# Quaternion: q_new = q_delta * q_obs
		# [x, y, z, w] format
		x0, y0, z0, w0 = (
			q_obs[..., 0],
			q_obs[..., 1],
			q_obs[..., 2],
			q_obs[..., 3],
		)
		x1, y1, z1, w1 = (
			q_delta[..., 0],
			q_delta[..., 1],
			q_delta[..., 2],
			q_delta[..., 3],
		)

		w_new = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
		x_new = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
		y_new = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
		z_new = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0

		q_new = torch.stack([x_new, y_new, z_new, w_new], dim=-1)

		# Normalize quaternion to handle numerical errors
		q_new = q_new / (torch.norm(q_new, dim=-1, keepdim=True) + 1e-8)

		# Concatenate results
		return torch.cat([new_non_quat, q_new], dim=-1)

	def _fit_batch(self, batch_X, batch_Y):
		pred_X = self._env_model(batch_X)
		loss, per_output_loss = self._loss_fn(pred_X, batch_Y)

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

	def _compute_cost(self, states, actions):
		"""Compute cost for a trajectory."""
		local_goal_pos = states[..., :3]
		is_flippeds = states[..., -1]

		distances = torch.norm(local_goal_pos, dim=-1)

		terminal_cost = distances[:, -1]

		path_cost = distances.mean(dim=-1)

		flipped_cost = 1000.0 * is_flippeds.sum(dim=-1)

		action_cost = 0.01 * (actions**2).sum(dim=-1).sum(dim=-1)

		return (
			# terminal_cost + path_cost + action_cost + flipped_cost
			#terminal_cost + action_cost + flipped_cost
			terminal_cost #+ path_cost #+ flipped_cost
		)

	def _rollout_batch(self, initial_state, action_sequences):
		"""Rollout a batch of action sequences"""
		batch_size = action_sequences.shape[0]
		states = initial_state.unsqueeze(0).repeat(batch_size, 1)

		trajectory_states = []

		for t in range(self._horizon):
			# Concatenate state and action for batch forward pass
			batch_input = torch.cat([action_sequences[:, t], states], dim=-1)
			state_deltas = self._env_model(batch_input)
			states = self.add_delta(states, state_deltas)
			trajectory_states.append(states)

		return torch.stack(trajectory_states, dim=1)

	def get_action(self, observation):
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
		obs_tensor = torch.tensor(observation, dtype=torch.float32)
		actions_tensor = torch.tensor(action_sequences, dtype=torch.float32)

		# Rollout trajectories
		states = self._rollout_batch(obs_tensor, actions_tensor)

		# Compute costs
		costs = self._compute_cost(states, actions_tensor)

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


if __name__ == "__main__":
	from drifter_env import DrifterEnv
	from memory import MPPIMemory, Transition
	from torch.utils.tensorboard import SummaryWriter

	env = DrifterEnv(gui=True)
	mppi_controller = MPPI(env)

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
	for i in tqdm(range(2_000)):
		a = expl_policy.get_action(s)
		transition = Transition(action=a, observation=s)
		if not done:
			memory.add(transition)
		s, r, done, trunc, _ = env.step(a)

		if i > 100 and (i % 100) == 0:
			batch_X, batch_Y = memory.sample(10_000)
			loss, loss_components = mppi_controller._fit_batch(batch_X, batch_Y)
			writer.add_scalar(f"Loss", loss, i)
			writer.add_scalars('loss_components', loss_components, i)

		if done:
			s, _ = env.reset()
			mppi_controller.reset_trajectory()

	# Phase 2: Use MPPI for control
	s, _ = env.reset()
	done = False
	mppi_controller.reset_trajectory()
	env.set_realtime(True)

	for j in range(100_000):
		a, costs = mppi_controller.get_action(s)
		transition = Transition(action=a, observation=s)
		if not done:
			memory.add(transition)

		s, r, done, trunc, _ = env.step(a)

		batch_X, batch_Y = memory.sample(128)
		loss, loss_components = mppi_controller._fit_batch(batch_X, batch_Y)
		writer.add_scalars('loss_components', loss_components, i+j)

		writer.add_scalar(f"Loss", loss, i + j)
		writer.add_scalar('Distance from goal', torch.norm(torch.tensor(s[:3])), i+j)
		writer.add_scalar('Minimum estimated cost', costs.min(), i+j)

		if done:
			s, _ = env.reset()
			mppi_controller.reset_trajectory()

	print("Done!")
