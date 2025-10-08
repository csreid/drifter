import torch
import numpy as np
from dataclasses import dataclass
from typing import Union
import random


@dataclass
class Transition:
	"""Stores a single transition (action, observation) pair."""

	action: Union[torch.Tensor, np.ndarray]
	observation: Union[torch.Tensor, np.ndarray]
	next_observation: Union[torch.Tensor, np.ndarray, None] = None


class MPPIMemory:
	"""
	Circular buffer for storing transitions with random sampling.

	When the buffer reaches max_len, new transitions overwrite the oldest ones.
	Useful for training environment models in MPPI control.
	"""

	def __init__(self, max_len: int):
		"""
		Initialize the memory buffer.

		Args:
		    max_len: Maximum number of transitions to store
		"""
		self.transitions: list[Transition] = []
		self.max_len = max_len
		self._idx = 0  # Current write position for circular buffer

	def obs_diff(self, obs_t0, obs_t1):
		"""
		Compute the difference between two observations.
		"""
		# Split into non-quaternion and quaternion parts
		non_quat_t0 = obs_t0[..., :7]
		non_quat_t1 = obs_t1[..., :7]

		q_t0 = obs_t0[..., 7:]
		q_t1 = obs_t1[..., 7:]

		# Non-quaternion delta: simple subtraction
		delta_non_quat = non_quat_t1 - non_quat_t0

		# Quaternion delta: q_delta = q_t1 * q_t0^-1
		# For unit quaternions, inverse is conjugate: [-x, -y, -z, w]
		q_t0_conj = torch.cat([-q_t0[..., :3], q_t0[..., 3:4]], dim=-1)

		# Quaternion multiplication: q_t1 * q_t0_conj
		# [x, y, z, w] format
		x0, y0, z0, w0 = (
			q_t0_conj[..., 0],
			q_t0_conj[..., 1],
			q_t0_conj[..., 2],
			q_t0_conj[..., 3],
		)
		x1, y1, z1, w1 = q_t1[..., 0], q_t1[..., 1], q_t1[..., 2], q_t1[..., 3]

		w_delta = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
		x_delta = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
		y_delta = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
		z_delta = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0

		q_delta = torch.stack([x_delta, y_delta, z_delta, w_delta], dim=-1)

		# Normalize quaternion to handle numerical errors
		q_delta = q_delta / (torch.norm(q_delta, dim=-1, keepdim=True) + 1e-8)

		# Concatenate results
		return torch.cat([delta_non_quat, q_delta], dim=-1)

	def add(self, t: Transition) -> None:
		"""
		Add a transition to the buffer.

		If the buffer is full, overwrites the oldest transition.
		Also updates the previous transition's next_observation field.

		Args:
		    t: Transition to add
		"""
		# Update previous transition's next_observation
		if len(self.transitions) > 0:
			prev_idx = (
				(self._idx - 1) % len(self.transitions)
				if len(self.transitions) == self.max_len
				else len(self.transitions) - 1
			)
			self.transitions[prev_idx].next_observation = t.observation

		if len(self.transitions) < self.max_len:
			self.transitions.append(t)
		else:
			# Overwrite oldest transition (circular buffer)
			self.transitions[self._idx] = t
			self._idx = (self._idx + 1) % self.max_len

	def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Randomly sample n transitions from the buffer for environment model training.

		Args:
		    n: Number of transitions to sample

		Returns:
		    Tuple of (inputs, targets) as stacked torch tensors:
		    - inputs: concatenated [observations, actions] of shape (n, obs_dim + action_dim)
		    - targets: next_observations of shape (n, obs_dim)

		    Only samples transitions that have a valid next_observation.
		    Returns empty tensors if no valid transitions available.
		"""
		# Filter to only transitions with next_observation
		valid_transitions = [
			t for t in self.transitions if t.next_observation is not None
		]

		if len(valid_transitions) == 0:
			return torch.empty(0), torch.empty(0)

		# Sample with replacement if requesting more than available
		sampled = random.choices(valid_transitions, k=n)

		# Stack into tensors
		observations = []
		actions = []
		next_observations = []

		for t in sampled:
			# Convert to torch if needed
			obs = (
				t.observation
				if isinstance(t.observation, torch.Tensor)
				else torch.from_numpy(t.observation)
			)
			action = (
				t.action
				if isinstance(t.action, torch.Tensor)
				else torch.from_numpy(t.action)
			)
			next_obs = (
				t.next_observation
				if isinstance(t.next_observation, torch.Tensor)
				else torch.from_numpy(t.next_observation)
			)

			observations.append(obs)
			actions.append(action)
			next_observations.append(next_obs)

		# Concatenate observations and actions as model input
		obs_tensor = torch.stack(observations)
		action_tensor = torch.stack(actions)
		inputs = torch.cat([obs_tensor, action_tensor], dim=-1)
		nexts = torch.stack(next_observations)

		targets = self.obs_diff(obs_tensor, nexts)

		return inputs.float(), targets.float()

	def __len__(self) -> int:
		"""Return the current number of transitions in the buffer."""
		return len(self.transitions)

	def clear(self) -> None:
		"""Clear all transitions from the buffer."""
		self.transitions = []
		self._idx = 0
