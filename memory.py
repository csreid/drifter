import torch
import numpy as np
from dataclasses import dataclass
from typing import Self, TYPE_CHECKING
import random

if TYPE_CHECKING:
	from batched_memory import (
		BatchedAction,
		BatchedState,
		BatchedStateDelta,
	)


@dataclass
class Vector3:
	_tensor: torch.Tensor

	def to(self, dev):
		self._tensor = self._tensor.to(dev)
		return self

	@property
	def x(self):
		return self._tensor[0]

	@property
	def y(self):
		return self._tensor[1]

	@property
	def z(self):
		return self._tensor[2]

	@classmethod
	def from_tensor(cls, tensor) -> Self:
		x, y, z = tensor

		return cls(_tensor=tensor)

	@classmethod
	def from_values(cls, x, y, z) -> Self:
		return cls.from_tensor(torch.tensor([x, y, z]))

	def __sub__(self, other) -> Self:
		return type(self).from_tensor(self._tensor - other._tensor)

	def __add__(self, other) -> Self:
		return type(self).from_tensor(self._tensor + other._tensor)


@dataclass
class Quaternion:
	_tensor: torch.Tensor

	def to(self, dev):
		self._tensor = self._tensor.to(dev)
		return self

	def __post_init__(self):
		assert len(self._tensor) == 4

	@property
	def x(self):
		return self._tensor[0]

	@property
	def y(self):
		return self._tensor[1]

	@property
	def z(self):
		return self._tensor[2]

	@property
	def w(self):
		return self._tensor[2]

	@classmethod
	def from_tensor(cls, tensor) -> Self:
		return cls(_tensor=tensor)

	@classmethod
	def from_values(cls, x, y, z, w) -> Self:
		return cls.from_tensor(torch.tensor([x, y, z, w]))

	def __sub__(self, other) -> Self:
		conj = torch.cat([-self._tensor[:3], self._tensor[3:]])

		x0, y0, z0, w0 = conj
		x1, y1, z1, w1 = other._tensor

		dw = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
		dx = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
		dy = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
		dz = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0

		dq = torch.stack([dx, dy, dz, dw], dim=-1)
		dq = dq / (torch.norm(dq, dim=-1, keepdim=True) + 1e-8)

		return type(self).from_tensor(dq)

	def __add__(self, other) -> Self:
		x0, y0, z0, w0 = self._tensor
		x1, y1, z1, w1 = other._tensor

		w_new = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
		x_new = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
		y_new = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
		z_new = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0

		q_new = torch.stack([x_new, y_new, z_new, w_new], dim=-1)
		q_new = q_new / (torch.norm(q_new, dim=-1, keepdim=True) + 1e-8)

		return type(self).from_tensor(q_new)


@dataclass
class Image:
	_tensor: torch.Tensor

	@classmethod
	def from_tensor(cls, tensor):
		return cls(_tensor=tensor)

	# ???? idk lol
	# Could maybe do optical flow or something?
	def __sub__(self, other):
		return other

	def __add__(self, other):
		return other


@dataclass
class State:
	position: Vector3
	local_goal_position: Vector3
	absolute_goal_position: Vector3
	orientation: Quaternion
	velocity: Vector3
	is_flipped: torch.Tensor
	_tensor: torch.Tensor

	def to(self, dev):
		self._tensor.to(dev)

	@classmethod
	def from_tensor(cls, tensor) -> Self:
		pos = Vector3.from_tensor(tensor[:3])
		local_goal_pos = Vector3.from_tensor(tensor[3:6])
		velocity = Vector3.from_tensor(tensor[6:9])
		is_flipped = tensor[10] == 1.0
		goal_pos = Vector3.from_tensor(tensor[10:13])
		orientation = Quaternion.from_tensor(tensor[13:17])

		return cls(
			position=pos,
			local_goal_position=local_goal_pos,
			absolute_goal_position=goal_pos,
			orientation=orientation,
			velocity=velocity,
			is_flipped=is_flipped,
			_tensor=tensor,
		)

	@classmethod
	def from_values(
		cls,
		position,
		local_goal_position,
		absolute_goal_position,
		orientation,
		velocity,
		is_flipped,
	) -> Self:
		tensor = torch.cat(
			[
				position._tensor,
				local_goal_position._tensor,
				velocity._tensor,
				torch.tensor([is_flipped]),
				absolute_goal_position._tensor,
				orientation._tensor,
			],
			dim=-1,
		)

		return cls.from_tensor(tensor)

	def __add__(self, other: "StateDelta") -> Self:
		if type(other) is not StateDelta:
			raise Exception(f"Cannot add {type(other)}) to {type(self)}")

		return type(self).from_values(
			position=self.position + other.position,
			local_goal_position=(
				self.local_goal_position + other.local_goal_position
			),
			absolute_goal_position=(
				self.absolute_goal_position + other.absolute_goal_position
			),
			orientation=self.orientation + other.orientation,
			velocity=self.velocity + other.velocity,
			is_flipped=self.is_flipped + other.is_flipped,
		)

	def __sub__(self, other: Self) -> "StateDelta":
		if type(other) is not State:
			raise Exception(f"Cannot subtract {type(other)}) from {type(self)}")

		return StateDelta.from_values(
			position=self.position - other.position,
			local_goal_position=(
				self.local_goal_position - other.local_goal_position
			),
			absolute_goal_position=(
				self.absolute_goal_position - other.absolute_goal_position
			),
			orientation=self.orientation - other.orientation,
			velocity=self.velocity - other.velocity,
			is_flipped=torch.logical_or(self.is_flipped, other.is_flipped),
		)


class StateDelta(State):
	def __add__(self, other: State):
		if type(other) is not State:
			raise Exception("Can only add `State` to `StateDelta`")

		# Delegate to `other`'s add method
		return other + self


@dataclass
class Action:
	steering_input: float
	throttle_input: float
	_tensor: torch.Tensor

	def to(self, dev):
		self._tensor = self._tensor.to(dev)
		return self

	@classmethod
	def from_tensor(cls, tensor) -> "Action":
		steer, throttle = tensor

		return cls(
			steering_input=steer, throttle_input=throttle, _tensor=tensor
		)


@dataclass
class Transition:
	state: State
	action: Action
	next_state: State

	def to(self, dev):
		self.state = self.state.to(dev)
		self.action = self.action.to(dev)
		self.next_state = self.next_state.to(dev)

		return self


class MPPIMemory:
	"""
	Circular buffer for storing transitions with random sampling.

	When the buffer reaches max_len, new transitions overwrite the oldest ones.
	Useful for training environment models in MPPI control.
	"""

	def __init__(self, max_len: int, dev="cpu"):
		"""
		Initialize the memory buffer.

		Args:
		    max_len: Maximum number of transitions to store
		"""
		self.transitions: list[Transition] = []
		self.max_len = max_len
		self._idx = 0  # Current write position for circular buffer

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

		if len(self.transitions) < self.max_len:
			self.transitions.append(t)
		else:
			# Overwrite oldest transition (circular buffer)
			self.transitions[self._idx] = t
			self._idx = (self._idx + 1) % self.max_len

	def sample(
		self, n: int
	) -> tuple["BatchedAction", "BatchedState", "BatchedStateDelta"]:
		from batched_memory import (
			BatchedAction,
			BatchedState,
		)

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
			t for t in self.transitions if t.next_state is not None
		]

		# Sample with replacement if requesting more than available
		sampled = random.choices(valid_transitions, k=n)

		# Stack into tensors
		observations = []
		actions = []
		next_observations = []

		for t in sampled:
			# Convert to torch if needed
			obs = t.state._tensor
			action = t.action._tensor
			if t.next_state is None:
				raise Exception("Impossible! I filtered those out!")
			next_obs = t.next_state._tensor

			observations.append(obs)
			actions.append(action)
			next_observations.append(next_obs)

		# Concatenate observations and actions as model input
		obs_tensor = torch.stack(observations)
		action_tensor = torch.stack(actions)
		inputs = torch.cat([obs_tensor, action_tensor], dim=-1)
		nexts = torch.stack(next_observations)

		# targets = obs_diff(obs_tensor, nexts)
		batched_action = BatchedAction.from_tensor(action_tensor)
		batched_state = BatchedState.from_tensor(obs_tensor)
		batched_next_state = BatchedState.from_tensor(nexts)

		return batched_action, batched_state, batched_next_state - batched_state

	def to_pandas(self):
		"""
		Convert the memory buffer to a pandas DataFrame.

		Returns:
				pd.DataFrame: DataFrame with columns for all observation dimensions,
										 actions, and next observations.
		"""
		import pandas as pd

		if len(self.transitions) == 0:
			return pd.DataFrame()

		data = []
		for t in self.transitions:
			row = {}

			# Current observation
			obs = (
				t.state._tensor
				if isinstance(t.state._tensor, np.ndarray)
				else t.state._tensor.numpy()
			)
			row["pos_x"] = obs[0]
			row["pos_y"] = obs[1]
			row["pos_z"] = obs[2]
			row["goal_pos_x"] = obs[3]
			row["goal_pos_y"] = obs[4]
			row["goal_pos_z"] = obs[5]
			row["vel_x"] = obs[6]
			row["vel_y"] = obs[7]
			row["vel_z"] = obs[8]
			row["is_flipped"] = obs[9]
			row["absolute_goal_pos_x"] = obs[10]
			row["absolute_goal_pos_y"] = obs[11]
			row["absolute_goal_pos_z"] = obs[12]
			row["orn_x"] = obs[13]
			row["orn_y"] = obs[14]
			row["orn_z"] = obs[15]
			row["orn_w"] = obs[16]

			# Action
			action = (
				t.action._tensor
				if isinstance(t.action._tensor, np.ndarray)
				else t.action._tensor.numpy()
			)
			for i, val in enumerate(action):
				row[f"action_{i}"] = val

			# Next observation (if available)
			if t.next_state is not None:
				next_obs = (
					t.next_state
					if isinstance(t.next_state, np.ndarray)
					else t.next_state._tensor.numpy()
				)
				row["next_pos_x"] = next_obs[0]
				row["next_pos_y"] = next_obs[1]
				row["next_pos_z"] = next_obs[2]
				row["next_goal_pos_x"] = next_obs[3]
				row["next_goal_pos_y"] = next_obs[4]
				row["next_goal_pos_z"] = next_obs[5]
				row["next_vel_x"] = next_obs[6]
				row["next_vel_y"] = next_obs[7]
				row["next_vel_z"] = next_obs[8]
				row["next_is_flipped"] = next_obs[9]
				row["next_absolute_goal_pos_x"] = obs[10]
				row["next_absolute_goal_pos_y"] = obs[11]
				row["next_absolute_goal_pos_z"] = obs[12]
				row["next_orn_x"] = next_obs[13]
				row["next_orn_y"] = next_obs[14]
				row["next_orn_z"] = next_obs[15]
				row["next_orn_w"] = next_obs[16]

			data.append(row)

		return pd.DataFrame(data)

	def __len__(self) -> int:
		"""Return the current number of transitions in the buffer."""
		return len(self.transitions)

	def clear(self) -> None:
		"""Clear all transitions from the buffer."""
		self.transitions = []
		self._idx = 0
