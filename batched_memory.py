import torch
from dataclasses import dataclass
from typing import Self

from memory import Vector3, Quaternion, State, Action, Transition, StateDelta, Image

# Extend your existing classes with batch support


@dataclass
class BatchedVector3:
	"""Batched version that shares tensor storage with no copying."""

	_tensor: torch.Tensor  # Shape: (batch_size, 3)

	@classmethod
	def from_tensor(cls, tensor: torch.Tensor) -> "BatchedVector3":
		assert tensor.shape[-1] == 3, f"Expected last dim=3, got {tensor.shape}"
		return cls(_tensor=tensor.float())

	@property
	def x(self) -> torch.Tensor:
		return self._tensor[..., 0]

	@property
	def y(self) -> torch.Tensor:
		return self._tensor[..., 1]

	@property
	def z(self) -> torch.Tensor:
		return self._tensor[..., 2]

	def __getitem__(self, idx) -> "Vector3":
		return Vector3.from_tensor(self._tensor[idx])

	def __sub__(self, other: "BatchedVector3") -> "BatchedVector3":
		return type(self).from_tensor(self._tensor - other._tensor)

	def __add__(self, other: "BatchedVector3") -> "BatchedVector3":
		return type(self).from_tensor(self._tensor + other._tensor)

	def __len__(self) -> int:
		return self._tensor.shape[0]

@dataclass
class BatchedImage:
	_tensor: torch.Tensor

	def __getitem__(self, idx):
		return Image.from_tensor(self._tensor[idx])

	def __sub__(self, other):
		return other

	def __add__(self, other):
		return other

@dataclass
class BatchedQuaternion:
	"""Batched quaternion operations."""

	_tensor: torch.Tensor  # Shape: (batch_size, 4)

	@classmethod
	def from_tensor(cls, tensor: torch.Tensor) -> "BatchedQuaternion":
		assert tensor.shape[-1] == 4, f"Expected last dim=4, got {tensor.shape}"
		return cls(_tensor=tensor.float())

	@property
	def x(self) -> torch.Tensor:
		return self._tensor[..., 0]

	@property
	def y(self) -> torch.Tensor:
		return self._tensor[..., 1]

	@property
	def z(self) -> torch.Tensor:
		return self._tensor[..., 2]

	@property
	def w(self) -> torch.Tensor:
		return self._tensor[..., 3]

	def __getitem__(self, idx) -> "Quaternion":
		return Quaternion.from_tensor(self._tensor[idx])

	def __sub__(self, other: "BatchedQuaternion") -> "BatchedQuaternion":
		# Conjugate of self
		conj = torch.cat(
			[-self._tensor[..., :3], self._tensor[..., 3:4]], dim=-1
		)

		x0, y0, z0, w0 = conj[..., 0], conj[..., 1], conj[..., 2], conj[..., 3]
		x1, y1, z1, w1 = (
			other._tensor[..., 0],
			other._tensor[..., 1],
			other._tensor[..., 2],
			other._tensor[..., 3],
		)

		dw = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
		dx = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
		dy = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
		dz = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0

		dq = torch.stack([dx, dy, dz, dw], dim=-1)
		dq = dq / (torch.norm(dq, dim=-1, keepdim=True) + 1e-8)

		return type(self).from_tensor(dq)

	def __add__(self, other: "BatchedQuaternion") -> "BatchedQuaternion":
		x0, y0, z0, w0 = (
			self._tensor[..., 0],
			self._tensor[..., 1],
			self._tensor[..., 2],
			self._tensor[..., 3],
		)
		x1, y1, z1, w1 = (
			other._tensor[..., 0],
			other._tensor[..., 1],
			other._tensor[..., 2],
			other._tensor[..., 3],
		)

		w_new = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
		x_new = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
		y_new = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
		z_new = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0

		q_new = torch.stack([x_new, y_new, z_new, w_new], dim=-1)
		q_new = q_new / (torch.norm(q_new, dim=-1, keepdim=True) + 1e-8)

		return type(self).from_tensor(q_new)

	def __len__(self) -> int:
		return self._tensor.shape[0]


@dataclass
class BatchedState:
	"""Batched state with shared tensor storage."""

	_tensor: torch.Tensor  # Shape: (batch_size, 17)

	@classmethod
	def from_tensor(cls, tensor: torch.Tensor) -> "BatchedState":
		assert tensor.shape[-1] == 17, (
			f"Expected last dim=17, got {tensor.shape}"
		)
		return cls(_tensor=tensor.float())

	@classmethod
	def from_list(cls, states: list[State]) -> "BatchedState":
		"""Create batched state from list of State objects."""
		tensor = torch.stack([s._tensor for s in states])
		return cls.from_tensor(tensor)

	@property
	def position(self) -> BatchedVector3:
		return BatchedVector3.from_tensor(self._tensor[..., :3])

	@property
	def local_goal_position(self) -> BatchedVector3:
		return BatchedVector3.from_tensor(self._tensor[..., 3:6])

	@property
	def velocity(self) -> BatchedVector3:
		return BatchedVector3.from_tensor(self._tensor[..., 6:9])

	@property
	def is_flipped(self) -> torch.Tensor:
		return self._tensor[..., 9] == 1.0

	@property
	def absolute_goal_position(self) -> BatchedVector3:
		return BatchedVector3.from_tensor(self._tensor[..., 10:13])

	@property
	def orientation(self) -> BatchedQuaternion:
		return BatchedQuaternion.from_tensor(self._tensor[..., 13:17])

	@classmethod
	def from_values(
		cls,
		position: BatchedVector3,
		local_goal_position: BatchedVector3,
		absolute_goal_position: BatchedVector3,
		orientation: BatchedQuaternion,
		velocity: BatchedVector3,
		is_flipped: torch.Tensor,
	) -> "BatchedState":
		tensor = torch.cat(
			[
				position._tensor,
				local_goal_position._tensor,
				velocity._tensor,
				is_flipped.unsqueeze(-1)
				if is_flipped.dim() == 1
				else is_flipped,
				absolute_goal_position._tensor,
				orientation._tensor,
			],
			dim=-1,
		)
		return cls.from_tensor(tensor)

	def __getitem__(self, idx) -> "State":
		"""Get single state from batch - returns view."""

		return State.from_tensor(self._tensor[idx])

	def __sub__(self, other: Self) -> "BatchedStateDelta":
		return BatchedStateDelta.from_values(
			position=self.position - other.position,
			local_goal_position=self.local_goal_position
			- other.local_goal_position,
			absolute_goal_position=self.absolute_goal_position
			- other.absolute_goal_position,
			orientation=self.orientation - other.orientation,
			velocity=self.velocity - other.velocity,
			is_flipped=torch.logical_or(self.is_flipped, other.is_flipped),
		)

	def __add__(self, other: "BatchedStateDelta") -> "BatchedState":
		return BatchedState.from_values(
			position=self.position + other.position,
			local_goal_position=self.local_goal_position
			+ other.local_goal_position,
			absolute_goal_position=self.absolute_goal_position
			+ other.absolute_goal_position,
			orientation=self.orientation + other.orientation,
			velocity=self.velocity + other.velocity,
			is_flipped=self.is_flipped + other.is_flipped,
		)

	def __len__(self) -> int:
		return self._tensor.shape[0]


@dataclass
class BatchedStateDelta:
	"""Batched state delta."""

	_tensor: torch.Tensor

	@classmethod
	def from_tensor(cls, tensor: torch.Tensor) -> Self:
		assert tensor.shape[-1] == 17, (
			f"Expected last dim=17, got {tensor.shape}"
		)
		return cls(_tensor=tensor.float())

	@classmethod
	def from_list(cls, states: list[StateDelta]) -> Self:
		"""Create batched state delta from list of StateDelta objects."""
		tensor = torch.stack([s._tensor for s in states])
		return cls.from_tensor(tensor)

	@property
	def position(self) -> BatchedVector3:
		return BatchedVector3.from_tensor(self._tensor[..., :3])

	@property
	def local_goal_position(self) -> BatchedVector3:
		return BatchedVector3.from_tensor(self._tensor[..., 3:6])

	@property
	def velocity(self) -> BatchedVector3:
		return BatchedVector3.from_tensor(self._tensor[..., 6:9])

	@property
	def is_flipped(self) -> torch.Tensor:
		return self._tensor[..., 9] == 1.0

	@property
	def absolute_goal_position(self) -> BatchedVector3:
		return BatchedVector3.from_tensor(self._tensor[..., 10:13])

	@property
	def orientation(self) -> BatchedQuaternion:
		return BatchedQuaternion.from_tensor(self._tensor[..., 13:17])

	@classmethod
	def from_values(
		cls,
		position: BatchedVector3,
		local_goal_position: BatchedVector3,
		absolute_goal_position: BatchedVector3,
		orientation: BatchedQuaternion,
		velocity: BatchedVector3,
		is_flipped: torch.Tensor,
	) -> "BatchedStateDelta":
		tensor = torch.cat(
			[
				position._tensor,
				local_goal_position._tensor,
				velocity._tensor,
				is_flipped.unsqueeze(-1)
				if is_flipped.dim() == 1
				else is_flipped,
				absolute_goal_position._tensor,
				orientation._tensor,
			],
			dim=-1,
		)
		return cls.from_tensor(tensor)

	def __add__(self, other: BatchedState) -> BatchedState:
		return BatchedState.from_values(
			position=self.position + other.position,
			local_goal_position=self.local_goal_position
			+ other.local_goal_position,
			absolute_goal_position=self.absolute_goal_position
			+ other.absolute_goal_position,
			orientation=self.orientation + other.orientation,
			velocity=self.velocity + other.velocity,
			is_flipped=self.is_flipped + other.is_flipped,
		)


@dataclass
class BatchedAction:
	"""Batched actions with shared storage."""

	_tensor: torch.Tensor  # Shape: (batch_size, 2)

	@classmethod
	def from_tensor(cls, tensor: torch.Tensor) -> "BatchedAction":
		assert tensor.shape[-1] == 2, f"Expected last dim=2, got {tensor.shape}"
		return cls(_tensor=tensor.float())

	@classmethod
	def from_list(cls, actions: list[Action]) -> "BatchedAction":
		"""Create batched action from list of Action objects."""
		tensor = torch.stack([a._tensor for a in actions])
		return cls.from_tensor(tensor.float())

	@property
	def steering_input(self) -> torch.Tensor:
		return self._tensor[..., 0]

	@property
	def throttle_input(self) -> torch.Tensor:
		return self._tensor[..., 1]

	def __getitem__(self, idx) -> "Action":
		return Action.from_tensor(self._tensor[idx])

	def __len__(self) -> int:
		return self._tensor.shape[0]


@dataclass
class BatchedTransition:
	"""Batched transitions."""

	state: BatchedState
	action: BatchedAction
	next_state: BatchedState

	@classmethod
	def from_list(cls, transitions: list[Transition]) -> "BatchedTransition":
		"""Create batched transition from list of Transition objects."""
		states = BatchedState.from_list([t.state for t in transitions])
		actions = BatchedAction.from_list([t.action for t in transitions])

		next_states = BatchedState.from_list(
			[t.next_state for t in transitions]
		)

		return cls(state=states, action=actions, next_state=next_states)

	def __getitem__(self, idx) -> "Transition":
		return Transition(
			state=self.state[idx],
			action=self.action[idx],
			next_state=self.next_state[idx]
		)

	def __len__(self) -> int:
		return len(self.state)
