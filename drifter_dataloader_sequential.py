import gzip
import numpy as np
import torch
from typing import Tuple, Dict, List
from sequential_dataset import (
	SequentialDatabaseDataset,
	create_sequential_dataloader,
)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
	"""
	Convert quaternion to rotation matrix.

	Args:
	    q: Quaternion [w, x, y, z] or [x, y, z, w] - check your convention!
	        Assuming [w, x, y, z] format

	Returns:
	    3x3 rotation matrix
	"""
	w, x, y, z = q[0], q[1], q[2], q[3]

	R = np.array(
		[
			[1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
			[2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
			[2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
		]
	)

	return R


def world_to_body_frame(
	orientation_quat: np.ndarray, world_vector: np.ndarray
) -> np.ndarray:
	"""
	Transform a world-frame vector to body-frame.

	Args:
	    orientation_quat: Current orientation as quaternion [4]
	    world_vector: Vector in world frame to transform [3]

	Returns:
	    Vector in body frame [3]
	"""
	R = quaternion_to_rotation_matrix(orientation_quat)
	# R transforms body to world, so R.T transforms world to body
	return R.T @ world_vector


class DrifterSequenceDataset(SequentialDatabaseDataset):
	"""
	PyTorch Dataset for loading sequences of drifter simulation data for LSTM/RNN training.

	Samples are variable-length sequences that never cross episode boundaries.

	Returns:
	    X: Sequence of camera images [seq_len, C, H, W]
	    Y: Dictionary containing sequences of:
	        - position: [seq_len, 3]
	        - orientation: [seq_len, 4] (quaternion)
	        - velocity: [seq_len, 3]
	        - local_goal: [seq_len, 3]
	        - goal: [seq_len, 3]
	    seq_len: Actual length of the sequence (for handling padding)
	"""

	def __init__(
		self,
		db_path: str,
		min_seq_len: int = 10,
		max_seq_len: int = 50,
		transform=None,
		seed: int = None,
	):
		"""
		Args:
		    db_path: Path to the SQLite database
		    min_seq_len: Minimum sequence length
		    max_seq_len: Maximum sequence length
		    transform: Optional transform to apply to images
		    seed: Random seed for reproducibility
		"""
		self.transform = transform

		super().__init__(
			db_path=db_path,
			table_name="transitions",
			episode_column="episode",
			id_column="id",
			min_seq_len=min_seq_len,
			max_seq_len=max_seq_len,
			seed=seed,
		)

	def get_columns(self) -> List[str]:
		"""Return the list of columns needed from the database."""
		return [
			"position_x",
			"position_y",
			"position_z",
			"orientation_0",
			"orientation_1",
			"orientation_2",
			"orientation_3",
			"velocity_x",
			"velocity_y",
			"velocity_z",
			"local_goal_x",
			"local_goal_y",
			"local_goal_z",
			"goal_x",
			"goal_y",
			"goal_z",
			"camera_image",
			"camera_shape_0",
			"camera_shape_1",
			"camera_shape_2",
			"camera_dtype",
		]

	def parse_rows(
		self, rows: List[Tuple]
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
		"""
		Parse database rows into camera images and state dictionary.

		Converts position, velocity to body-relative frame using the vehicle's orientation.

		Args:
		    rows: List of database row tuples

		Returns:
		    images: Sequence of camera images [seq_len, C, H, W]
		    state_dict: Dictionary of state sequences [seq_len, feature_dim]
		                - position: relative position in body frame
		                - orientation: kept as quaternion (represents body orientation)
		                - velocity: velocity in body frame
		                - local_goal: already in body frame
		                - goal: goal in body frame
		    seq_len: Actual length of the sequence
		"""
		seq_len = len(rows)

		images_list = []
		positions = []
		orientations = []
		velocities = []
		local_goals = []
		goals = []

		# Use first frame as reference for relative positioning
		first_frame = rows[0]
		initial_position = np.array(
			[first_frame[0], first_frame[1], first_frame[2]]
		)

		for row in rows:
			# Parse state components
			position = np.array([row[0], row[1], row[2]], dtype=np.float32)
			orientation = np.array(
				[row[3], row[4], row[5], row[6]], dtype=np.float32
			)
			velocity = np.array([row[7], row[8], row[9]], dtype=np.float32)
			local_goal = np.array([row[10], row[11], row[12]], dtype=np.float32)
			goal = np.array([row[13], row[14], row[15]], dtype=np.float32)

			# Convert position to body-relative frame
			# (relative to initial position, then rotated to body frame)
			relative_position = position - initial_position
			position_body = world_to_body_frame(orientation, relative_position)

			# Convert velocity to body-relative frame
			velocity_body = world_to_body_frame(orientation, velocity)

			# Convert goal to body-relative frame
			goal_body = world_to_body_frame(orientation, goal)

			positions.append(position_body)
			orientations.append(orientation)  # Keep as quaternion
			velocities.append(velocity_body)
			local_goals.append(local_goal)  # Already in body frame
			goals.append(goal_body)

			# Decompress and reshape camera image
			compressed_img = row[16]
			shape = (row[17], row[18], row[19])
			dtype = row[20]

			decompressed = gzip.decompress(compressed_img)
			image = (
				np.frombuffer(decompressed, dtype=dtype).reshape(shape).copy()
			)

			# Convert to torch tensor
			image = torch.from_numpy(image).float()
			if image.ndim == 3:  # If image has channels
				image = image.permute(2, 0, 1)  # HWC to CHW
			if image.max() > 1.0:  # If not already normalized
				image = image / 255.0

			# Apply optional transform
			if self.transform is not None:
				image = self.transform(image)

			images_list.append(image)

		# Stack into sequences
		images = torch.stack(images_list)  # [seq_len, C, H, W]

		state_dict = {
			"position": torch.from_numpy(np.stack(positions)),
			"orientation": torch.from_numpy(np.stack(orientations)),
			"velocity": torch.from_numpy(np.stack(velocities)),
			"local_goal": torch.from_numpy(np.stack(local_goals)),
			"goal": torch.from_numpy(np.stack(goals)),
		}

		return images, state_dict, seq_len


def create_drifter_dataloader(
	db_path: str,
	min_seq_len: int = 10,
	max_seq_len: int = 50,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 4,
	transform=None,
	seed: int = None,
):
	"""
	Create a DataLoader for drifter sequential data.

	Args:
	    db_path: Path to SQLite database
	    min_seq_len: Minimum sequence length
	    max_seq_len: Maximum sequence length
	    batch_size: Batch size for the dataloader
	    shuffle: Whether to shuffle the data
	    num_workers: Number of worker processes for data loading
	    transform: Optional transform to apply to images
	    seed: Random seed for reproducibility

	Returns:
	    DataLoader instance
	"""
	dataset = DrifterSequenceDataset(
		db_path,
		min_seq_len=min_seq_len,
		max_seq_len=max_seq_len,
		transform=transform,
		seed=seed,
	)

	return create_sequential_dataloader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
	)


# Example usage
if __name__ == "__main__":
	# Create sequence dataloader
	db_path = "drifter_data.db"
	dataloader = create_drifter_dataloader(
		db_path,
		min_seq_len=10,
		max_seq_len=30,
		batch_size=8,
		shuffle=True,
		num_workers=0,  # Set to 0 for debugging
	)

	# Test the dataloader
	print(f"Dataset size: {len(dataloader.dataset)}")
	print(f"Number of episodes: {len(dataloader.dataset.episodes)}")
	print(f"Number of batches: {len(dataloader)}")

	# Get a single batch
	for images, states, seq_lengths in dataloader:
		print("\nBatch shapes:")
		print(f"  Images: {images.shape}")  # [batch, max_seq_len, C, H, W]
		print(
			f"  Position: {states['position'].shape}"
		)  # [batch, max_seq_len, 3]
		print(f"  Orientation: {states['orientation'].shape}")
		print(f"  Velocity: {states['velocity'].shape}")
		print(f"  Local goal: {states['local_goal'].shape}")
		print(f"  Goal: {states['goal'].shape}")
		print(f"  Sequence lengths: {seq_lengths}")

		print("\nActual vs padded:")
		for i in range(min(3, len(seq_lengths))):
			print(
				f"  Sample {i}: actual_len={seq_lengths[i]}, padded_len={images.shape[1]}"
			)

		break  # Only show first batch
