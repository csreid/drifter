import sqlite3
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict
import random
import matplotlib.pyplot as plt


class DrifterSequenceDataset(Dataset):
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
		self.db_path = db_path
		self.min_seq_len = min_seq_len
		self.max_seq_len = max_seq_len
		self.transform = transform

		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)

		# Connect to database and build episode index
		self.conn = sqlite3.connect(db_path, check_same_thread=False)
		self._build_episode_index()

	def _build_episode_index(self):
		"""
		Build an index of all episodes and their transition ranges.
		This allows us to sample sequences without crossing episode boundaries.
		"""
		cursor = self.conn.cursor()

		# Get all episodes with their row ranges
		cursor.execute("""
            SELECT 
                episode,
                MIN(id) as start_id,
                MAX(id) as end_id,
                COUNT(*) as length
            FROM transitions
            GROUP BY episode
            ORDER BY MIN(id)
            limit 1
        """)

		self.episodes = []
		for row in cursor.fetchall():
			episode_id, start_id, end_id, length = row
			if length >= self.min_seq_len:  # Only include episodes long enough
				self.episodes.append(
					{
						"episode_id": episode_id,
						"start_id": start_id,
						"end_id": end_id,
						"length": length,
					}
				)

		if not self.episodes:
			raise ValueError(
				f"No episodes found with length >= {self.min_seq_len}. "
				f"Check your database or reduce min_seq_len."
			)

		# Calculate total number of valid sequences we can sample
		# Each episode can produce multiple overlapping sequences
		self.num_sequences = sum(
			max(1, ep["length"] - self.min_seq_len + 1) for ep in self.episodes
		)

	def __len__(self) -> int:
		return self.num_sequences

	def _sample_sequence_from_episode(
		self, episode_info: Dict
	) -> Tuple[int, int]:
		"""
		Sample a random sequence start and length from an episode.

		Args:
		    episode_info: Dictionary with episode metadata

		Returns:
		    start_id: Starting row ID
		    seq_len: Length of the sequence
		"""
		episode_length = episode_info["length"]

		# Determine sequence length (random between min and max, but not longer than episode)
		max_possible_len = min(self.max_seq_len, episode_length)
		seq_len = random.randint(self.min_seq_len, max_possible_len)

		# Sample starting position (ensure sequence fits within episode)
		max_start_offset = episode_length - seq_len
		start_offset = random.randint(0, max_start_offset)

		start_id = episode_info["start_id"] + start_offset

		return start_id, seq_len

	def __getitem__(
		self, idx: int
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
		"""
		Get a sequence sample from the dataset.

		Args:
		    idx: Index of the sample (used to seed episode selection)

		Returns:
		    images: Sequence of camera images [seq_len, C, H, W]
		    state_dict: Dictionary of state sequences [seq_len, feature_dim]
		    seq_len: Actual length of the sequence
		"""
		# Select a random episode (use idx for some determinism)
		episode_idx = idx % len(self.episodes)
		episode_info = self.episodes[episode_idx]

		# Sample a sequence from this episode
		start_id, seq_len = self._sample_sequence_from_episode(episode_info)

		# Fetch the sequence from database
		cursor = self.conn.cursor()
		cursor.execute(
			"""
            SELECT 
                position_x, position_y, position_z,
                orientation_0, orientation_1, orientation_2, orientation_3,
                velocity_x, velocity_y, velocity_z,
                local_goal_x, local_goal_y, local_goal_z,
                goal_x, goal_y, goal_z,
                camera_image, camera_shape_0, camera_shape_1, camera_shape_2, camera_dtype
            FROM transitions
            WHERE id >= ? AND id < ?
            ORDER BY id
        """,
			(start_id, start_id + seq_len),
		)

		rows = cursor.fetchall()

		# Parse sequences
		images_list = []
		positions = []
		orientations = []
		velocities = []
		local_goals = []
		goals = []

		for row in rows:
			# Parse state components
			position = np.array([row[0], row[1], row[2]], dtype=np.float32)
			orientation = np.array(
				[row[3], row[4], row[5], row[6]], dtype=np.float32
			)
			velocity = np.array([row[7], row[8], row[9]], dtype=np.float32)
			local_goal = np.array([row[10], row[11], row[12]], dtype=np.float32)
			goal = np.array([row[13], row[14], row[15]], dtype=np.float32)

			positions.append(position)
			orientations.append(orientation)
			velocities.append(velocity)
			local_goals.append(local_goal)
			goals.append(goal)

			# Decompress and reshape camera image
			compressed_img = row[16]
			shape = (row[17], row[18], row[19])
			dtype = row[20]

			decompressed = gzip.decompress(compressed_img)
			image = np.frombuffer(decompressed, dtype=dtype).reshape(shape).copy()

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

	def __del__(self):
		"""Close database connection when dataset is deleted."""
		if hasattr(self, "conn"):
			self.conn.close()


def collate_fn_sequences(batch):
	"""
	Custom collate function for variable-length sequences.
	Pads sequences to the same length within a batch.

	Args:
	    batch: List of (images, state_dict, seq_len) tuples

	Returns:
	    images: Padded images tensor [batch_size, max_seq_len, C, H, W]
	    states: Dictionary of padded state tensors [batch_size, max_seq_len, feature_dim]
	    seq_lengths: Tensor of actual sequence lengths [batch_size]
	"""
	# Separate batch components
	images_list = [item[0] for item in batch]
	states_list = [item[1] for item in batch]
	seq_lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)

	# Pad image sequences
	# pad_sequence expects [seq_len, batch, ...] so we need to transpose
	images_padded = pad_sequence(
		images_list, batch_first=True, padding_value=0.0
	)
	# Result: [batch_size, max_seq_len, C, H, W]

	# Pad each state component
	states_padded = {}
	for key in states_list[0].keys():
		state_sequences = [states[key] for states in states_list]
		padded = pad_sequence(
			state_sequences, batch_first=True, padding_value=0.0
		)
		states_padded[key] = padded

	return images_padded, states_padded, seq_lengths


def create_sequence_dataloader(
	db_path: str,
	min_seq_len: int = 10,
	max_seq_len: int = 50,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 4,
	transform=None,
	seed: int = None,
) -> DataLoader:
	"""
	Create a DataLoader for sequential drifter data.

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

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate_fn_sequences,
		pin_memory=True,
	)

	return dataloader


# Example usage
if __name__ == "__main__":
	# Create sequence dataloader
	db_path = "drifter_data.db"
	dataloader = create_sequence_dataloader(
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
