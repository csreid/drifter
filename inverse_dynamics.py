import sqlite3
import numpy as np
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import io


class InverseDynamicsDataset(Dataset):
	"""
	Dataset for inverse dynamics learning from RC car simulation data.

	Given a sequence of camera images, predict the sequence of actions taken.
	"""

	def __init__(
		self,
		db_path: str,
		sequence_length: int = 4,
		stride: int = 1,
		normalize_actions: bool = True,
		cache_images: bool = False,
	):
		"""
		Args:
		    db_path: Path to the SQLite database
		    sequence_length: Number of consecutive frames to include in each sequence
		    stride: Step size when creating sequences (stride=1 means overlapping sequences)
		    normalize_actions: Whether actions are already in [-1, 1] (they are, but kept for clarity)
		    cache_images: Whether to cache decompressed images in memory (faster but uses more RAM)
		"""
		self.db_path = db_path
		self.sequence_length = sequence_length
		self.stride = stride
		self.normalize_actions = normalize_actions
		self.cache_images = cache_images

		# Build index of valid sequences
		self.sequences = self._build_sequence_index()

		# Optional image cache
		self.image_cache = {} if cache_images else None

	def _build_sequence_index(self):
		"""
		Build an index of all valid sequences in the database.

		A valid sequence is one where:
		- All frames are from the same episode
		- The sequence has the required length
		- No frames are missing (consecutive steps)
		"""
		conn = sqlite3.connect(self.db_path)
		cursor = conn.cursor()

		# Get all episodes
		cursor.execute(
			"SELECT DISTINCT episode FROM transitions ORDER BY episode"
		)
		episodes = [row[0] for row in cursor.fetchall()]

		sequences = []

		for episode in episodes:
			# Get all transition IDs and steps for this episode, ordered by step
			cursor.execute(
				"SELECT id, step FROM transitions WHERE episode = ? ORDER BY step",
				(episode,),
			)
			transitions = cursor.fetchall()

			# Create sequences with the specified stride
			for i in range(
				0, len(transitions) - self.sequence_length + 1, self.stride
			):
				# Get the sequence of transitions
				seq_transitions = transitions[i : i + self.sequence_length]

				# Verify steps are consecutive
				steps = [t[1] for t in seq_transitions]
				if steps == list(
					range(steps[0], steps[0] + self.sequence_length)
				):
					# Store the transition IDs for this sequence
					transition_ids = [t[0] for t in seq_transitions]
					sequences.append(transition_ids)

		conn.close()

		print(
			f"Built index with {len(sequences)} valid sequences from {len(episodes)} episodes"
		)
		return sequences

	def _decompress_image(
		self, image_blob: bytes, shape: Tuple[int, int, int]
	) -> np.ndarray:
		"""Decompress a gzipped image blob and reshape it."""
		decompressed = gzip.decompress(image_blob)
		image = np.frombuffer(decompressed, dtype=np.float32)
		return image.reshape(shape)

	def _load_image(
		self, conn: sqlite3.Connection, transition_id: int
	) -> np.ndarray:
		"""Load and decompress a single image."""
		# Check cache first
		if self.image_cache is not None and transition_id in self.image_cache:
			return self.image_cache[transition_id]

		cursor = conn.cursor()
		cursor.execute(
			"""SELECT camera_image, camera_shape_0, camera_shape_1, camera_shape_2 
               FROM transitions WHERE id = ?""",
			(transition_id,),
		)
		row = cursor.fetchone()

		if row is None:
			raise ValueError(f"Transition ID {transition_id} not found")

		image_blob, h, w, c = row
		image = self._decompress_image(image_blob, (h, w, c))

		# Cache if enabled
		if self.image_cache is not None:
			self.image_cache[transition_id] = image

		return image

	def _load_action(
		self, conn: sqlite3.Connection, transition_id: int
	) -> np.ndarray:
		"""Load the action for a given transition."""
		cursor = conn.cursor()
		cursor.execute(
			"SELECT action_0, action_1 FROM transitions WHERE id = ?",
			(transition_id,),
		)
		row = cursor.fetchone()

		if row is None:
			raise ValueError(f"Transition ID {transition_id} not found")

		return np.array(row, dtype=np.float32)

	def __len__(self) -> int:
		return len(self.sequences)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Get a sequence of images and corresponding actions.

		Returns:
		    images: Tensor of shape (sequence_length, C, H, W) in range [0, 1]
		    actions: Tensor of shape (sequence_length, 2) with [throttle, steering] in range [-1, 1]
		"""
		transition_ids = self.sequences[idx]

		# Open connection for this sample
		conn = sqlite3.connect(self.db_path)

		# Load all images and actions in the sequence
		images = []
		actions = []

		for tid in transition_ids:
			# Load image (H, W, C) format
			image = self._load_image(conn, tid)
			images.append(image)

			# Load action
			action = self._load_action(conn, tid)
			actions.append(action)

		conn.close()

		# Stack into arrays
		images = np.stack(images, axis=0)  # (T, H, W, C)
		actions = np.stack(actions, axis=0)  # (T, 2)

		# Convert to torch tensors and rearrange to (T, C, H, W)
		images = torch.from_numpy(images).permute(0, 3, 1, 2)  # (T, C, H, W)
		actions = torch.from_numpy(actions)  # (T, 2)

		return images, actions


def create_inverse_dynamics_dataloader(
	db_path: str,
	sequence_length: int = 4,
	stride: int = 1,
	batch_size: int = 32,
	shuffle: bool = True,
	num_workers: int = 4,
	cache_images: bool = False,
	**dataloader_kwargs,
) -> DataLoader:
	"""
	Create a DataLoader for inverse dynamics learning.

	Args:
	    db_path: Path to the SQLite database
	    sequence_length: Number of consecutive frames per sequence
	    stride: Step size when creating sequences
	    batch_size: Batch size for the dataloader
	    shuffle: Whether to shuffle the data
	    num_workers: Number of worker processes for data loading
	    cache_images: Whether to cache decompressed images (uses more memory)
	    **dataloader_kwargs: Additional arguments to pass to DataLoader

	Returns:
	    DataLoader instance
	"""
	dataset = InverseDynamicsDataset(
		db_path=db_path,
		sequence_length=sequence_length,
		stride=stride,
		cache_images=cache_images,
	)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		**dataloader_kwargs,
	)

	return dataloader


if __name__ == "__main__":
	# Example usage
	db_path = "path/to/your/database.db"

	# Create dataloader
	dataloader = create_inverse_dynamics_dataloader(
		db_path=db_path,
		sequence_length=4,
		stride=1,
		batch_size=16,
		shuffle=True,
		num_workers=2,
	)

	# Iterate through a few batches
	print(f"Total batches: {len(dataloader)}")

	for batch_idx, (images, actions) in enumerate(dataloader):
		print(f"\nBatch {batch_idx}:")
		print(f"  Images shape: {images.shape}")  # (B, T, C, H, W)
		print(f"  Actions shape: {actions.shape}")  # (B, T, 2)
		print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
		print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")

		if batch_idx >= 2:  # Just show a few batches
			break
