import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict, List, Optional
import random
import gzip


class ForwardDynamicsDataset(Dataset):
	"""
	Dataset for training forward dynamics in latent space.

	Uses a pre-trained inverse dynamics model to encode image sequences into
	hidden states, then provides (h_t, a_t) -> h_{t+1} pairs for training.

	Args:
	    db_path: Path to the SQLite database
	    id_model: Pre-trained inverse dynamics model (frozen)
	    device: Device to run ID model on ('cuda' or 'cpu')
	    lstm_context: Number of images needed by LSTM to produce hidden state
	    table_name: Name of the table containing transition data
	    episode_column: Name of the column identifying episodes
	    id_column: Name of the column for ordering
	    min_seq_len: Minimum sequence length (must be >= lstm_context + 1)
	    max_seq_len: Maximum sequence length
	    seed: Random seed for reproducibility
	"""

	def __init__(
		self,
		db_path: str,
		id_model: torch.nn.Module,
		device: str = "cuda",
		lstm_context: int = 3,
		table_name: str = "transitions",
		episode_column: str = "episode",
		id_column: str = "id",
		min_seq_len: Optional[int] = None,
		max_seq_len: int = 50,
		seed: Optional[int] = None,
	):
		self.db_path = db_path
		self.table_name = table_name
		self.episode_column = episode_column
		self.id_column = id_column
		self.lstm_context = lstm_context
		self.device = device

		# Minimum sequence must be long enough to produce at least one transition
		# We need lstm_context images to get h_0, then 1 more for h_1
		if min_seq_len is None:
			min_seq_len = lstm_context + 1
		else:
			if min_seq_len < lstm_context + 1:
				raise ValueError(
					f"min_seq_len must be >= lstm_context + 1 "
					f"(got {min_seq_len}, need >= {lstm_context + 1})"
				)

		self.min_seq_len = min_seq_len
		self.max_seq_len = max_seq_len

		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)

		# Store and freeze the ID model
		self.id_model = id_model.to(device)
		self.id_model.eval()
		for param in self.id_model.parameters():
			param.requires_grad = False

		# Connect to database and build episode index
		self.conn = sqlite3.connect(db_path, check_same_thread=False)
		self._build_episode_index()

	def _build_episode_index(self):
		"""Build an index of all episodes and their row ranges."""
		cursor = self.conn.cursor()

		query = f"""
            SELECT 
                {self.episode_column},
                MIN({self.id_column}) as start_id,
                MAX({self.id_column}) as end_id,
                COUNT(*) as length
            FROM {self.table_name}
            GROUP BY {self.episode_column}
            ORDER BY MIN({self.id_column})
        """
		cursor.execute(query)

		self.episodes = []
		for row in cursor.fetchall():
			episode_id, start_id, end_id, length = row
			if length >= self.min_seq_len:
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

		# Calculate total number of valid transitions we can sample
		# Each episode of length L can produce (L - lstm_context) transitions
		self.num_transitions = sum(
			max(1, ep["length"] - self.lstm_context) for ep in self.episodes
		)

	def __len__(self) -> int:
		return self.num_transitions

	def _decompress_image(
		self, blob: bytes, shape: Tuple[int, int, int], dtype: str
	) -> np.ndarray:
		"""Decompress a gzipped image blob."""
		decompressed = gzip.decompress(blob)
		dtype_np = np.dtype(dtype)
		img = np.frombuffer(decompressed, dtype=dtype_np)
		return img.reshape(shape)

	def _sample_sequence_from_episode(
		self, episode_info: Dict
	) -> Tuple[int, int]:
		"""
		Sample a random sequence start and length from an episode.

		Returns:
		    start_id: Starting row ID
		    seq_len: Length of the sequence (in frames)
		"""
		episode_length = episode_info["length"]

		# Determine sequence length
		max_possible_len = min(self.max_seq_len, episode_length)
		seq_len = random.randint(self.min_seq_len, max_possible_len)

		# Sample starting position
		max_start_offset = episode_length - seq_len
		start_offset = random.randint(0, max_start_offset)
		start_id = episode_info["start_id"] + start_offset

		return start_id, seq_len

	def _fetch_sequence(self, start_id: int, seq_len: int) -> List[Tuple]:
		"""Fetch a sequence of rows from the database."""
		cursor = self.conn.cursor()

		columns = [
			"camera_image",
			"camera_shape_0",
			"camera_shape_1",
			"camera_shape_2",
			"camera_dtype",
			"action_0",
			"action_1",
		]
		columns_str = ", ".join(columns)

		query = f"""
            SELECT {columns_str}
            FROM {self.table_name}
            WHERE {self.id_column} >= ? AND {self.id_column} < ?
            ORDER BY {self.id_column}
        """
		cursor.execute(query, (start_id, start_id + seq_len))

		return cursor.fetchall()

	def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
		"""
		Encode a sequence of images to hidden states using the ID model.

		Args:
		    images: [seq_len, C, H, W] tensor

		Returns:
		    hidden_states: [seq_len - lstm_context + 1, hidden_dim] tensor
		"""
		seq_len = images.shape[0]

		# We need to extract hidden states for overlapping windows
		# For a sequence of length N with context C:
		# - h_0 comes from images[0:C]
		# - h_1 comes from images[1:C+1]
		# - h_N-C comes from images[N-C:N]

		num_hidden_states = seq_len - self.lstm_context + 1
		hidden_states = []

		with torch.no_grad():
			for i in range(num_hidden_states):
				# Get context window
				img_window = images[i : i + self.lstm_context].unsqueeze(
					0
				)  # [1, context, C, H, W]
				img_window = img_window.to(self.device)

				# Get hidden state (last timestep of this window)
				seq_lens = torch.tensor(
					[self.lstm_context], dtype=torch.long
				)
				hidden = self.id_model._get_hidden(
					img_window, seq_lens
				)  # [1, hidden_dim]

				hidden_states.append(hidden.cpu())

		# Stack all hidden states
		hidden_states = torch.cat(
			hidden_states, dim=0
		)  # [num_hidden_states, hidden_dim]

		return hidden_states

	def __getitem__(
		self, idx: int
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
		"""
		Get a batch of (h_t, a_t) -> h_{t+1} transitions.

		Returns:
		    hidden_states: [num_transitions, hidden_dim] - h_t values
		    actions: [num_transitions, action_dim] - a_t values
		    next_hidden_states: [num_transitions, hidden_dim] - h_{t+1} values
		    num_transitions: int - actual number of transitions in this sequence
		"""
		# Select an episode
		episode_idx = idx % len(self.episodes)
		episode_info = self.episodes[episode_idx]

		# Sample a sequence from this episode
		start_id, seq_len = self._sample_sequence_from_episode(episode_info)

		# Fetch the sequence from database
		rows = self._fetch_sequence(start_id, seq_len)

		# Parse images and actions
		images = []
		actions = []

		for row in rows:
			(
				camera_blob,
				shape_0,
				shape_1,
				shape_2,
				dtype_str,
				action_0,
				action_1,
			) = row

			# Decompress image
			shape = (shape_2, shape_1, shape_0)
			img = self._decompress_image(camera_blob, shape, dtype_str)
			images.append(img)

			# Store action
			actions.append([action_0, action_1])

		# Convert to tensors
		images = torch.from_numpy(
			np.stack(images)
		).float()  # [seq_len, C, H, W]
		actions = torch.tensor(actions, dtype=torch.float32)  # [seq_len, 2]

		# Encode images to hidden states
		# This gives us hidden states for positions 0, 1, ..., seq_len - lstm_context
		hidden_states = self._encode_images(
			images
		)  # [seq_len - context + 1, hidden_dim]

		# Create transitions: (h_t, a_t) -> h_{t+1}
		# We have actions at positions 0, 1, ..., seq_len - 1
		# We have hidden states at positions 0, 1, ..., seq_len - context
		# So we can create transitions from positions 0 to seq_len - context - 1

		num_transitions = hidden_states.shape[0] - 1

		h_t = hidden_states[:-1]  # [num_transitions, hidden_dim]
		a_t = actions[self.lstm_context - 1 : -1]  # [num_transitions, 2]
		h_t_next = hidden_states[1:]  # [num_transitions, hidden_dim]

		return h_t, a_t, h_t_next, num_transitions

	def __del__(self):
		"""Close database connection when dataset is deleted."""
		if hasattr(self, "conn"):
			self.conn.close()


def collate_forward_dynamics(batch):
	"""
	Collate function for forward dynamics dataset.

	Handles variable-length sequences by padding.

	Args:
	    batch: List of (h_t, a_t, h_t_next, num_transitions) tuples

	Returns:
	    h_t_padded: [batch_size, max_transitions, hidden_dim]
	    a_t_padded: [batch_size, max_transitions, action_dim]
	    h_t_next_padded: [batch_size, max_transitions, hidden_dim]
	    seq_lengths: [batch_size] - number of valid transitions in each sample
	"""
	h_t_list = [item[0] for item in batch]
	a_t_list = [item[1] for item in batch]
	h_t_next_list = [item[2] for item in batch]
	seq_lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)

	# Pad sequences
	h_t_padded = pad_sequence(h_t_list, batch_first=True, padding_value=0.0)
	a_t_padded = pad_sequence(a_t_list, batch_first=True, padding_value=0.0)
	h_t_next_padded = pad_sequence(
		h_t_next_list, batch_first=True, padding_value=0.0
	)

	return h_t_padded, a_t_padded, h_t_next_padded, seq_lengths


def create_forward_dynamics_dataloader(
	db_path: str,
	id_model: torch.nn.Module,
	device: str = "cuda",
	sequence_length: int = 5,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 0,  # Set to 0 to avoid issues with CUDA and multiprocessing
	**dataset_kwargs,
) -> DataLoader:
	"""
	Create a DataLoader for forward dynamics training in latent space.

	Args:
	    db_path: Path to the SQLite database
	    id_model: Pre-trained inverse dynamics model
	    device: Device to run ID model on
	    sequence_length: Number of images needed by LSTM
	    batch_size: Batch size
	    shuffle: Whether to shuffle
	    num_workers: Number of worker processes (recommend 0 for CUDA models)
	    **dataset_kwargs: Additional arguments for ForwardDynamicsDataset

	Returns:
	    DataLoader instance
	"""
	dataset = ForwardDynamicsDataset(
		db_path=db_path,
		id_model=id_model,
		device=device,
		lstm_context=sequence_length,
		**dataset_kwargs,
	)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate_forward_dynamics,
		pin_memory=False,
	)

	return dataloader
