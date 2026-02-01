import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict, List, Optional
import random
import gzip

class ForwardDynamicsDecoderDataset(Dataset):
	"""
	Dataset for training forward dynamics + decoder in latent space.

	Uses a pre-trained inverse dynamics model to encode image sequences into
	hidden states, then provides:
	- (h_t, a_t) -> h_{t+1} pairs for forward dynamics
	- h_t -> state_t pairs for decoder

	All positions are made relative to the initial position of the sequence.

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

		# Connect to database and build episode index
		self.conn = sqlite3.connect(db_path, check_same_thread=False)
		print(f'Building episode index...')
		self._build_episode_index()

		print(f'Presampling sequence windows...')
		self._sample_all_sequences()


		self._embedding_cache = {}

	def _sample_all_sequences():
		self.sequences = []
		for _ in range(self.num_transitions):
			episode_idx = random.randint(0, len(self.episodes) - 1)
			episode_info = self.episodes[episode_idx]
			start_id, seq_len = self._sample_sequence_from_episode(episode_info)
			self.sequences.append((episode_info['episode_id'], start_id, seq_len))

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
			# State variables
			"position_x",
			"position_y",
			"position_z",
			"local_goal_x",
			"local_goal_y",
			"local_goal_z",
			"velocity_x",
			"velocity_y",
			"velocity_z",
			"is_flipped",
			"orientation_0",
			"orientation_1",
			"orientation_2",
			"orientation_3",
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

	def _parse_state(
		self, row: Tuple, initial_position: np.ndarray
	) -> np.ndarray:
		"""
		Parse state variables from a database row.

		Makes positions relative to initial_position.

		Returns:
		    state: [14] array with [pos(3), local_goal(3), vel(3), is_flipped(1), orient(4)]
		"""
		(
			_camera_blob,
			_shape_0,
			_shape_1,
			_shape_2,
			_dtype_str,
			_action_0,
			_action_1,
			pos_x,
			pos_y,
			pos_z,
			local_goal_x,
			local_goal_y,
			local_goal_z,
			vel_x,
			vel_y,
			vel_z,
			is_flipped,
			orient_0,
			orient_1,
			orient_2,
			orient_3,
		) = row

		# Make position relative to start
		position = np.array([pos_x, pos_y, pos_z]) - initial_position
		local_goal = np.array([local_goal_x, local_goal_y, local_goal_z])
		velocity = np.array([vel_x, vel_y, vel_z])
		orientation = np.array([orient_0, orient_1, orient_2, orient_3])

		# Concatenate: [pos(3), local_goal(3), vel(3), is_flipped(1), orient(4)]
		state = np.concatenate(
			[position, local_goal, velocity, [is_flipped], orientation]
		)

		return state

	def __getitem__(
		self, idx: int
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		int,
	]:
		"""
		Get a batch of transitions with states for decoder.

		Returns:
		    hidden_states: [num_transitions, hidden_dim] - h_t values
		    actions: [num_transitions, action_dim] - a_t values
		    next_hidden_states: [num_transitions, hidden_dim] - h_{t+1} values
		    states: [num_transitions, state_dim] - ground truth states at t
		    next_states: [num_transitions, state_dim] - ground truth states at t+1
		    num_transitions: int - actual number of transitions in this sequence
		"""

		episode_id, start_id, seq_len = self.sequences[idx]
		cache_key = (episode_id, start_id, seq_len)

		if cache_key in self._embedding_cache:
			return self._embedding_cache[idx]

		# Select an episode
		episode_idx = idx % len(self.episodes)
		episode_info = self.episodes[episode_idx]

		# Sample a sequence from this episode
		start_id, seq_len = self._sample_sequence_from_episode(episode_info)

		# Fetch the sequence from database
		rows = self._fetch_sequence(start_id, seq_len)

		# Get initial position for making everything relative
		initial_position = np.array(
			[rows[0][7], rows[0][8], rows[0][9]]
		)  # position_x, y, z

		# Parse images, actions, and states
		images = []
		actions = []
		states = []

		for row in rows:
			# Decompress image
			camera_blob = row[0]
			shape = (row[1], row[2], row[3])
			dtype_str = row[4]
			img = self._decompress_image(camera_blob, shape, dtype_str)
			images.append(img)

			# Store action
			action_0, action_1 = row[5], row[6]
			actions.append([action_0, action_1])

			# Parse state (relative to initial position)
			state = self._parse_state(row, initial_position)
			states.append(state)

		# Convert to tensors
		images = (
			torch.from_numpy(np.stack(images)).float().permute(0, 3, 1, 2)
		)  # [T, C, H, W]
		actions = torch.tensor(actions, dtype=torch.float32)  # [seq_len, 2]
		states = torch.tensor(
			np.stack(states), dtype=torch.float32
		)  # [seq_len, 14]

		num_transitions = (
			seq_len - self.lstm_context
		)  # Need context+1 frames per transition
		h_t_list = []
		a_t_list = []
		h_t_next_list = []
		state_t_list = []
		state_t_next_list = []

		with torch.no_grad():
			for i in range(num_transitions):
				# Get h_t from context window
				imgs_t = (
					images[i : i + self.lstm_context]
					.unsqueeze(0)
					.to(self.device)
				)  # [1, context, C, H, W]
				seq_lens_t = torch.tensor([self.lstm_context], dtype=torch.long)
				h_t = self.id_model._get_hidden(imgs_t, seq_lens_t)[
					:, -1, :
				]  # [1, hidden_dim]

				# Get h_{t+1} from context+1 window
				imgs_t_next = (
					images[i : i + self.lstm_context + 1]
					.unsqueeze(0)
					.to(self.device)
				)  # [1, context+1, C, H, W]
				seq_lens_t_next = torch.tensor(
					[self.lstm_context + 1], dtype=torch.long
				)
				h_t_next = self.id_model._get_hidden(
					imgs_t_next, seq_lens_t_next
				)[:, -1, :]  # [1, hidden_dim]

				# Get action at time i+context-1 (the action taken after seeing context images)
				a_t = actions[i + self.lstm_context - 1]

				# Get states at t and t+1
				# State at t corresponds to the last frame of the context window
				state_t = states[i + self.lstm_context - 1]
				# State at t+1 is one frame later
				state_t_next = states[i + self.lstm_context]

				h_t_list.append(h_t.cpu().squeeze(0))
				h_t_next_list.append(h_t_next.cpu().squeeze(0))
				a_t_list.append(a_t)
				state_t_list.append(state_t)
				state_t_next_list.append(state_t_next)

		# Stack into tensors
		h_t_batch = torch.stack(h_t_list)  # [num_transitions, hidden_dim]
		a_t_batch = torch.stack(a_t_list)  # [num_transitions, action_dim]
		h_t_next_batch = torch.stack(
			h_t_next_list
		)  # [num_transitions, hidden_dim]
		state_t_batch = torch.stack(state_t_list)  # [num_transitions, 14]
		state_t_next_batch = torch.stack(
			state_t_next_list
		)  # [num_transitions, 14]

		val = (
			h_t_batch,
			a_t_batch,
			h_t_next_batch,
			state_t_batch,
			state_t_next_batch,
			num_transitions,
		)

		self._embedding_cache[idx] = val
		return val

	def __del__(self):
		"""Close database connection when dataset is deleted."""
		if hasattr(self, "conn"):
			self.conn.close()


def collate_forward_dynamics_decoder(batch):
	"""
	Collate function for forward dynamics + decoder dataset.

	Handles variable-length sequences by padding.

	Args:
	    batch: List of (h_t, a_t, h_t_next, state_t, state_t_next, num_transitions) tuples

	Returns:
	    h_t_padded: [batch_size, max_transitions, hidden_dim]
	    a_t_padded: [batch_size, max_transitions, action_dim]
	    h_t_next_padded: [batch_size, max_transitions, hidden_dim]
	    state_t_padded: [batch_size, max_transitions, state_dim]
	    state_t_next_padded: [batch_size, max_transitions, state_dim]
	    seq_lengths: [batch_size] - number of valid transitions in each sample
	"""
	h_t_list = [item[0] for item in batch]
	a_t_list = [item[1] for item in batch]
	h_t_next_list = [item[2] for item in batch]
	state_t_list = [item[3] for item in batch]
	state_t_next_list = [item[4] for item in batch]
	seq_lengths = torch.tensor([item[5] for item in batch], dtype=torch.long)

	# Pad sequences
	h_t_padded = pad_sequence(h_t_list, batch_first=True, padding_value=0.0)
	a_t_padded = pad_sequence(a_t_list, batch_first=True, padding_value=0.0)
	h_t_next_padded = pad_sequence(
		h_t_next_list, batch_first=True, padding_value=0.0
	)
	state_t_padded = pad_sequence(
		state_t_list, batch_first=True, padding_value=0.0
	)
	state_t_next_padded = pad_sequence(
		state_t_next_list, batch_first=True, padding_value=0.0
	)

	return (
		h_t_padded,
		a_t_padded,
		h_t_next_padded,
		state_t_padded,
		state_t_next_padded,
		seq_lengths,
	)


def create_forward_dynamics_decoder_dataloader(
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
	Create a DataLoader for forward dynamics + decoder training in latent space.

	Args:
	    db_path: Path to the SQLite database
	    id_model: Pre-trained inverse dynamics model
	    device: Device to run ID model on
	    sequence_length: Number of images needed by LSTM
	    batch_size: Batch size
	    shuffle: Whether to shuffle
	    num_workers: Number of worker processes (recommend 0 for CUDA models)
	    **dataset_kwargs: Additional arguments for ForwardDynamicsDecoderDataset

	Returns:
	    DataLoader instance
	"""
	dataset = ForwardDynamicsDecoderDataset(
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
		collate_fn=collate_forward_dynamics_decoder,
		pin_memory=False,
	)

	return dataloader
