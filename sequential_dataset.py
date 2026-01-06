import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict, List, Any, Optional
import random
from abc import ABC, abstractmethod


class SequentialDatabaseDataset(Dataset, ABC):
	"""
	Abstract base class for loading variable-length sequences from SQLite databases.

	Handles:
	- Database connection and episode indexing
	- Sequence sampling (respecting episode boundaries)
	- Variable-length sequence management

	Subclasses must implement:
	- get_columns(): Return list of column names to fetch
	- parse_rows(): Convert database rows into input/target pairs
	"""

	def __init__(
		self,
		db_path: str,
		table_name: str = "transitions",
		episode_column: str = "episode",
		id_column: str = "id",
		min_seq_len: int = 10,
		max_seq_len: int = 50,
		seed: Optional[int] = None,
	):
		"""
		Args:
		    db_path: Path to the SQLite database
		    table_name: Name of the table containing transition data
		    episode_column: Name of the column identifying episodes
		    id_column: Name of the column for ordering (usually a primary key)
		    min_seq_len: Minimum sequence length
		    max_seq_len: Maximum sequence length
		    seed: Random seed for reproducibility
		"""
		self.db_path = db_path
		self.table_name = table_name
		self.episode_column = episode_column
		self.id_column = id_column
		self.min_seq_len = min_seq_len
		self.max_seq_len = max_seq_len

		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)

		# Connect to database and build episode index
		self.conn = sqlite3.connect(db_path, check_same_thread=False)
		self._build_episode_index()

	def _build_episode_index(self):
		"""
		Build an index of all episodes and their row ranges.
		This allows us to sample sequences without crossing episode boundaries.
		"""
		cursor = self.conn.cursor()

		# Get all episodes with their row ranges
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

	def _fetch_sequence(self, start_id: int, seq_len: int) -> List[Tuple]:
		"""
		Fetch a sequence of rows from the database.

		Args:
		    start_id: Starting row ID
		    seq_len: Number of rows to fetch

		Returns:
		    List of database rows as tuples
		"""
		cursor = self.conn.cursor()
		columns = self.get_columns()
		columns_str = ", ".join(columns)

		query = f"""
            SELECT {columns_str}
            FROM {self.table_name}
            WHERE {self.id_column} >= ? AND {self.id_column} < ?
            ORDER BY {self.id_column}
        """
		cursor.execute(query, (start_id, start_id + seq_len))

		return cursor.fetchall()

	@abstractmethod
	def get_columns(self) -> List[str]:
		"""
		Return the list of column names to fetch from the database.

		Returns:
		    List of column names (strings)
		"""
		pass

	@abstractmethod
	def parse_rows(self, rows: List[Tuple]) -> Tuple[Any, Any, int]:
		"""
		Parse database rows into input/target pairs.

		This method should handle all data-specific parsing, transformation,
		and tensorization.

		Args:
		    rows: List of database row tuples

		Returns:
		    inputs: Model inputs (e.g., images tensor, combined features, etc.)
		    targets: Model targets (e.g., state dict, next state, etc.)
		    seq_len: Actual length of the sequence
		"""
		pass

	def __getitem__(self, idx: int) -> Tuple[Any, Any, int]:
		"""
		Get a sequence sample from the dataset.

		Args:
		    idx: Index of the sample (used to seed episode selection)

		Returns:
		    inputs: Model inputs (defined by parse_rows)
		    targets: Model targets (defined by parse_rows)
		    seq_len: Actual length of the sequence
		"""
		# Select an episode (use idx for some determinism)
		episode_idx = idx % len(self.episodes)
		episode_info = self.episodes[episode_idx]

		# Sample a sequence from this episode
		start_id, seq_len = self._sample_sequence_from_episode(episode_info)

		# Fetch the sequence from database
		rows = self._fetch_sequence(start_id, seq_len)

		# Let subclass parse the rows
		return self.parse_rows(rows)

	def __del__(self):
		"""Close database connection when dataset is deleted."""
		if hasattr(self, "conn"):
			self.conn.close()


def collate_sequences_generic(batch):
	"""
	Generic collate function for variable-length sequences.
	Handles different return types from parse_rows.

	Works with:
	- Tensors: Pads along sequence dimension
	- Dicts of tensors: Pads each value
	- Tuples/lists: Recursively handles each element

	Args:
	    batch: List of (inputs, targets, seq_len) tuples

	Returns:
	    inputs_padded: Padded inputs
	    targets_padded: Padded targets
	    seq_lengths: Tensor of actual sequence lengths [batch_size]
	"""
	inputs_list = [item[0] for item in batch]
	targets_list = [item[1] for item in batch]
	seq_lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)

	def pad_item(item_list):
		"""Recursively pad items based on their type."""
		if isinstance(item_list[0], torch.Tensor):
			# Pad tensor sequences
			return pad_sequence(item_list, batch_first=True, padding_value=0.0)
		elif isinstance(item_list[0], dict):
			# Pad each dictionary value
			padded_dict = {}
			for key in item_list[0].keys():
				sequences = [item[key] for item in item_list]
				padded_dict[key] = pad_item(sequences)
			return padded_dict
		elif isinstance(item_list[0], (tuple, list)):
			# Recursively handle tuples/lists
			padded_items = []
			for i in range(len(item_list[0])):
				sequences = [item[i] for item in item_list]
				padded_items.append(pad_item(sequences))
			return type(item_list[0])(padded_items)
		else:
			# Can't pad this type, return as list
			return item_list

	inputs_padded = pad_item(inputs_list)
	targets_padded = pad_item(targets_list)

	return inputs_padded, targets_padded, seq_lengths


def create_sequential_dataloader(
	dataset: Dataset,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 4,
	collate_fn=None,
) -> DataLoader:
	"""
	Create a DataLoader for sequential data.

	Args:
	    dataset: Dataset instance (subclass of SequentialDatabaseDataset)
	    batch_size: Batch size for the dataloader
	    shuffle: Whether to shuffle the data
	    num_workers: Number of worker processes for data loading
	    collate_fn: Custom collate function (defaults to collate_sequences_generic)

	Returns:
	    DataLoader instance
	"""
	if collate_fn is None:
		collate_fn = collate_sequences_generic

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate_fn,
		pin_memory=False,
	)

	return dataloader
