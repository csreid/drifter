import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Optional
import pickle


class PrecomputedEmbeddingDataset(Dataset):
	"""
	Dataset for training forward dynamics using pre-computed embeddings.

	Loads pre-computed latent embeddings and states from a database created
	by the embed.py script.

	Args:
	    embedding_db_path: Path to the SQLite database with pre-computed embeddings
	    seed: Random seed for reproducibility
	"""

	def __init__(
		self,
		embedding_db_path: str,
		seed: Optional[int] = None,
	):
		self.db_path = embedding_db_path

		if seed is not None:
			torch.manual_seed(seed)

		# Connect to database
		self.conn = sqlite3.connect(embedding_db_path, check_same_thread=False)
		self._build_index()

	def _build_index(self):
		"""Build an index of all available embeddings."""
		cursor = self.conn.cursor()

		# Get count of embeddings
		cursor.execute("SELECT COUNT(*) FROM embeddings")
		self.num_samples = cursor.fetchone()[0]

		if self.num_samples == 0:
			raise ValueError(
				f"No embeddings found in database {self.db_path}. "
				f"Did you run embed.py first?"
			)

	def __len__(self) -> int:
		return self.num_samples

	def __getitem__(
		self, idx: int
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
		"""
		Get a pre-computed embedding by index.

		Returns:
		    hidden_states: [num_transitions, hidden_dim] - h_t values
		    actions: [num_transitions, action_dim] - a_t values
		    next_hidden_states: [num_transitions, hidden_dim] - h_{t+1} values
		    states: [num_transitions, state_dim] - ground truth states at t
		    next_states: [num_transitions, state_dim] - ground truth states at t+1
		    num_transitions: int - actual number of transitions in this sequence
		"""
		cursor = self.conn.cursor()

		# Fetch embedding by rowid (SQLite's implicit row number)
		# Note: rowid starts at 1, so we add 1 to idx
		cursor.execute(
			"""
            SELECT h_t, a_t, h_t_next, state_t, state_t_next, num_transitions
            FROM embeddings
            WHERE rowid = ?
        """,
			(idx + 1,),
		)

		row = cursor.fetchone()
		if row is None:
			raise IndexError(f"Index {idx} out of range")

		h_t_blob, a_t_blob, h_t_next_blob, state_t_blob, state_t_next_blob, num_transitions = (
			row
		)

		# Deserialize tensors
		h_t = torch.from_numpy(pickle.loads(h_t_blob))
		a_t = torch.from_numpy(pickle.loads(a_t_blob))
		h_t_next = torch.from_numpy(pickle.loads(h_t_next_blob))
		state_t = torch.from_numpy(pickle.loads(state_t_blob))
		state_t = torch.cat([state_t[..., :3], state_t[..., 6:]], dim=-1)

		state_t_next = torch.from_numpy(pickle.loads(state_t_next_blob))
		state_t_next = torch.cat([state_t_next[..., :3], state_t_next[..., 6:]], dim=-1)

		return h_t, a_t, h_t_next, state_t, state_t_next, num_transitions

	def __del__(self):
		"""Close database connection when dataset is deleted."""
		if hasattr(self, "conn"):
			self.conn.close()


def collate_precomputed_embeddings(batch):
	"""
	Collate function for pre-computed embeddings.

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
	state_t_padded = pad_sequence(state_t_list, batch_first=True, padding_value=0.0)
	state_t_next_padded = pad_sequence(
		state_t_next_list, batch_first=True, padding_value=0.0
	)

	return h_t_padded, a_t_padded, h_t_next_padded, state_t_padded, state_t_next_padded, seq_lengths


def create_precomputed_dataloader(
	embedding_db_path: str,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 4,  # Can use more workers since no CUDA operations
	seed: Optional[int] = None,
) -> DataLoader:
	"""
	Create a DataLoader for pre-computed embeddings.

	Args:
	    embedding_db_path: Path to the SQLite database with pre-computed embeddings
	    batch_size: Batch size
	    shuffle: Whether to shuffle
	    num_workers: Number of worker processes
	    seed: Random seed for reproducibility

	Returns:
	    DataLoader instance
	"""
	dataset = PrecomputedEmbeddingDataset(
		embedding_db_path=embedding_db_path,
		seed=seed,
	)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate_precomputed_embeddings,
		pin_memory=True,  # Can use pin_memory now since no GPU operations in dataset
	)

	return dataloader
