"""
Backward-compatible sequential drifter dataloaders.
New code should use DrifterDataloader from drifter_dataloader directly.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Optional

from drifter_dataloader import DrifterDataset

_STATE_FIELDS = ["position", "orientation", "velocity", "local_goal", "goal"]


def _make_loader(dataset, batch_size, shuffle, num_workers):
	seqlen = dataset.seqlen

	def collate(batch):
		images = torch.stack([item["images"] for item in batch])
		states = {
			f: torch.stack([item[f] for item in batch]) for f in _STATE_FIELDS
		}
		seq_lens = torch.full((len(batch),), seqlen, dtype=torch.long)
		return images, states, seq_lens

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate,
	)


def create_drifter_dataloader(
	db_path: str,
	min_seq_len: int = 10,
	max_seq_len: int = 50,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 4,
	transform=None,
	seed: Optional[int] = None,
	test_split: float = 0.2,
	allow_mid_episode: bool = True,
):
	"""
	Create train and test DataLoaders returning (images, states, seq_lens) batches.
	Uses max_seq_len as the fixed sequence length.
	"""
	fields = ["images"] + _STATE_FIELDS
	dataset = DrifterDataset(
		db_path=db_path,
		fields=fields,
		seqlen=max_seq_len,
		allow_mid_episode=allow_mid_episode,
		seed=seed,
	)

	episodes = dataset.episodes
	rng = np.random.RandomState(seed if seed is not None else 42)
	order = rng.permutation(len(episodes))
	test_size = int(len(episodes) * test_split)
	test_ep_idx = set(order[:test_size])
	train_ep_idx = set(order[test_size:])

	print(
		f"Total episodes: {len(episodes)}, train: {len(train_ep_idx)}, test: {len(test_ep_idx)}"
	)

	def make_split(ep_idx):
		ds = DrifterDataset(
			db_path=db_path,
			fields=fields,
			seqlen=max_seq_len,
			allow_mid_episode=allow_mid_episode,
			seed=seed,
		)
		ds.episodes = [episodes[i] for i in sorted(ep_idx)]
		ds._n = sum(ep["length"] - max_seq_len + 1 for ep in ds.episodes)
		return ds

	train_loader = _make_loader(
		make_split(train_ep_idx), batch_size, shuffle, num_workers
	)
	test_loader = _make_loader(
		make_split(test_ep_idx), batch_size, False, num_workers
	)
	return train_loader, test_loader


# Alias used by render_episode.py
create_sequence_dataloader = create_drifter_dataloader
