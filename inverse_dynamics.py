import sqlite3

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple

from drifter_dataloader import DrifterDataset


def create_inverse_dynamics_dataloaders(
	db_path: str,
	sequence_length: int = 4,
	batch_size: int = 32,
	val_fraction: float = 0.2,
	shuffle: bool = True,
	num_workers: int = 4,
	seed: int = 42,
	stride: int = 1,  # kept for call-site compatibility, unused
	cache_images: bool = False,  # kept for call-site compatibility, unused
	allow_mid_episode: bool = True,
	**dataloader_kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
	"""
	Create training and validation DataLoaders for inverse dynamics learning.

	Each batch yields (images, actions):
	    images:  [B, T, C, H, W]
	    actions: [B, T, 2]
	"""
	# Split on episodes to avoid frame-level leakage from overlapping sequences.
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	cursor.execute(
		"SELECT episode FROM transitions GROUP BY episode HAVING COUNT(*) >= ?",
		(sequence_length,),
	)
	all_episodes: List[int] = [row[0] for row in cursor.fetchall()]
	conn.close()

	rng = np.random.RandomState(seed)
	rng.shuffle(all_episodes)
	n_val = int(len(all_episodes) * val_fraction)
	val_episodes = all_episodes[:n_val]
	train_episodes = all_episodes[n_val:]

	def make_dataset(episode_ids):
		return DrifterDataset(
			db_path=db_path,
			fields=["images", "action"],
			seqlen=sequence_length,
			allow_mid_episode=allow_mid_episode,
			seed=seed,
			episode_ids=episode_ids,
		)

	def collate(batch):
		return (
			torch.stack([item["images"] for item in batch]),
			torch.stack([item["action"] for item in batch]),
		)

	train_loader = DataLoader(
		make_dataset(train_episodes),
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate,
		**dataloader_kwargs,
	)
	val_loader = (
		DataLoader(
			make_dataset(val_episodes),
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			collate_fn=collate,
			**dataloader_kwargs,
		)
		if n_val > 0
		else None
	)

	print(
		f"Episode split: {len(train_episodes)} train, {len(val_episodes)} val"
		f" (of {len(all_episodes)} total episodes)"
	)
	return train_loader, val_loader
