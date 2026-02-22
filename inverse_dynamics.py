import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional, Tuple

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
	dataset = DrifterDataset(
		db_path=db_path,
		fields=["images", "action"],
		seqlen=sequence_length,
		allow_mid_episode=allow_mid_episode,
		seed=seed,
	)

	n = len(dataset)
	indices = np.arange(n)
	np.random.RandomState(seed).shuffle(indices)
	val_size = int(n * val_fraction)

	def collate(batch):
		return (
			torch.stack([item["images"] for item in batch]),
			torch.stack([item["action"] for item in batch]),
		)

	train_loader = DataLoader(
		Subset(dataset, indices[val_size:]),
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate,
		**dataloader_kwargs,
	)
	val_loader = (
		DataLoader(
			Subset(dataset, indices[:val_size]),
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			collate_fn=collate,
			**dataloader_kwargs,
		)
		if val_size > 0
		else None
	)

	print(f"Dataset split: {n - val_size} training, {val_size} validation")
	return train_loader, val_loader
