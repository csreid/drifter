import gzip
import random
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Semantic field name → database column(s)
FIELDS: Dict[str, List[str]] = {
	"images": [
		"camera_image",
		"camera_shape_0",
		"camera_shape_1",
		"camera_shape_2",
		"camera_dtype",
	],
	"position": ["position_x", "position_y", "position_z"],
	"velocity": ["velocity_x", "velocity_y", "velocity_z"],
	"orientation": [
		"orientation_0",
		"orientation_1",
		"orientation_2",
		"orientation_3",
	],
	"local_goal": ["local_goal_x", "local_goal_y", "local_goal_z"],
	"goal": ["goal_x", "goal_y", "goal_z"],
	"action": ["action_0", "action_1"],
	"is_flipped": ["is_flipped"],
}


def _load_image(blob: bytes, shape: tuple, dtype: str) -> torch.Tensor:
	arr = (
		np.frombuffer(gzip.decompress(blob), dtype=np.dtype(dtype))
		.reshape(shape)
		.copy()
	)
	t = torch.from_numpy(arr).float()
	if t.ndim == 3:
		t = t.permute(2, 0, 1)  # HWC → CHW
	return t / 255.0 if t.max() > 1.0 else t


def _parse_field(field: str, vals: tuple) -> torch.Tensor:
	if field == "images":
		return _load_image(vals[0], (vals[1], vals[2], vals[3]), vals[4])
	return torch.tensor(list(vals), dtype=torch.float32)


class DrifterDataset(Dataset):
	"""
	Dataset for drifter simulation data from a SQLite database.

	Each sample is a dict mapping field name → tensor.
	For seqlen > 1, tensors are shaped [seqlen, ...]; for seqlen == 1, [...].

	Args:
	    db_path:           Path to the SQLite database.
	    fields:            Semantic fields to load. Valid values:
	                       'images', 'position', 'velocity', 'orientation',
	                       'local_goal', 'goal', 'action', 'is_flipped'.
	    seqlen:            Timesteps per sample (1 = single-frame, no episode logic).
	    allow_mid_episode: If True (default), sequences may start anywhere in an
	                       episode. If False, each sequence starts at the beginning
	                       of its episode (one sequence per episode).
	    seed:              Optional random seed.
	"""

	def __init__(
		self,
		db_path: str,
		fields: List[str],
		seqlen: int = 1,
		allow_mid_episode: bool = True,
		seed: Optional[int] = None,
		episode_ids: Optional[List[int]] = None,
	):
		unknown = [f for f in fields if f not in FIELDS]
		if unknown:
			raise ValueError(
				f"Unknown fields: {unknown}. Valid: {sorted(FIELDS)}"
			)

		self.fields = fields
		self.seqlen = seqlen
		self.allow_mid_episode = allow_mid_episode
		self._episode_ids = (
			set(episode_ids) if episode_ids is not None else None
		)

		if seed is not None:
			random.seed(seed)
			np.random.seed(seed)

		# Build flat column list and per-field slices for fast row parsing.
		cols: List[str] = []
		self._slices: Dict[str, Tuple[int, int]] = {}
		for f in fields:
			s = len(cols)
			cols.extend(FIELDS[f])
			self._slices[f] = (s, len(cols))
		self._col_str = ", ".join(cols)

		self.conn = sqlite3.connect(db_path, check_same_thread=False)
		self._build_index()

	def _build_index(self):
		cursor = self.conn.cursor()
		if self.seqlen == 1:
			cursor.execute("SELECT COUNT(*) FROM transitions")
			self._n = cursor.fetchone()[0]
			self.episodes = None
		else:
			cursor.execute(
				"SELECT episode, MIN(id), MAX(id), COUNT(*) "
				"FROM transitions GROUP BY episode ORDER BY MIN(id)"
			)
			self.episodes = [
				{"episode_id": ep, "start_id": s, "end_id": e, "length": n}
				for ep, s, e, n in cursor.fetchall()
				if n >= self.seqlen
				and (self._episode_ids is None or ep in self._episode_ids)
			]
			if not self.episodes:
				raise ValueError(f"No episodes with length >= {self.seqlen}")
			self._n = (
				sum(ep["length"] - self.seqlen + 1 for ep in self.episodes)
				if self.allow_mid_episode
				else len(self.episodes)
			)

	def __len__(self) -> int:
		return self._n

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		if self.seqlen == 1:
			rows = self._fetch(idx + 1, 1)  # SQLite IDs are 1-based
		elif self.allow_mid_episode:
			ep = self.episodes[idx % len(self.episodes)]
			offset = random.randint(0, ep["length"] - self.seqlen)
			rows = self._fetch(ep["start_id"] + offset, self.seqlen)
		else:
			ep = self.episodes[idx % len(self.episodes)]
			rows = self._fetch(ep["start_id"], self.seqlen)

		frames = [
			{f: _parse_field(f, row[s:e]) for f, (s, e) in self._slices.items()}
			for row in rows
		]
		stacked = {
			f: torch.stack([fr[f] for fr in frames]) for f in self.fields
		}
		if self.seqlen == 1:
			return {f: t.squeeze(0) for f, t in stacked.items()}
		return stacked

	def _fetch(self, start_id: int, count: int) -> list:
		cursor = self.conn.cursor()
		cursor.execute(
			f"SELECT {self._col_str} FROM transitions WHERE id >= ? AND id < ? ORDER BY id",
			(start_id, start_id + count),
		)
		return cursor.fetchall()

	def __del__(self):
		if hasattr(self, "conn"):
			self.conn.close()


def DrifterDataloader(
	db_path: str,
	input: List[str],
	output: List[str],
	seqlen: int = 1,
	allow_mid_episode: bool = True,
	batch_size: int = 32,
	shuffle: bool = True,
	num_workers: int = 4,
	seed: Optional[int] = None,
	**kwargs,
) -> DataLoader:
	"""
	Create a DataLoader for drifter simulation data.

	Yields (input_dict, output_dict) batches, where each dict maps field names
	to tensors of shape [batch, seqlen, ...] (or [batch, ...] for seqlen=1).

	Args:
	    db_path:           Path to the SQLite database.
	    input:             Input field names (e.g. ['images']).
	    output:            Target field names (e.g. ['velocity', 'position']).
	    seqlen:            Timesteps per sample. Use 1 for single-frame batches.
	    allow_mid_episode: If True, sequences may start at any point in an episode.
	                       If False, every sequence starts at the episode beginning.
	    batch_size:        Batch size.
	    shuffle:           Whether to shuffle.
	    num_workers:       DataLoader worker count.
	    seed:              Optional random seed passed to DrifterDataset.
	    **kwargs:          Additional DataLoader keyword arguments.

	Example::

	    loader = DrifterDataloader(
	        'drifter_data.db',
	        input=['images'],
	        output=['velocity', 'position'],
	        seqlen=20,
	    )
	    for x, y in loader:
	        imgs = x['images']    # [B, 20, C, H, W]
	        vel  = y['velocity']  # [B, 20, 3]
	"""
	all_fields = list(dict.fromkeys(input + output))
	dataset = DrifterDataset(
		db_path=db_path,
		fields=all_fields,
		seqlen=seqlen,
		allow_mid_episode=allow_mid_episode,
		seed=seed,
	)

	def collate(batch):
		combined = {
			f: torch.stack([item[f] for item in batch]) for f in all_fields
		}
		return {f: combined[f] for f in input}, {f: combined[f] for f in output}

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate,
		pin_memory=True,
		**kwargs,
	)
