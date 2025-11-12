import sqlite3
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict


class DrifterDataset(Dataset):
	"""
	PyTorch Dataset for loading drifter simulation data from SQLite database.

	Returns:
	    X: Camera images (compressed in DB, decompressed on load)
	    Y: Dictionary containing:
	        - position: [x, y, z]
	        - orientation: [qw, qx, qy, qz] (quaternion)
	        - velocity: [vx, vy, vz]
	        - local_goal: [x, y, z]
	        - goal: [x, y, z]
	"""

	def __init__(self, db_path: str, transform=None):
		"""
		Args:
		    db_path: Path to the SQLite database
		    transform: Optional transform to apply to images
		"""
		self.db_path = db_path
		self.transform = transform

		# Connect to database and get total count
		self.conn = sqlite3.connect(db_path, check_same_thread=False)
		cursor = self.conn.cursor()
		cursor.execute("SELECT COUNT(*) FROM transitions")
		self.length = cursor.fetchone()[0]

	def __len__(self) -> int:
		return self.length

	def __getitem__(
		self, idx: int
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""
		Get a single sample from the dataset.

		Args:
		    idx: Index of the sample

		Returns:
		    image: Camera image tensor (C, H, W)
		    state_dict: Dictionary of state tensors
		"""
		cursor = self.conn.cursor()

		# Fetch the row (SQLite uses 1-based indexing for ROWID)
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
            WHERE id = ?
        """,
			(idx + 1,),
		)  # +1 because AUTOINCREMENT starts at 1

		row = cursor.fetchone()

		if row is None:
			raise IndexError(f"Index {idx} out of range")

		# Parse state components
		position = np.array([row[0], row[1], row[2]], dtype=np.float32)
		orientation = np.array(
			[row[3], row[4], row[5], row[6]], dtype=np.float32
		)
		velocity = np.array([row[7], row[8], row[9]], dtype=np.float32)
		local_goal = np.array([row[10], row[11], row[12]], dtype=np.float32)
		goal = np.array([row[13], row[14], row[15]], dtype=np.float32)

		# Decompress and reshape camera image
		compressed_img = row[16]
		shape = (row[17], row[18], row[19])
		dtype = row[20]

		decompressed = gzip.decompress(compressed_img)
		image = np.frombuffer(decompressed, dtype=dtype).reshape(shape)

		# Convert to torch tensors
		# Image: Convert from (H, W, C) to (C, H, W) and normalize to [0, 1]
		image = torch.from_numpy(image).float()
		if image.ndim == 3:  # If image has channels
			image = image.permute(2, 0, 1)  # HWC to CHW
		if image.max() > 1.0:  # If not already normalized
			image = image / 255.0

		# Apply optional transform
		if self.transform is not None:
			image = self.transform(image)

		# Create state dictionary
		y = {
			"position": torch.from_numpy(position),
			"orientation": torch.from_numpy(orientation),
			"velocity": torch.from_numpy(velocity),
			"local_goal": torch.from_numpy(local_goal),
			"goal": torch.from_numpy(goal),
		}

		return image, y

	def __del__(self):
		"""Close database connection when dataset is deleted."""
		if hasattr(self, "conn"):
			self.conn.close()


def collate_fn(batch):
	"""
	Custom collate function to handle dictionary outputs.

	Args:
	    batch: List of (image, state_dict) tuples

	Returns:
	    images: Batched images tensor
	    states: Dictionary of batched state tensors
	"""
	images = torch.stack([item[0] for item in batch])

	# Stack each component of the state dictionary
	states = {
		"position": torch.stack([item[1]["position"] for item in batch]),
		"orientation": torch.stack([item[1]["orientation"] for item in batch]),
		"velocity": torch.stack([item[1]["velocity"] for item in batch]),
		"local_goal": torch.stack([item[1]["local_goal"] for item in batch]),
		"goal": torch.stack([item[1]["goal"] for item in batch]),
	}

	return images, states


def create_dataloader(
	db_path: str,
	batch_size: int = 32,
	shuffle: bool = True,
	num_workers: int = 4,
	transform=None,
) -> DataLoader:
	"""
	Create a DataLoader for the drifter dataset.

	Args:
	    db_path: Path to SQLite database
	    batch_size: Batch size for the dataloader
	    shuffle: Whether to shuffle the data
	    num_workers: Number of worker processes for data loading
	    transform: Optional transform to apply to images

	Returns:
	    DataLoader instance
	"""
	dataset = DrifterDataset(db_path, transform=transform)

	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		collate_fn=collate_fn,
		pin_memory=True,  # Speeds up transfer to GPU
	)

	return dataloader


# Example usage
if __name__ == "__main__":
	# Create dataloader
	db_path = "drifter_data.db"
	dataloader = create_dataloader(
		db_path,
		batch_size=32,
		shuffle=True,
		num_workers=0,  # Set to 0 for debugging, increase for production
	)

	# Test the dataloader
	print(f"Dataset size: {len(dataloader.dataset)}")
	print(f"Number of batches: {len(dataloader)}")

	# Get a single batch
	for images, states in dataloader:
		print(f"\nBatch shapes:")
		print(f"  Images: {images.shape}")
		print(f"  Position: {states['position'].shape}")
		print(f"  Orientation: {states['orientation'].shape}")
		print(f"  Velocity: {states['velocity'].shape}")
		print(f"  Local goal: {states['local_goal'].shape}")
		print(f"  Goal: {states['goal'].shape}")

		print(f"\nSample values from first item in batch:")
		print(f"  Image range: [{images[0].min():.3f}, {images[0].max():.3f}]")
		print(f"  Position: {states['position'][0]}")
		print(f"  Orientation: {states['orientation'][0]}")
		print(f"  Velocity: {states['velocity'][0]}")

		break  # Only show first batch
