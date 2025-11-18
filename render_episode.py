import sqlite3
import gzip
import numpy as np
import cv2
from pathlib import Path
import random
from drifter_dataloader_sequential import create_sequence_dataloader as create_dataloader
import matplotlib.pyplot as plt

def decompress_image(compressed_data, shape, dtype):
	"""Decompress gzip-compressed image data."""
	decompressed = gzip.decompress(compressed_data)
	dtype_obj = np.dtype(dtype)
	image = np.frombuffer(decompressed, dtype=dtype_obj)
	image = image.reshape(shape)
	return image

def render_episode_to_video(
	db_path, output_path="episode_video.mp4", fps=10
):
	"""
	Render an episode from the database to a video file.

	Args:
	    db_path: Path to the SQLite database
	    output_path: Output video file path
	    fps: Frames per second for the video
	"""

	dl = create_dataloader(
		db_path=db_path, batch_size=1, shuffle=True
	)

	imgs, states, seq_lens = next(iter(dl))

	# Initialize video writer
	video_writer = None
	total_reward = 0.0

	for image in imgs[0]:
		image = (image.detach().numpy() * 255.).astype(np.uint8)
		image = image.transpose(1, 2, 0)
		# Initialize video writer on first frame
		if video_writer is None:
			height, width = image.shape[:2]
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			video_writer = cv2.VideoWriter(
				output_path, fourcc, fps, (width, height)
			)

		# Write frame
		video_writer.write(image)

	# Release video writer
	if video_writer is not None:
		video_writer.release()

	print(f"Video saved to: {output_path}")


def list_episodes(db_path, limit=10):
	"""List available episodes in the database."""
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	cursor.execute(
		"""
        SELECT episode, COUNT(*) as num_steps, SUM(reward) as total_reward
        FROM transitions
        GROUP BY episode
        ORDER BY MIN(id)
        LIMIT ?
    """,
		(limit,),
	)

	episodes = cursor.fetchall()
	conn.close()

	print(f"\nAvailable episodes (showing first {limit}):")
	print("-" * 80)
	for ep_id, num_steps, total_reward in episodes:
		print(
			f"Episode: {ep_id[:8]}... | Steps: {num_steps} | Total Reward: {total_reward:.3f}"
		)
	print("-" * 80)

	return episodes


def main():
	db_path = "drifter_data.db"

	# Check if database exists
	if not Path(db_path).exists():
		print(f"Error: Database not found at {db_path}")
		return

	# List some episodes
	list_episodes(db_path, limit=5)

	# Render a random episode
	print("\nRendering random episode...")
	render_episode_to_video(
		db_path=db_path,
		output_path="episode_video.mp4",
		fps=10,
	)

	print("\nDone!")


if __name__ == "__main__":
	main()
