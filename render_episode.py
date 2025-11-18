import sqlite3
import gzip
import numpy as np
import cv2
from pathlib import Path
import random


def decompress_image(compressed_data, shape, dtype):
	"""Decompress gzip-compressed image data."""
	decompressed = gzip.decompress(compressed_data)
	dtype_obj = np.dtype(dtype)
	image = np.frombuffer(decompressed, dtype=dtype_obj)
	image = image.reshape(shape)
	return image


def get_random_episode(db_path):
	"""Get a random episode ID from the database."""
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	# Get all unique episodes
	cursor.execute("SELECT DISTINCT episode FROM transitions")
	episodes = [row[0] for row in cursor.fetchall()]

	if not episodes:
		conn.close()
		raise ValueError("No episodes found in database")

	# Select random episode
	episode = random.choice(episodes)
	conn.close()

	return episode


def fetch_episode_data(db_path, episode_id):
	"""Fetch all transitions for a given episode."""
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	cursor.execute(
		"""
        SELECT step, camera_image, camera_shape_0, camera_shape_1, 
               camera_shape_2, camera_dtype, reward, done
        FROM transitions
        WHERE episode = ?
        ORDER BY step
    """,
		(episode_id,),
	)

	rows = cursor.fetchall()
	conn.close()

	return rows


def render_episode_to_video(
	db_path, episode_id=None, output_path="episode_video.mp4", fps=30
):
	"""
	Render an episode from the database to a video file.

	Args:
	    db_path: Path to the SQLite database
	    episode_id: Specific episode ID to render (if None, selects random)
	    output_path: Output video file path
	    fps: Frames per second for the video
	"""
	# Get episode ID
	if episode_id is None:
		episode_id = get_random_episode(db_path)
		print(f"Selected random episode: {episode_id}")
	else:
		print(f"Rendering episode: {episode_id}")

	# Fetch episode data
	episode_data = fetch_episode_data(db_path, episode_id)

	if not episode_data:
		raise ValueError(f"No data found for episode {episode_id}")

	print(f"Found {len(episode_data)} frames in episode")

	# Initialize video writer
	video_writer = None
	total_reward = 0.0

	for (
		step,
		compressed_img,
		shape_0,
		shape_1,
		shape_2,
		dtype,
		reward,
		done,
	) in episode_data:
		# Decompress image
		shape = (
			(shape_0, shape_1, shape_2) if shape_2 > 0 else (shape_0, shape_1)
		)
		image = decompress_image(compressed_img, shape, dtype)

		# Convert to uint8 if necessary
		if image.dtype != np.uint8:
			image = (
				(image * 255).astype(np.uint8)
				if image.max() <= 1.0
				else image.astype(np.uint8)
			)

		# Convert grayscale to BGR if needed
		if len(image.shape) == 2:
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		elif image.shape[2] == 3:
			# Convert RGB to BGR for OpenCV
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# Add text overlay with step and reward info
		total_reward += reward
		text = (
			f"Step: {step} | Reward: {reward:.3f} | Total: {total_reward:.3f}"
		)
		cv2.putText(
			image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
		)

		if done:
			cv2.putText(
				image,
				"EPISODE END",
				(10, 60),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(0, 0, 255),
				2,
			)

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
	print(f"Total episode reward: {total_reward:.3f}")
	print(f"Episode length: {len(episode_data)} steps")


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
		episode_id=None,  # None = random episode
		output_path="episode_video.mp4",
		fps=10,
	)

	print("\nDone!")


if __name__ == "__main__":
	main()
