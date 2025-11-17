from tqdm import tqdm
import gzip
import sqlite3
import numpy as np
from drifter_env import DrifterEnv
from exploration_policy import ExplorationPolicy
from uuid import uuid4


def compress_image(image):
	"""Compress image array using gzip."""
	# Convert image to bytes if it's a numpy array
	if isinstance(image, np.ndarray):
		image_bytes = image.tobytes()
	else:
		image_bytes = bytes(image)

		# Compress with gzip
	compressed = gzip.compress(image_bytes)
	return compressed


def init_database(db_path):
	"""Initialize SQLite database with required schema."""
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	# Create table for transitions
	cursor.execute("""
		CREATE TABLE IF NOT EXISTS transitions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			episode text,
			step INTEGER,
			position_x REAL,
			position_y REAL,
			position_z REAL,
			local_goal_x REAL,
			local_goal_y REAL,
			local_goal_z REAL,
			velocity_x REAL,
			velocity_y REAL,
			velocity_z REAL,
			is_flipped REAL,
			goal_x REAL,
			goal_y REAL,
			goal_z REAL,
			orientation_0 REAL,
			orientation_1 REAL,
			orientation_2 REAL,
			orientation_3 REAL,
			action_0 REAL,
			action_1 REAL,
			next_position_x REAL,
			next_position_y REAL,
			next_position_z REAL,
			next_local_goal_x REAL,
			next_local_goal_y REAL,
			next_local_goal_z REAL,
			next_velocity_x REAL,
			next_velocity_y REAL,
			next_velocity_z REAL,
			next_is_flipped REAL,
			next_goal_x REAL,
			next_goal_y REAL,
			next_goal_z REAL,
			next_orientation_0 REAL,
			next_orientation_1 REAL,
			next_orientation_2 REAL,
			next_orientation_3 REAL,
			reward REAL,
			done INTEGER,
			truncated INTEGER,
			camera_image BLOB,
			camera_shape_0 INTEGER,
			camera_shape_1 INTEGER,
			camera_shape_2 INTEGER,
			camera_dtype TEXT
		)
	""")

	conn.commit()
	return conn


def parse_state_dict(state_dict):
	"""Parse state dictionary into individual components."""
	state = state_dict["state"]
	camera = state_dict["camera"]

	# Based on _get_obs structure:
	# pos (3), local_goal_pos (3), vel (3), is_flipped (1), goal_pos (3), orn (4)
	idx = 0
	pos = state[idx : idx + 3]
	idx += 3
	local_goal = state[idx : idx + 3]
	idx += 3
	vel = state[idx : idx + 3]
	idx += 3
	is_flipped = state[idx]
	idx += 1
	goal = state[idx : idx + 3]
	idx += 3
	orn = state[idx : idx + 4]

	return {
		"position": pos,
		"local_goal": local_goal,
		"velocity": vel,
		"is_flipped": is_flipped,
		"goal": goal,
		"orientation": orn,
		"camera": camera,
	}


def add_batch_to_database(conn, batch, episode):
	"""Add a batch of transitions to the database."""
	cursor = conn.cursor()

	for step, trans in enumerate(batch):
		state_parsed = parse_state_dict(trans["state"])
		next_state_parsed = parse_state_dict(trans["next_state"])
		action = trans["action"]

		# Compress camera image
		camera = state_parsed["camera"]
		compressed_img = compress_image(camera)

		# Prepare data tuple
		data = (
			episode,
			step,
			# Current state
			float(state_parsed["position"][0]),
			float(state_parsed["position"][1]),
			float(state_parsed["position"][2]),
			float(state_parsed["local_goal"][0]),
			float(state_parsed["local_goal"][1]),
			float(state_parsed["local_goal"][2]),
			float(state_parsed["velocity"][0]),
			float(state_parsed["velocity"][1]),
			float(state_parsed["velocity"][2]),
			float(state_parsed["is_flipped"]),
			float(state_parsed["goal"][0]),
			float(state_parsed["goal"][1]),
			float(state_parsed["goal"][2]),
			float(state_parsed["orientation"][0]),
			float(state_parsed["orientation"][1]),
			float(state_parsed["orientation"][2]),
			float(state_parsed["orientation"][3]),
			# Action
			float(action[0]),
			float(action[1]),
			# Next state
			float(next_state_parsed["position"][0]),
			float(next_state_parsed["position"][1]),
			float(next_state_parsed["position"][2]),
			float(next_state_parsed["local_goal"][0]),
			float(next_state_parsed["local_goal"][1]),
			float(next_state_parsed["local_goal"][2]),
			float(next_state_parsed["velocity"][0]),
			float(next_state_parsed["velocity"][1]),
			float(next_state_parsed["velocity"][2]),
			float(next_state_parsed["is_flipped"]),
			float(next_state_parsed["goal"][0]),
			float(next_state_parsed["goal"][1]),
			float(next_state_parsed["goal"][2]),
			float(next_state_parsed["orientation"][0]),
			float(next_state_parsed["orientation"][1]),
			float(next_state_parsed["orientation"][2]),
			float(next_state_parsed["orientation"][3]),
			# Metadata
			float(trans["reward"]),
			int(trans["done"]),
			int(trans["truncated"]),
			# Camera
			compressed_img,
			int(camera.shape[0]) if hasattr(camera, "shape") else 0,
			int(camera.shape[1])
			if hasattr(camera, "shape") and len(camera.shape) > 1
			else 0,
			int(camera.shape[2])
			if hasattr(camera, "shape") and len(camera.shape) > 2
			else 0,
			str(camera.dtype) if hasattr(camera, "dtype") else "uint8",
		)

		cursor.execute(
			"""
			INSERT INTO transitions VALUES (
				NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
				?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
				?, ?, ?, ?, ?, ?, ?
			)
		""",
			data,
		)

	conn.commit()


def main():
	db_path = "drifter_data.db"
	batch_size = 100  # Batch transitions before writing to DB

	# Initialize database
	conn = init_database(db_path)

	env = DrifterEnv(gui=True)
	expl_policy = ExplorationPolicy(
		env.action_space,
		maneuver_duration=15,
		aggressive_prob=0.35,
		velocity_threshold=0.3,
	)

	s, _ = env.reset()
	batch = []
	episode = str(uuid4())

	for i in tqdm(range(10_000)):
		a = expl_policy.get_action(s)
		sp, r, done, trunc, _ = env.step(a)

		# Store transition in batch
		batch.append(
			{
				"state": s,
				"action": a,
				"next_state": sp,
				"reward": r,
				"done": done,
				"truncated": trunc,
			}
		)

		# Write batch to database when full
		if len(batch) >= batch_size:
			add_batch_to_database(conn, batch, episode)
			batch = []

		s = sp

		if done or trunc:
			# Write remaining batch before episode ends
			if batch:
				add_batch_to_database(conn, batch, episode)
				batch = []

			s, _ = env.reset()
			episode = str(uuid4())

	# Write any remaining transitions
	if batch:
		add_batch_to_database(conn, batch, episode)

	conn.close()
	print(f"\nData collection complete. Database saved to {db_path}")


if __name__ == "__main__":
	main()
