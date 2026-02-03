#!/usr/bin/env python3
"""
Pre-compute embeddings for forward dynamics training.

This script loads an inverse dynamics model and processes all sequences in the
database to pre-compute latent embeddings. Results are saved to a new database
that can be used for faster training.
"""

import click
import sqlite3
import torch
import gzip
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List
import pickle
from env_vision_model import EnvModel


def decompress_image(
	blob: bytes, shape: Tuple[int, int, int], dtype: str
) -> np.ndarray:
	"""Decompress a gzipped image blob."""
	decompressed = gzip.decompress(blob)
	dtype_np = np.dtype(dtype)
	img = np.frombuffer(decompressed, dtype=dtype_np)
	return img.reshape(shape)


def orientation_relative(quat, initial_quat):
	"""
	Compute relative orientation between two quaternions.

	Args:
			quat: Current orientation as [w, x, y, z]
			initial_quat: Initial orientation as [w, x, y, z]

	Returns:
			Relative quaternion such that initial orientation becomes [1, 0, 0, 0]
	"""
	# Quaternion inverse: for unit quaternions, it's the conjugate [w, -x, -y, -z]
	w0, x0, y0, z0 = initial_quat
	initial_inv = np.array([w0, -x0, -y0, -z0])

	# Quaternion multiplication: initial_inv * quat
	w1, x1, y1, z1 = initial_inv
	w2, x2, y2, z2 = quat

	return np.array(
		[
			w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
			w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
			w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
			w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
		]
	)


def fetch_sequence(
	conn: sqlite3.Connection,
	table_name: str,
	id_column: str,
	start_id: int,
	seq_len: int,
) -> List[Tuple]:
	"""Fetch a sequence of rows from the database."""
	cursor = conn.cursor()

	columns = [
		"camera_image",
		"camera_shape_0",
		"camera_shape_1",
		"camera_shape_2",
		"camera_dtype",
		"action_0",
		"action_1",
		# State variables
		"position_x",
		"position_y",
		"position_z",
		"velocity_x",
		"velocity_y",
		"velocity_z",
		"is_flipped",
		"orientation_0",
		"orientation_1",
		"orientation_2",
		"orientation_3",
	]
	columns_str = ", ".join(columns)

	query = f"""
        SELECT {columns_str}
        FROM {table_name}
        WHERE {id_column} >= ? AND {id_column} < ?
        ORDER BY {id_column}
    """
	cursor.execute(query, (start_id, start_id + seq_len))

	return cursor.fetchall()


def parse_state(row: Tuple, initial_position: np.ndarray, initial_orientation) -> np.ndarray:
	"""
	Parse state variables from a database row.

	Returns:
	    state: [11] array with [pos(3), vel(3), is_flipped(1), orient(4)]
	"""
	(
		_camera_blob,
		_shape_0,
		_shape_1,
		_shape_2,
		_dtype_str,
		_action_0,
		_action_1,
		pos_x,
		pos_y,
		pos_z,
		vel_x,
		vel_y,
		vel_z,
		is_flipped,
		orient_0,
		orient_1,
		orient_2,
		orient_3,
	) = row

	# Make position relative to start
	position = np.array([pos_x, pos_y, pos_z]) - initial_position
	velocity = np.array([vel_x, vel_y, vel_z])
	org_orientation = np.array([orient_0, orient_1, orient_2, orient_3])
	orientation = orientation_relative(org_orientation, initial_orientation)

	# Concatenate: [pos(3), local_goal(3), vel(3), is_flipped(1), orient(4)]
	state = np.concatenate(
		[position, velocity, [is_flipped], orientation]
	)

	return state


def build_episode_index(
	conn: sqlite3.Connection,
	table_name: str,
	episode_column: str,
	id_column: str,
	min_seq_len: int,
) -> List[dict]:
	"""Build an index of all episodes and their row ranges."""
	cursor = conn.cursor()

	query = f"""
        SELECT 
            {episode_column},
            MIN({id_column}) as start_id,
            MAX({id_column}) as end_id,
            COUNT(*) as length
        FROM {table_name}
        GROUP BY {episode_column}
        ORDER BY MIN({id_column})
    """
	cursor.execute(query)

	episodes = []
	for row in cursor.fetchall():
		episode_id, start_id, end_id, length = row
		if length >= min_seq_len:
			episodes.append(
				{
					"episode_id": episode_id,
					"start_id": start_id,
					"end_id": end_id,
					"length": length,
				}
			)

	return episodes


def create_embedding_db(output_path: str):
	"""Create the output database for embeddings."""
	conn = sqlite3.connect(output_path)
	cursor = conn.cursor()

	cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT,
            start_id INTEGER,
            seq_len INTEGER,
            h_t BLOB,
            a_t BLOB,
            h_t_next BLOB,
            state_t BLOB,
            state_t_next BLOB,
            num_transitions INTEGER,
            UNIQUE(episode_id, start_id, seq_len)
        )
    """)

	cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_episode_start 
        ON embeddings(episode_id, start_id)
    """)

	conn.commit()
	return conn


def process_sequence(
	id_model, device: str, lstm_context: int, rows: List[Tuple]
) -> Tuple:
	"""Process a single sequence and return embeddings."""

	# Get initial position
	initial_position = np.array([rows[0][7], rows[0][8], rows[0][9]])
	initial_orientation = np.array(
		[
			rows[0][13],
			rows[0][14],
			rows[0][15],
			rows[0][16],
		]
	)

	# Parse images, actions, and states
	images = []
	actions = []
	states = []

	for row in rows:
		# Decompress image
		camera_blob = row[0]
		shape = (row[1], row[2], row[3])
		dtype_str = row[4]
		img = decompress_image(camera_blob, shape, dtype_str)
		images.append(img)

		# Store action
		action_0, action_1 = row[5], row[6]
		actions.append([action_0, action_1])

		# Parse state
		state = parse_state(row, initial_position, initial_orientation)
		states.append(state)

	# Convert to tensors
	images = torch.from_numpy(np.stack(images)).float().permute(0, 3, 1, 2)
	actions = torch.tensor(actions, dtype=torch.float32)
	states = torch.tensor(np.stack(states), dtype=torch.float32)

	seq_len = len(rows)
	num_transitions = seq_len - lstm_context

	h_t_list = []
	a_t_list = []
	h_t_next_list = []
	state_t_list = []
	state_t_next_list = []

	with torch.no_grad():
		for i in range(num_transitions):
			# Get h_t from context window
			imgs_t = images[i : i + lstm_context].unsqueeze(0).to(device)
			seq_lens_t = torch.tensor([lstm_context], dtype=torch.long)
			h_t = id_model._get_hidden(imgs_t, seq_lens_t)[:, -1, :]

			# Get h_{t+1} from context+1 window
			imgs_t_next = (
				images[i : i + lstm_context + 1].unsqueeze(0).to(device)
			)
			seq_lens_t_next = torch.tensor([lstm_context + 1], dtype=torch.long)
			h_t_next = id_model._get_hidden(imgs_t_next, seq_lens_t_next)[
				:, -1, :
			]

			# Get action and states
			a_t = actions[i + lstm_context - 1]
			state_t = states[i + lstm_context - 1]
			state_t_next = states[i + lstm_context]

			h_t_list.append(h_t.cpu().squeeze(0))
			h_t_next_list.append(h_t_next.cpu().squeeze(0))
			a_t_list.append(a_t)
			state_t_list.append(state_t)
			state_t_next_list.append(state_t_next)

	# Stack into tensors
	h_t_batch = torch.stack(h_t_list)
	a_t_batch = torch.stack(a_t_list)
	h_t_next_batch = torch.stack(h_t_next_list)
	state_t_batch = torch.stack(state_t_list)
	state_t_next_batch = torch.stack(state_t_next_list)

	return (
		h_t_batch,
		a_t_batch,
		h_t_next_batch,
		state_t_batch,
		state_t_next_batch,
		num_transitions,
	)


def save_embedding(
	conn: sqlite3.Connection,
	episode_id: str,
	start_id: int,
	seq_len: int,
	embedding_data: Tuple,
):
	"""Save embedding to database."""
	cursor = conn.cursor()

	h_t, a_t, h_t_next, state_t, state_t_next, num_transitions = embedding_data

	# Serialize tensors
	h_t_blob = pickle.dumps(h_t.numpy())
	a_t_blob = pickle.dumps(a_t.numpy())
	h_t_next_blob = pickle.dumps(h_t_next.numpy())
	state_t_blob = pickle.dumps(state_t.numpy())
	state_t_next_blob = pickle.dumps(state_t_next.numpy())

	cursor.execute(
		"""
        INSERT OR REPLACE INTO embeddings 
        (episode_id, start_id, seq_len, h_t, a_t, h_t_next, state_t, state_t_next, num_transitions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
		(
			episode_id,
			start_id,
			seq_len,
			h_t_blob,
			a_t_blob,
			h_t_next_blob,
			state_t_blob,
			state_t_next_blob,
			num_transitions,
		),
	)

	conn.commit()


@click.command()
@click.option(
	"--input-db",
	required=True,
	type=click.Path(exists=True),
	help="Path to input SQLite database with transitions",
)
@click.option(
	"--output-db",
	required=True,
	type=click.Path(),
	help="Path to output SQLite database for embeddings",
)
@click.option(
	"--id-model",
	required=True,
	type=click.Path(exists=True),
	help="Path to pre-trained inverse dynamics model (.pth file)",
)
@click.option(
	"--sequence-length",
	default=3,
	type=int,
	help="LSTM context length (default: 3)",
)
@click.option(
	"--max-seq-len",
	default=50,
	type=int,
	help="Maximum sequence length to process (default: 50)",
)
@click.option(
	"--stride",
	default=10,
	type=int,
	help="Stride between sequence samples from same episode (default: 10)",
)
@click.option(
	"--device",
	default="cuda",
	type=click.Choice(["cuda", "cpu"]),
	help="Device to run model on (default: cuda)",
)
@click.option(
	"--table-name",
	default="transitions",
	help="Name of the transitions table (default: transitions)",
)
@click.option(
	"--episode-column",
	default="episode",
	help="Name of the episode column (default: episode)",
)
@click.option(
	"--id-column", default="id", help="Name of the ID column (default: id)"
)
def main(
	input_db,
	output_db,
	id_model,
	sequence_length,
	max_seq_len,
	stride,
	device,
	table_name,
	episode_column,
	id_column,
):
	"""
	Pre-compute embeddings for forward dynamics training.

	This script processes all episodes in the input database, computes latent
	embeddings using the inverse dynamics model, and saves them to an output
	database for faster training.
	"""

	click.echo(f"Loading inverse dynamics model from {id_model}...")
	model_sd = torch.load(id_model, map_location=device)["model_state_dict"]
	model = EnvModel()
	model.load_state_dict(model_sd)
	model.eval()
	model.to(device)

	click.echo(f"Connecting to input database: {input_db}")
	input_conn = sqlite3.connect(input_db)

	click.echo(f"Creating output database: {output_db}")
	output_conn = create_embedding_db(output_db)

	min_seq_len = sequence_length + 1
	click.echo(f"Building episode index (min_seq_len={min_seq_len})...")
	episodes = build_episode_index(
		input_conn, table_name, episode_column, id_column, min_seq_len
	)

	click.echo(f"Found {len(episodes)} valid episodes")

	total_sequences = 0
	for episode in episodes:
		num_sequences = (episode["length"] - max_seq_len) // stride + 1
		total_sequences += max(1, num_sequences)

	click.echo(f"Processing {total_sequences} sequences...")

	with tqdm(total=total_sequences, desc="Computing embeddings") as pbar:
		for episode in episodes:
			episode_id = episode["episode_id"]
			episode_length = episode["length"]

			# Sample sequences with stride
			start_offset = 0
			while start_offset + min_seq_len <= episode_length:
				seq_len = min(max_seq_len, episode_length - start_offset)
				start_id = episode["start_id"] + start_offset

				# Fetch and process sequence
				rows = fetch_sequence(
					input_conn, table_name, id_column, start_id, seq_len
				)

				embedding_data = process_sequence(
					model, device, sequence_length, rows
				)

				# Save to database
				save_embedding(
					output_conn, episode_id, start_id, seq_len, embedding_data
				)

				pbar.update(1)

				# Move to next sequence
				start_offset += stride

				# If we can't fit another full sequence, break
				if start_offset + min_seq_len > episode_length:
					break

	input_conn.close()
	output_conn.close()

	click.echo(f"✓ Done! Embeddings saved to {output_db}")
	click.echo(f"  Total sequences processed: {total_sequences}")


if __name__ == "__main__":
	main()
