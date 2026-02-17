import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from pathlib import Path
from datetime import datetime
from env_vision_model import EnvModel
import matplotlib.pyplot as plt
import numpy as np

from precomputed_dataloader import create_precomputed_dataloaders


def state_tensor_to_dict(state_tensor):
	"""
	Convert flat state tensor to dict format for visualization.

	State tensor has 11 dims: pos(3), vel(3), flipped(1), orient(4)
	Note: local_goal has already been removed from the dataloader output

	Args:
		state_tensor: [batch_size, seq_len, 11] or [seq_len, 11]

	Returns:
		dict with keys: position, velocity, orientation (quaternion in [w,x,y,z] format)
	"""
	# Handle both batched and unbatched inputs
	if state_tensor.dim() == 2:
		# [seq_len, 11]
		return {
			"position": state_tensor[..., :3],  # [seq_len, 3]
			"velocity": state_tensor[..., 3:6],  # [seq_len, 3]
			"orientation": state_tensor[
				..., 7:11
			],  # [seq_len, 4] (skip flipped at 6:7)
		}
	else:
		# [batch_size, seq_len, 11]
		return {
			"position": state_tensor[..., :3],  # [batch_size, seq_len, 3]
			"velocity": state_tensor[..., 3:6],  # [batch_size, seq_len, 3]
			"orientation": state_tensor[..., 7:11],  # [batch_size, seq_len, 4]
		}


def quaternion_to_euler(q):
	"""Convert quaternion [w, x, y, z] to euler angles [roll, pitch, yaw]"""
	w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

	# Roll (x-axis rotation)
	sinr_cosp = 2 * (w * x + y * z)
	cosr_cosp = 1 - 2 * (x * x + y * y)
	roll = np.arctan2(sinr_cosp, cosr_cosp)

	# Pitch (y-axis rotation)
	sinp = 2 * (w * y - z * x)
	pitch = np.arcsin(np.clip(sinp, -1, 1))

	# Yaw (z-axis rotation)
	siny_cosp = 2 * (w * z + x * y)
	cosy_cosp = 1 - 2 * (y * y + z * z)
	yaw = np.arctan2(siny_cosp, cosy_cosp)

	return roll, pitch, yaw


def visualize_trajectory(true_states, pred_states, seq_len, writer, step):
	"""
	Visualize predicted vs true trajectory for a single sequence.

	Args:
		true_states: Dict of true state tensors
		pred_states: Dict of predicted state tensors
		seq_len: Actual sequence length (for masking padding)
		writer: TensorBoard SummaryWriter
		step: Global step for logging
	"""
	# Move to CPU and convert to numpy
	true_pos = true_states["position"][:seq_len].cpu().numpy()  # [seq_len, 3]
	pred_pos = pred_states["position"][:seq_len].detach().cpu().numpy()

	true_orient = (
		true_states["orientation"][:seq_len].cpu().numpy()
	)  # [seq_len, 4]
	pred_orient = pred_states["orientation"][:seq_len].detach().cpu().numpy()

	true_vel = true_states["velocity"][:seq_len].cpu().numpy()  # [seq_len, 3]
	pred_vel = pred_states["velocity"][:seq_len].detach().cpu().numpy()

	# Convert quaternions to yaw for 2D visualization
	_, _, true_yaw = quaternion_to_euler(true_orient)
	_, _, pred_yaw = quaternion_to_euler(pred_orient)

	# Create figure with subplots
	fig = plt.figure(figsize=(16, 10))

	# 1. Top-down trajectory view (position x, y)
	ax1 = plt.subplot(2, 3, 1)
	ax1.plot(true_pos[:, 0], true_pos[:, 1], "b-", label="True", linewidth=2)
	ax1.plot(
		pred_pos[:, 0], pred_pos[:, 1], "r--", label="Predicted", linewidth=2
	)
	ax1.scatter(
		true_pos[0, 0],
		true_pos[0, 1],
		c="green",
		s=100,
		marker="o",
		label="Start",
	)
	ax1.scatter(
		true_pos[-1, 0],
		true_pos[-1, 1],
		c="black",
		s=100,
		marker="x",
		label="End",
	)
	ax1.set_xlabel("X Position (m)")
	ax1.set_ylabel("Y Position (m)")
	ax1.set_title("Top-Down Trajectory")
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	ax1.axis("equal")

	# 2. Position components over time
	ax2 = plt.subplot(2, 3, 2)
	time = np.arange(seq_len)
	ax2.plot(time, true_pos[:, 0], "b-", label="True X", alpha=0.7)
	ax2.plot(time, pred_pos[:, 0], "b--", label="Pred X", alpha=0.7)
	ax2.plot(time, true_pos[:, 1], "r-", label="True Y", alpha=0.7)
	ax2.plot(time, pred_pos[:, 1], "r--", label="Pred Y", alpha=0.7)
	ax2.plot(time, true_pos[:, 2], "g-", label="True Z", alpha=0.7)
	ax2.plot(time, pred_pos[:, 2], "g--", label="Pred Z", alpha=0.7)
	ax2.set_xlabel("Timestep")
	ax2.set_ylabel("Position (m)")
	ax2.set_title("Position Components")
	ax2.legend(fontsize=8)
	ax2.grid(True, alpha=0.3)

	# 3. Yaw angle over time
	ax3 = plt.subplot(2, 3, 3)
	ax3.plot(time, np.degrees(true_yaw), "b-", label="True Yaw", linewidth=2)
	ax3.plot(time, np.degrees(pred_yaw), "r--", label="Pred Yaw", linewidth=2)
	ax3.set_xlabel("Timestep")
	ax3.set_ylabel("Yaw (degrees)")
	ax3.set_title("Heading Angle")
	ax3.legend()
	ax3.grid(True, alpha=0.3)

	# 4. Velocity magnitude
	ax4 = plt.subplot(2, 3, 4)
	true_speed = np.linalg.norm(true_vel, axis=1)
	pred_speed = np.linalg.norm(pred_vel, axis=1)
	ax4.plot(time, true_speed, "b-", label="True Speed", linewidth=2)
	ax4.plot(time, pred_speed, "r--", label="Pred Speed", linewidth=2)
	ax4.set_xlabel("Timestep")
	ax4.set_ylabel("Speed (m/s)")
	ax4.set_title("Velocity Magnitude")
	ax4.legend()
	ax4.grid(True, alpha=0.3)

	# 5. Velocity components
	ax5 = plt.subplot(2, 3, 5)
	ax5.plot(time, true_vel[:, 0], "b-", label="True Vx", alpha=0.7)
	ax5.plot(time, pred_vel[:, 0], "b--", label="Pred Vx", alpha=0.7)
	ax5.plot(time, true_vel[:, 1], "r-", label="True Vy", alpha=0.7)
	ax5.plot(time, pred_vel[:, 1], "r--", label="Pred Vy", alpha=0.7)
	ax5.plot(time, true_vel[:, 2], "g-", label="True Vz", alpha=0.7)
	ax5.plot(time, pred_vel[:, 2], "g--", label="Pred Vz", alpha=0.7)
	ax5.set_xlabel("Timestep")
	ax5.set_ylabel("Velocity (m/s)")
	ax5.set_title("Velocity Components (Body Frame)")
	ax5.legend(fontsize=8)
	ax5.grid(True, alpha=0.3)

	# 6. Trajectory with velocity vectors
	ax6 = plt.subplot(2, 3, 6)
	# Plot every Nth point to avoid clutter
	skip = max(1, seq_len // 10)
	for i in range(0, seq_len, skip):
		# True trajectory arrows
		ax6.arrow(
			true_pos[i, 0],
			true_pos[i, 1],
			true_vel[i, 0] * 0.1,
			true_vel[i, 1] * 0.1,
			head_width=0.05,
			head_length=0.03,
			fc="blue",
			ec="blue",
			alpha=0.5,
		)
		# Predicted trajectory arrows
		ax6.arrow(
			pred_pos[i, 0],
			pred_pos[i, 1],
			pred_vel[i, 0] * 0.1,
			pred_vel[i, 1] * 0.1,
			head_width=0.05,
			head_length=0.03,
			fc="red",
			ec="red",
			alpha=0.5,
		)

	ax6.plot(true_pos[:, 0], true_pos[:, 1], "b-", alpha=0.3, linewidth=1)
	ax6.plot(pred_pos[:, 0], pred_pos[:, 1], "r--", alpha=0.3, linewidth=1)
	ax6.set_xlabel("X Position (m)")
	ax6.set_ylabel("Y Position (m)")
	ax6.set_title("Trajectory with Velocity Vectors")
	ax6.grid(True, alpha=0.3)
	ax6.axis("equal")

	plt.tight_layout()

	# Save to tensorboard
	writer.add_figure("validation/trajectory", fig, step)
	plt.close(fig)


def fit_batch(model, optimizer, criterion, batch, device):
	"""Train on a single batch."""
	h_t, a_t, h_t_next, state_t, state_t_next, seq_lengths = batch

	# Move to device
	h_t = h_t.to(device)
	a_t = a_t.to(device)
	h_t_next = h_t_next.to(device)
	state_t = state_t.to(device)
	state_t_next = state_t_next.to(device)

	optimizer.zero_grad()

	# Forward pass
	h_t_next_pred = model.forward_dynamics_from_hidden(h_t, a_t, 0)
	state_pred, outputs_t = model.decode_state(h_t, 0)
	state_next_pred, outputs_t_next = model.decode_state(h_t_next, 0)

	# Compute losses
	fd_loss = criterion(h_t_next_pred, h_t_next)
	decoder_loss_t = criterion(state_pred, state_t)
	decoder_loss_t_next = criterion(state_next_pred, state_t_next)
	decoder_loss = decoder_loss_t + decoder_loss_t_next

	# Compute per-component decoder losses (state has 14 dims: pos(3), vel(3), flipped(1), orient(4))
	# Remove local_goal from comparison

	# Component losses (without local_goal)
	pos_loss_t = criterion(state_pred[..., :3], state_t[..., :3])
	vel_loss_t = criterion(state_pred[..., 3:6], state_t[..., 3:6])
	flipped_loss_t = criterion(state_pred[..., 6:7], state_t[..., 6:7])
	orient_loss_t = criterion(state_pred[..., 7:], state_t[..., 7:])

	pos_loss_t_next = criterion(state_next_pred[..., :3], state_t_next[..., :3])
	vel_loss_t_next = criterion(
		state_next_pred[..., 3:6], state_t_next[..., 3:6]
	)
	flipped_loss_t_next = criterion(
		state_next_pred[..., 6:7], state_t_next[..., 6:7]
	)
	orient_loss_t_next = criterion(
		state_next_pred[..., 7:], state_t_next[..., 7:]
	)

	# Total loss
	loss = fd_loss + decoder_loss

	loss.backward()

	# Compute gradient norms before clipping
	total_norm = 0.0
	for p in model.parameters():
		if p.grad is not None:
			param_norm = p.grad.data.norm(2)
			total_norm += param_norm.item() ** 2
	total_norm = total_norm**0.5

	optimizer.step()

	return {
		"total_loss": loss.item(),
		"fd_loss": fd_loss.item(),
		"decoder_loss": decoder_loss.item(),
		"decoder_loss_t": decoder_loss_t.item(),
		"decoder_loss_t_next": decoder_loss_t_next.item(),
		"pos_loss_t": pos_loss_t.item(),
		"vel_loss_t": vel_loss_t.item(),
		"flipped_loss_t": flipped_loss_t.item(),
		"orient_loss_t": orient_loss_t.item(),
		"pos_loss_t_next": pos_loss_t_next.item(),
		"vel_loss_t_next": vel_loss_t_next.item(),
		"flipped_loss_t_next": flipped_loss_t_next.item(),
		"orient_loss_t_next": orient_loss_t_next.item(),
		"grad_norm": total_norm,
		# Return predictions for logging
		"h_t_next_pred": h_t_next_pred.detach(),
		"h_t_next": h_t_next.detach(),
		"state_pred": state_pred.detach(),
		"state_t": state_t.detach(),
	}


def log_detailed_metrics(writer, metrics, prefix, step):
	"""Log detailed metrics to tensorboard."""
	# Scalars
	for key in [
		"total_loss",
		"fd_loss",
		"decoder_loss",
		"decoder_loss_t",
		"decoder_loss_t_next",
		"grad_norm",
	]:
		if key in metrics:
			writer.add_scalar(f"{prefix}/{key}", metrics[key], step)

	# Component losses
	for component in ["pos", "vel", "flipped", "orient"]:
		for time in ["t", "t_next"]:
			key = f"{component}_loss_{time}"
			if key in metrics:
				writer.add_scalar(
					f"{prefix}/components/{key}", metrics[key], step
				)

	# Prediction statistics (if available)
	if "h_t_next_pred" in metrics and "h_t_next" in metrics:
		h_error = (metrics["h_t_next_pred"] - metrics["h_t_next"]).abs()
		writer.add_scalar(
			f"{prefix}/hidden_error_mean", h_error.mean().item(), step
		)
		writer.add_scalar(
			f"{prefix}/hidden_error_max", h_error.max().item(), step
		)
		writer.add_scalar(
			f"{prefix}/hidden_error_std", h_error.std().item(), step
		)

	if "state_pred" in metrics and "state_t" in metrics:
		s_error = (metrics["state_pred"] - metrics["state_t"]).abs()
		writer.add_scalar(
			f"{prefix}/state_error_mean", s_error.mean().item(), step
		)
		writer.add_scalar(
			f"{prefix}/state_error_max", s_error.max().item(), step
		)

		# Per-dimension state errors (11 dims without local_goal)
		for i, name in enumerate(
			[
				"pos_x",
				"pos_y",
				"pos_z",
				"vel_x",
				"vel_y",
				"vel_z",
				"flipped",
				"q_w",
				"q_x",
				"q_y",
				"q_z",
			]
		):
			if i < s_error.shape[-1]:
				writer.add_scalar(
					f"{prefix}/state_errors/{name}",
					s_error[..., i].mean().item(),
					step,
				)


def train_epoch(
	model, dataloader, optimizer, criterion, device, epoch, writer, global_step
):
	"""Train for one epoch."""
	model.train()

	total_losses = {
		"total_loss": 0.0,
		"fd_loss": 0.0,
		"decoder_loss": 0.0,
		"decoder_loss_t": 0.0,
		"decoder_loss_t_next": 0.0,
		"grad_norm": 0.0,
	}

	# Add component losses
	for component in ["pos", "vel", "flipped", "orient"]:
		for time in ["t", "t_next"]:
			total_losses[f"{component}_loss_{time}"] = 0.0

	total_samples = 0

	pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

	for batch_idx, batch in enumerate(pbar):
		losses = fit_batch(model, optimizer, criterion, batch, device)

		batch_size = batch[0].size(0)

		# Track statistics
		for key in total_losses:
			if key in losses:
				total_losses[key] += losses[key] * batch_size
		total_samples += batch_size

		# Detailed logging every batch
		log_detailed_metrics(writer, losses, "train_batch", global_step)

		global_step += 1

		# Update progress bar
		pbar.set_postfix(
			{
				"loss": f"{losses['total_loss']:.4f}",
				"fd": f"{losses['fd_loss']:.4f}",
				"dec": f"{losses['decoder_loss']:.4f}",
				"grad": f"{losses['grad_norm']:.2f}",
			}
		)

	# Compute epoch averages
	avg_losses = {key: val / total_samples for key, val in total_losses.items()}

	# Log epoch averages
	for key, value in avg_losses.items():
		writer.add_scalar(f"train_epoch/{key}", value, epoch)

	# Log learning rate
	writer.add_scalar(
		"train/learning_rate", optimizer.param_groups[0]["lr"], epoch
	)

	return avg_losses, global_step


def validate_epoch(
	model, dataloader, criterion, device, epoch, writer, reference_sample
):
	"""Validate for one epoch."""
	model.eval()

	total_losses = {
		"total_loss": 0.0,
		"fd_loss": 0.0,
		"decoder_loss": 0.0,
		"decoder_loss_t": 0.0,
		"decoder_loss_t_next": 0.0,
	}

	# Add component losses
	for component in ["pos", "vel", "flipped", "orient"]:
		for time in ["t", "t_next"]:
			total_losses[f"{component}_loss_{time}"] = 0.0

	total_samples = 0

	# Collect predictions for visualization
	all_h_errors = []
	all_state_errors = []

	with torch.no_grad():
		pbar = tqdm(dataloader, desc=f"Validation {epoch}")

		for batch in pbar:
			h_t, a_t, h_t_next, state_t, state_t_next, seq_lengths = batch

			# Move to device
			h_t = h_t.to(device)
			a_t = a_t.to(device)
			h_t_next = h_t_next.to(device)
			state_t = state_t.to(device)
			state_t_next = state_t_next.to(device)

			# Forward pass
			h_t_next_pred = model.forward_dynamics_from_hidden(h_t, a_t, 0)
			state_pred, _ = model.decode_state(h_t, 0)
			state_next_pred, _ = model.decode_state(h_t_next, 0)

			# Compute losses
			fd_loss = criterion(h_t_next_pred, h_t_next)
			decoder_loss_t = criterion(state_pred, state_t)
			decoder_loss_t_next = criterion(state_next_pred, state_t_next)
			decoder_loss = decoder_loss_t + decoder_loss_t_next
			loss = fd_loss + decoder_loss

			# Remove local_goal for component analysis
			state_t = torch.cat([state_t[..., :3], state_t[..., 6:]], dim=-1)
			state_pred = torch.cat(
				[state_pred[..., :3], state_pred[..., 6:]], dim=-1
			)
			state_t_next = torch.cat(
				[state_t_next[..., :3], state_t_next[..., 6:]], dim=-1
			)
			state_next_pred = torch.cat(
				[state_next_pred[..., :3], state_next_pred[..., 6:]], dim=-1
			)

			# Component losses
			pos_loss_t = criterion(state_pred[..., :3], state_t[..., :3])
			vel_loss_t = criterion(state_pred[..., 3:6], state_t[..., 3:6])
			flipped_loss_t = criterion(state_pred[..., 6:7], state_t[..., 6:7])
			orient_loss_t = criterion(state_pred[..., 7:], state_t[..., 7:])

			pos_loss_t_next = criterion(
				state_next_pred[..., :3], state_t_next[..., :3]
			)
			vel_loss_t_next = criterion(
				state_next_pred[..., 3:6],
				state_t_next[..., 3:6],
			)
			flipped_loss_t_next = criterion(
				state_next_pred[..., 6:7],
				state_t_next[..., 6:7],
			)
			orient_loss_t_next = criterion(
				state_next_pred[..., 7:], state_t_next[..., 7:]
			)

			batch_size = h_t.size(0)

			# Track statistics
			total_losses["total_loss"] += loss.item() * batch_size
			total_losses["fd_loss"] += fd_loss.item() * batch_size
			total_losses["decoder_loss"] += decoder_loss.item() * batch_size
			total_losses["decoder_loss_t"] += decoder_loss_t.item() * batch_size
			total_losses["decoder_loss_t_next"] += (
				decoder_loss_t_next.item() * batch_size
			)
			total_losses["pos_loss_t"] += pos_loss_t.item() * batch_size
			total_losses["vel_loss_t"] += vel_loss_t.item() * batch_size
			total_losses["flipped_loss_t"] += flipped_loss_t.item() * batch_size
			total_losses["orient_loss_t"] += orient_loss_t.item() * batch_size
			total_losses["pos_loss_t_next"] += (
				pos_loss_t_next.item() * batch_size
			)
			total_losses["vel_loss_t_next"] += (
				vel_loss_t_next.item() * batch_size
			)
			total_losses["flipped_loss_t_next"] += (
				flipped_loss_t_next.item() * batch_size
			)
			total_losses["orient_loss_t_next"] += (
				orient_loss_t_next.item() * batch_size
			)
			total_samples += batch_size

			# Collect errors for histograms
			all_h_errors.append((h_t_next_pred - h_t_next).abs().cpu())
			all_state_errors.append((state_pred - state_t).abs().cpu())

			pbar.set_postfix({"loss": f"{loss.item():.4f}"})

	# Compute averages
	avg_losses = {key: val / total_samples for key, val in total_losses.items()}

	# Log scalar metrics
	for key, value in avg_losses.items():
		writer.add_scalar(f"val/{key}", value, epoch)

	# Log component losses grouped
	for component in ["pos", "vel", "flipped", "orient"]:
		for time in ["t", "t_next"]:
			key = f"{component}_loss_{time}"
			writer.add_scalar(f"val/components/{key}", avg_losses[key], epoch)

	# Log error histograms
	all_h_errors = torch.cat(all_h_errors, dim=0)
	all_state_errors = torch.cat(all_state_errors, dim=0)

	writer.add_histogram("val/hidden_errors", all_h_errors, epoch)
	writer.add_histogram("val/state_errors", all_state_errors, epoch)

	# Log per-dimension state error histograms
	for i, name in enumerate(
		[
			"pos_x",
			"pos_y",
			"pos_z",
			"vel_x",
			"vel_y",
			"vel_z",
			"flipped",
			"q_w",
			"q_x",
			"q_y",
			"q_z",
		]
	):
		if i < all_state_errors.shape[-1]:
			writer.add_histogram(
				f"val/state_errors/{name}", all_state_errors[..., i], epoch
			)

	# Visualize reference trajectory if provided
	if reference_sample is not None:
		ref_h_t, ref_state_t, ref_seq_len = reference_sample

		# Decode the reference hidden state to get predictions
		with torch.no_grad():
			ref_state_pred, _ = model.decode_state(ref_h_t.to(device), 0)

		# Convert flat tensors to dicts for visualization
		true_states_dict = state_tensor_to_dict(
			ref_state_t[0]
		)  # Remove batch dim
		pred_states_dict = state_tensor_to_dict(
			ref_state_pred[0]
		)  # Remove batch dim

		# Visualize
		visualize_trajectory(
			true_states_dict,
			pred_states_dict,
			ref_seq_len,
			writer,
			epoch,
		)

	return avg_losses


def log_model_weights(model, writer, epoch):
	"""Log model weight statistics to tensorboard."""
	for name, param in model.named_parameters():
		if param.requires_grad:
			writer.add_histogram(f"weights/{name}", param.data, epoch)
			if param.grad is not None:
				writer.add_histogram(f"gradients/{name}", param.grad, epoch)
				writer.add_scalar(
					f"grad_norms/{name}", param.grad.norm(2).item(), epoch
				)


def save_checkpoint(
	model,
	optimizer,
	epoch,
	train_losses,
	val_losses,
	checkpoint_dir,
	is_best=False,
):
	"""Save model checkpoint."""
	checkpoint_dir = Path(checkpoint_dir)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)

	checkpoint = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"train_losses": train_losses,
		"val_losses": val_losses,
	}

	# Save latest checkpoint
	latest_path = checkpoint_dir / "latest.pth"
	torch.save(checkpoint, latest_path)

	# Save epoch checkpoint every 10 epochs
	if epoch % 10 == 0:
		epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pth"
		torch.save(checkpoint, epoch_path)

	# Save best checkpoint
	if is_best:
		best_path = checkpoint_dir / "best.pth"
		torch.save(checkpoint, best_path)
		print(
			f"  💾 Saved best model (val_loss: {val_losses['total_loss']:.4f})"
		)


@click.command()
@click.option(
	"--train-db",
	required=True,
	type=click.Path(exists=True),
	help="Path to training embeddings database",
)
@click.option(
	"--val-db",
	type=click.Path(exists=True),
	help="Path to validation embeddings database (optional)",
)
@click.option(
	"--model",
	required=True,
	type=click.Path(exists=True),
	help="Path to model checkpoint (.pth file)",
)
@click.option(
	"--checkpoint-dir",
	default="checkpoints/decoder",
	help="Directory to save checkpoints (default: checkpoints/decoder)",
)
@click.option(
	"--log-dir",
	default="runs/decoder",
	help="Directory for tensorboard logs (default: runs/decoder)",
)
@click.option(
	"--batch-size", default=32, type=int, help="Batch size (default: 32)"
)
@click.option(
	"--lr", default=1e-4, type=float, help="Learning rate (default: 1e-4)"
)
@click.option(
	"--epochs", default=100, type=int, help="Number of epochs (default: 100)"
)
@click.option(
	"--num-workers",
	default=4,
	type=int,
	help="Number of dataloader workers (default: 4)",
)
@click.option(
	"--device",
	default="cuda",
	type=click.Choice(["cuda", "cpu"]),
	help="Device to train on (default: cuda)",
)
@click.option(
	"--resume",
	type=click.Path(exists=True),
	help="Path to checkpoint to resume from",
)
@click.option(
	"--val-interval",
	default=1,
	type=int,
	help="Validate every N epochs (default: 1)",
)
@click.option(
	"--weight-log-interval",
	default=10,
	type=int,
	help="Log weight histograms every N epochs (default: 10)",
)
@click.option("--seed", default=42, type=int, help="Random seed (default: 42)")
@click.option(
	"--hidden_size", type=int, default=512, help="Hidden size for the model"
)
def main(
	train_db,
	val_db,
	model,
	checkpoint_dir,
	log_dir,
	batch_size,
	lr,
	epochs,
	num_workers,
	device,
	resume,
	val_interval,
	weight_log_interval,
	seed,
	hidden_size,
):
	"""
	Train forward dynamics + decoder model.

	This script trains a model with both forward dynamics (predicting next hidden state)
	and a decoder (predicting ground truth states from hidden states).
	"""

	# Set random seeds
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)

	# Create dataloaders
	click.echo("Creating dataloaders...")
	train_dataloader, test_dataloader = create_precomputed_dataloaders(
		embedding_db_path=train_db,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		seed=seed,
	)

	# Get a reference sample from test set for visualization
	h_t, a_t, h_t_next, state_t, state_t_next, seq_lengths = next(
		iter(test_dataloader)
	)
	seq_len = seq_lengths[0].item()
	reference_sample = (
		h_t[0:1],  # [1, seq_len, hidden_dim]
		state_t[0:1],  # [1, seq_len, 14]
		seq_len,
	)
	print(f"Reference trajectory length: {seq_len}")

	# Load model
	click.echo(f"Loading model from {model}...")
	net = EnvModel(hidden_size)
	net.load_state_dict(
		torch.load(model, map_location=device)["model_state_dict"]
	)
	net.to(device)

	# Setup optimizer and criterion
	optimizer = optim.Adam(net.parameters(), lr=lr)
	criterion = nn.MSELoss()

	# Setup tensorboard
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_path = Path(log_dir) / timestamp
	writer = SummaryWriter(log_dir=log_path)
	click.echo(f"Tensorboard logs: {log_path}")

	# Log hyperparameters
	writer.add_text(
		"hyperparameters",
		f"""
	- batch_size: {batch_size}
	- learning_rate: {lr}
	- epochs: {epochs}
	- optimizer: Adam
	- criterion: MSELoss
	- device: {device}
	- seed: {seed}
	""",
		0,
	)

	# Resume from checkpoint if specified
	start_epoch = 0
	global_step = 0
	best_val_loss = float("inf")

	if resume:
		click.echo(f"Resuming from checkpoint: {resume}")
		checkpoint = torch.load(resume, map_location=device)
		net.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		start_epoch = checkpoint["epoch"] + 1
		if "val_losses" in checkpoint and checkpoint["val_losses"]:
			best_val_loss = checkpoint["val_losses"]["total_loss"]
		click.echo(f"  Resuming from epoch {start_epoch}")

	# Training loop
	click.echo(f"\nStarting training for {epochs} epochs...")
	click.echo(f"  Device: {device}")
	click.echo(f"  Batch size: {batch_size}")
	click.echo(f"  Learning rate: {lr}")
	click.echo(f"  Training samples: {len(train_dataloader.dataset)}")
	if test_dataloader:
		click.echo(f"  Validation samples: {len(test_dataloader.dataset)}")
	click.echo()

	for epoch in range(start_epoch, epochs):
		# Train
		train_losses, global_step = train_epoch(
			net,
			train_dataloader,
			optimizer,
			criterion,
			device,
			epoch,
			writer,
			global_step,
		)

		click.echo(
			f"Epoch {epoch}: train_loss={train_losses['total_loss']:.4f}, "
			f"fd_loss={train_losses['fd_loss']:.4f}, "
			f"decoder_loss={train_losses['decoder_loss']:.4f}"
		)

		# Validate
		val_losses = None
		if test_dataloader and (epoch % val_interval == 0):
			val_losses = validate_epoch(
				net,
				test_dataloader,
				criterion,
				device,
				epoch,
				writer,
				reference_sample,
			)
			click.echo(
				f"  Val: loss={val_losses['total_loss']:.4f}, "
				f"fd_loss={val_losses['fd_loss']:.4f}, "
				f"decoder_loss={val_losses['decoder_loss']:.4f}"
			)

			# Check if best model
			is_best = val_losses["total_loss"] < best_val_loss
			if is_best:
				best_val_loss = val_losses["total_loss"]
		else:
			is_best = False

		# Log model weights periodically
		if epoch % weight_log_interval == 0:
			log_model_weights(net, writer, epoch)

		# Save checkpoint
		save_checkpoint(
			net,
			optimizer,
			epoch,
			train_losses,
			val_losses,
			checkpoint_dir,
			is_best=is_best,
		)

	writer.close()
	click.echo(f"\n✓ Training complete! Best val_loss: {best_val_loss:.4f}")
	click.echo(f"  Checkpoints saved to: {checkpoint_dir}")
	click.echo(f"  Tensorboard logs: {log_path}")


if __name__ == "__main__":
	main()
