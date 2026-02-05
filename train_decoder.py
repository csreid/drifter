import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from pathlib import Path
from datetime import datetime
from env_vision_model import EnvModel

from precomputed_dataloader import create_precomputed_dataloader


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
t 
	# Component losses (without local_goal)
	pos_loss_t = criterion(
		state_pred[..., :3], state_t[..., :3]
	)
	vel_loss_t = criterion(
		state_pred[..., 3:6], state_t[..., 3:6]
	)
	flipped_loss_t = criterion(
		state_pred[..., 6:7], state_t[..., 6:7]
	)
	orient_loss_t = criterion(
		state_pred[..., 7:], state_t[..., 7:]
	)

	pos_loss_t_next = criterion(
		state_next_pred[..., :3], state_t_next[..., :3]
	)
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
		"state_pred": state_pred_no_goal.detach(),
		"state_t": state_t_no_goal.detach(),
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


def validate_epoch(model, dataloader, criterion, device, epoch, writer):
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
			state_t_no_goal = torch.cat(
				[state_t[..., :3], state_t[..., 6:]], dim=-1
			)
			state_pred_no_goal = torch.cat(
				[state_pred[..., :3], state_pred[..., 6:]], dim=-1
			)
			state_t_next_no_goal = torch.cat(
				[state_t_next[..., :3], state_t_next[..., 6:]], dim=-1
			)
			state_next_pred_no_goal = torch.cat(
				[state_next_pred[..., :3], state_next_pred[..., 6:]], dim=-1
			)

			# Component losses
			pos_loss_t = criterion(
				state_pred_no_goal[..., :3], state_t_no_goal[..., :3]
			)
			vel_loss_t = criterion(
				state_pred_no_goal[..., 3:6], state_t_no_goal[..., 3:6]
			)
			flipped_loss_t = criterion(
				state_pred_no_goal[..., 6:7], state_t_no_goal[..., 6:7]
			)
			orient_loss_t = criterion(
				state_pred_no_goal[..., 7:], state_t_no_goal[..., 7:]
			)

			pos_loss_t_next = criterion(
				state_next_pred_no_goal[..., :3], state_t_next_no_goal[..., :3]
			)
			vel_loss_t_next = criterion(
				state_next_pred_no_goal[..., 3:6],
				state_t_next_no_goal[..., 3:6],
			)
			flipped_loss_t_next = criterion(
				state_next_pred_no_goal[..., 6:7],
				state_t_next_no_goal[..., 6:7],
			)
			orient_loss_t_next = criterion(
				state_next_pred_no_goal[..., 7:], state_t_next_no_goal[..., 7:]
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
			all_state_errors.append(
				(state_pred_no_goal - state_t_no_goal).abs().cpu()
			)

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
	hidden_size
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
	train_dataloader = create_precomputed_dataloader(
		embedding_db_path=train_db,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		seed=seed,
	)

	val_dataloader = None
	if val_db:
		val_dataloader = create_precomputed_dataloader(
			embedding_db_path=val_db,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			seed=seed,
		)

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
	if val_dataloader:
		click.echo(f"  Validation samples: {len(val_dataloader.dataset)}")
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
		if val_dataloader and (epoch % val_interval == 0):
			val_losses = validate_epoch(
				net, val_dataloader, criterion, device, epoch, writer
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
