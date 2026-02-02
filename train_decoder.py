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
	# Model should have:
	# - forward_dynamics_from_hidden(h_t, a_t) -> h_t_next_pred
	# - decode_state(h_t) -> (state_pred, outputs_dict)
	h_t_next_pred = model.forward_dynamics_from_hidden(h_t, a_t)
	state_pred, _ = model.decode_state(h_t)
	state_next_pred, _ = model.decode_state(h_t_next)

	# Compute losses
	# Forward dynamics loss: predict next hidden state
	fd_loss = criterion(h_t_next_pred, h_t_next)

	# Decoder loss: predict current and next states
	decoder_loss_t = criterion(state_pred, state_t)
	decoder_loss_t_next = criterion(state_next_pred, state_t_next)
	decoder_loss = decoder_loss_t + decoder_loss_t_next

	# Total loss (you can adjust weightings)
	loss = fd_loss + decoder_loss

	loss.backward()
	optimizer.step()

	return {
		"total_loss": loss.item(),
		"fd_loss": fd_loss.item(),
		"decoder_loss": decoder_loss.item(),
		"decoder_loss_t": decoder_loss_t.item(),
		"decoder_loss_t_next": decoder_loss_t_next.item(),
	}


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
	}
	total_samples = 0

	pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

	for batch_idx, batch in enumerate(pbar):
		losses = fit_batch(model, optimizer, criterion, batch, device)

		batch_size = batch[0].size(0)

		# Track statistics
		for key in total_losses:
			total_losses[key] += losses[key] * batch_size
		total_samples += batch_size

		# Log to tensorboard every batch
		for key, value in losses.items():
			writer.add_scalar(f"train_batch/{key}", value, global_step)
		global_step += 1

		# Update progress bar
		pbar.set_postfix(
			{
				"loss": f"{losses['total_loss']:.4f}",
				"fd": f"{losses['fd_loss']:.4f}",
				"dec": f"{losses['decoder_loss']:.4f}",
			}
		)

	# Compute epoch averages
	avg_losses = {key: val / total_samples for key, val in total_losses.items()}

	# Log epoch averages to tensorboard
	for key, value in avg_losses.items():
		writer.add_scalar(f"train_epoch/{key}", value, epoch)

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
	total_samples = 0

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
			h_t_next_pred = model.forward_dynamics_from_hidden(h_t, a_t)
			state_pred, _ = model.decode_state(h_t)
			state_next_pred, _ = model.decode_state(h_t_next)

			# Compute losses
			fd_loss = criterion(h_t_next_pred, h_t_next)
			decoder_loss_t = criterion(state_pred, state_t)
			decoder_loss_t_next = criterion(state_next_pred, state_t_next)
			decoder_loss = decoder_loss_t + decoder_loss_t_next
			loss = fd_loss + decoder_loss

			batch_size = h_t.size(0)

			# Track statistics
			total_losses["total_loss"] += loss.item() * batch_size
			total_losses["fd_loss"] += fd_loss.item() * batch_size
			total_losses["decoder_loss"] += decoder_loss.item() * batch_size
			total_losses["decoder_loss_t"] += decoder_loss_t.item() * batch_size
			total_losses["decoder_loss_t_next"] += (
				decoder_loss_t_next.item() * batch_size
			)
			total_samples += batch_size

			pbar.set_postfix(
				{
					"loss": f"{loss.item():.4f}",
				}
			)

	# Compute averages
	avg_losses = {key: val / total_samples for key, val in total_losses.items()}

	# Log to tensorboard
	for key, value in avg_losses.items():
		writer.add_scalar(f"val/{key}", value, epoch)

	return avg_losses


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
@click.option("--seed", default=42, type=int, help="Random seed (default: 42)")
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
	seed,
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
	net = EnvModel()
	net.load_state_dict(torch.load(model, map_location=device)['model_state_dict'])
	net.to(device)

	# Setup optimizer and criterion
	optimizer = optim.Adam(net.parameters(), lr=lr)
	criterion = nn.MSELoss()

	# Setup tensorboard
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_path = Path(log_dir) / timestamp
	writer = SummaryWriter(log_dir=log_path)
	click.echo(f"Tensorboard logs: {log_path}")

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
