import torch
import torch.nn as nn
from torch.optim import Adam
import click
from pathlib import Path
from tqdm import tqdm
import numpy as np

from inverse_dynamics import create_inverse_dynamics_dataloaders
from env_vision_model import EnvModel
import matplotlib.pyplot as plt

import mlflow

mlflow.enable_system_metrics_logging()
mlflow.set_tracking_uri("http://localhost:6006")

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
	"""Train for one epoch."""
	model.train()

	total_loss = 0.0
	total_samples = 0

	pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

	for batch_idx, (images, actions) in enumerate(pbar):
		# Move to device
		images = images.to(device)  # (B, T, C, H, W)
		actions = actions.to(device)  # (B, T, 2)

		# Drop last action (no next state for last image)
		target_actions = actions[:, :-1, :]  # (B, T-1, 2)

		# Get sequence lengths for pack_padded_sequence
		batch_size, seq_len = images.shape[0], images.shape[1]
		seqlens = torch.full((batch_size,), seq_len, dtype=torch.long)

		# Forward pass
		optimizer.zero_grad()
		predicted_actions = model.inverse_dynamics(images, seqlens)  # (B, T, 2)

		# Only compute loss for T-1 timesteps (excluding last prediction)
		predicted_actions = predicted_actions[:, :-1, :]  # (B, T-1, 2)

		# Compute loss
		loss = criterion(predicted_actions, target_actions)

		# Backward pass
		loss.backward()
		optimizer.step()

		# Track statistics
		total_loss += loss.item() * batch_size
		total_samples += batch_size

		# Update progress bar
		pbar.set_postfix(
			{
				"loss": f"{loss.item():.4f}",
				"avg_loss": f"{total_loss / total_samples:.4f}",
			}
		)

	avg_loss = total_loss / total_samples
	return avg_loss


def validate(model, dataloader, criterion, device):
	"""Validate the model."""
	model.eval()

	total_loss = 0.0
	total_samples = 0

	# Track per-action metrics
	steering_errors = []
	throttle_errors = []

	with torch.no_grad():
		for images, actions in tqdm(dataloader, desc="Validating"):
			images = images.to(device)
			actions = actions.to(device)

			target_actions = actions[:, :-1, :]

			batch_size, seq_len = images.shape[0], images.shape[1]
			seqlens = torch.full((batch_size,), seq_len, dtype=torch.long)

			predicted_actions = model.inverse_dynamics(images, seqlens)
			predicted_actions = predicted_actions[:, :-1, :]

			loss = criterion(predicted_actions, target_actions)

			total_loss += loss.item() * batch_size
			total_samples += batch_size

			# Track individual action errors
			steering_error = torch.abs(
				predicted_actions[:, :, 0] - target_actions[:, :, 0]
			)
			throttle_error = torch.abs(
				predicted_actions[:, :, 1] - target_actions[:, :, 1]
			)

			steering_errors.append(steering_error.cpu().numpy())
			throttle_errors.append(throttle_error.cpu().numpy())

	avg_loss = total_loss / total_samples

	# Compute per-action MAE
	steering_errors = np.concatenate(steering_errors)
	throttle_errors = np.concatenate(throttle_errors)

	metrics = {
		"validation_loss": avg_loss,
		"steering_mae": np.mean(steering_errors),
		"throttle_mae": np.mean(throttle_errors),
	}

	return metrics


def plot_predictions_to_tensorboard(
	model, sample_seq, writer, global_step, device
):
	"""
	Visualize model predictions vs ground truth and log to tensorboard.

	Args:
	    model: The trained model
	    sample_seq: Tuple of (images, actions) from dataloader
	    writer: TensorBoard SummaryWriter
	    global_step: Current training step/epoch
	    device: torch device
	"""
	model.eval()

	with torch.no_grad():
		images, actions = sample_seq
		images = images.to(device).float()
		actions = actions.to(device).float()

		# Get first sequence from batch
		images_single = images[0:1]  # (1, T, C, H, W)
		actions_single = actions[0]  # (T, 2)

		# Target actions (drop last)
		target_actions = actions_single[:-1]  # (T-1, 2)

		# Get predictions
		batch_size, seq_len = images_single.shape[0], images_single.shape[1]
		seqlens = torch.full((batch_size,), seq_len, dtype=torch.long)
		predicted_actions = model.inverse_dynamics(
			images_single, seqlens
		)  # (1, T, 2)
		predicted_actions = predicted_actions[0, :-1, :]  # (T-1, 2)

		# Move to CPU for plotting
		pred_np = predicted_actions.cpu().numpy()
		target_np = target_actions.cpu().numpy()

		# Create plot
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
		timesteps = range(len(target_np))

		# Steering plot
		ax1.plot(
			timesteps, target_np[:, 0], "b-", linewidth=2, label="True Steering"
		)
		ax1.plot(
			timesteps,
			pred_np[:, 0],
			"b--",
			linewidth=2,
			label="Predicted Steering",
		)
		ax1.set_ylabel("Steering", fontsize=12)
		ax1.set_ylim(-1.1, 1.1)
		ax1.axhline(y=0, color="k", linestyle=":", alpha=0.3)
		ax1.legend(loc="upper right")
		ax1.grid(True, alpha=0.3)

		# Throttle plot
		ax2.plot(
			timesteps, target_np[:, 1], "r-", linewidth=2, label="True Throttle"
		)
		ax2.plot(
			timesteps,
			pred_np[:, 1],
			"r--",
			linewidth=2,
			label="Predicted Throttle",
		)
		ax2.set_xlabel("Timestep", fontsize=12)
		ax2.set_ylabel("Throttle", fontsize=12)
		ax2.set_ylim(-1.1, 1.1)
		ax2.axhline(y=0, color="k", linestyle=":", alpha=0.3)
		ax2.legend(loc="upper right")
		ax2.grid(True, alpha=0.3)

		plt.tight_layout()

		# Log to tensorboard
		plt.close(fig)

	model.train()


@click.command()
# Data arguments
@click.option(
	"--train_db", type=str, required=True, help="Path to training database"
)
@click.option(
	"--sequence_length",
	type=int,
	default=200,
	help="Length of sequences to process",
)
@click.option(
	"--stride", type=int, default=100, help="Stride for sequence creation"
)
@click.option(
	"--cache_images",
	is_flag=True,
	help="Cache decompressed images in memory",
)
# Model arguments
@click.option(
	"--hidden_size", type=int, default=512, help="Hidden size for the model"
)
# Training arguments
@click.option(
	"--batch_size", type=int, default=8, help="Batch size for training"
)
@click.option("--lr", type=float, default=0.001, help="Learning rate")
@click.option(
	"--epochs", type=int, default=50, help="Number of epochs to train"
)
@click.option(
	"--num_workers",
	type=int,
	default=4,
	help="Number of dataloader workers",
)
# Checkpointing arguments
@click.option(
	"--output_dir",
	type=str,
	default="./outputs",
	help="Directory to save outputs",
)
@click.option(
	"--save_every",
	type=int,
	default=10,
	help="Save checkpoint every N epochs",
)
@click.option(
	"--val_every", type=int, default=1, help="Validate every N epochs"
)
@click.option(
	"--description", type=str, default=None, help="Descriptor that is sent to the logging stuff to help identify a run"
)
@click.option(
	"--val-frac", type=float, default=0., help="Fraction of the dataset that should be held out for validation"
)
# Other arguments
@click.option("--seed", type=int, default=42, help="Random seed")
def main(
	train_db,
	sequence_length,
	stride,
	cache_images,
	hidden_size,
	batch_size,
	lr,
	epochs,
	num_workers,
	output_dir,
	save_every,
	val_every,
	val_frac,
	seed,
	description
):
	# Set random seeds for reproducibility
	torch.manual_seed(seed)
	np.random.seed(seed)

	# Setup device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	# Create output directory
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Create dataloaders
	print("Creating dataloaders...")
	train_loader, val_loader = create_inverse_dynamics_dataloaders(
		db_path=train_db,
		sequence_length=sequence_length,
		stride=stride,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		cache_images=cache_images,
		val_fraction=val_frac
	)

	print(f"Training batches: {len(train_loader)}")
	if val_loader:
		print(f"Validation batches: {len(val_loader)}")

	# Create model
	print(f"Creating model with hidden_size={hidden_size}")
	model = EnvModel(hidden_size=hidden_size)
	model = model.to(device)

	# Print model size
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model has {num_params:,} trainable parameters")

	# Create optimizer and loss
	optimizer = Adam(model.parameters(), lr=lr)
	criterion = nn.MSELoss()

	# Training loop
	best_val_loss = float("inf")

	#sample_seq = next(iter(train_loader))
	#writer.add_video("sample", sample_seq[0], 0)

	params = {
		"epochs": epochs,
		"dataset_size": len(train_loader),
		"batch_size": batch_size,
		"sequence_length": sequence_length,
		"hidden_size": hidden_size,
	}

	with mlflow.start_run():
		mlflow.log_params(params)

		for epoch in range(epochs):
			# Train
			train_loss = train_epoch(
				model, train_loader, optimizer, criterion, device, epoch
			)

			mlflow.log_metrics(
				{
					"train_loss": train_loss,
				},
				step=epoch
			)

			# Validate
			if val_loader and epoch % val_every == 0:
				val_metrics = validate(model, val_loader, criterion, device)
				print(f"Val Loss: {val_metrics['validation_loss']:.4f}")
				print(f"Val Steering MAE: {val_metrics['steering_mae']:.4f}")
				print(f"Val Throttle MAE: {val_metrics['throttle_mae']:.4f}")

				mlflow.log_metrics(val_metrics, step=epoch)

				# Save best model
				if val_metrics["validation_loss"] < best_val_loss:
					best_val_loss = val_metrics["validation_loss"]
					model_info = mlflow.pytorch.log_model(
						pytorch_model=model,
						name='best_id_model_epoch{epoch}',
						step=epoch,
					)
					mlflow.log_metric(
						key="validation_loss",
						value=val_metrics['validation_loss'],
						step=epoch,
						model_id=model_info.model_id
					)

			# Save checkpoint
			if epoch % save_every == 0:
				model_info = mlflow.pytorch.log_model(
					model,
					name='id_model_checkpoint',
					step=epoch
				)
				mlflow.log_metric(
					key="validation_loss",
					value=val_metrics['validation_loss'],
					step=epoch,
					model_id=model_info.model_id
				)

	print("\nTraining complete!")


if __name__ == "__main__":
	main()
