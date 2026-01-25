import torch
import torch.nn as nn
from torch.optim import Adam
import click
from pathlib import Path
from tqdm import tqdm
import numpy as np

from forward_dynamics_dataloader import create_forward_dynamics_dataloader
from env_vision_model import EnvModel

import mlflow

mlflow.enable_system_metrics_logging()
mlflow.set_tracking_uri("http://localhost:6006")

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
	"""Train for one epoch."""
	model.train()

	total_loss = 0.0
	total_samples = 0

	pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

	for batch_idx, (h_t, a_t, h_t_next, seqlens) in enumerate(pbar):
		# Move to device
		h_t = h_t.to(device)  # (B, T, N)
		a_t = a_t.to(device)  # (B, T, 2)
		h_t_next = h_t_next.to(device)  # (B, T, N)

		print(f"{h_t_next.shape=}")
		seqlens = seqlens.int()

		batch_size = h_t.shape[0]

		# Forward pass
		optimizer.zero_grad()
		predicted_next = model._simple_forward_dynamics(h_t, a_t)  # (B, T, 2)

		# Compute loss
		loss = criterion(predicted_next, h_t_next.detach())

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
	"--description",
	type=str,
	default=None,
	help="Descriptor that is sent to the logging stuff to help identify a run",
)
@click.option(
	"--id-model",
	type=str,
	default=None,
	help="Path to trained ID model",
	required=True
)
@click.option(
	"--val-frac",
	type=float,
	default=0.0,
	help="Fraction of the dataset that should be held out for validation",
)
# Other arguments
@click.option("--seed", type=int, default=42, help="Random seed")
def main(
	train_db,
	id_model,
	sequence_length,
	batch_size,
	lr,
	epochs,
	num_workers,
	output_dir,
	save_every,
	val_every,
	val_frac,
	seed,
	description,
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

	model=torch.load(id_model, weights_only=False, map_location=torch.device(device))
	model.train()

	# Create dataloaders
	print("Creating dataloaders...")
	train_loader = create_forward_dynamics_dataloader(
		db_path=train_db,
		id_model=model,
		sequence_length=sequence_length,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		device=device
	)

	print(f"Training batches: {len(train_loader)}")

	# Print model size
	model.train()
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model has {num_params:,} trainable parameters")

	# Create optimizer and loss
	optimizer = Adam(model.parameters(), lr=lr)
	criterion = nn.MSELoss()

	params = {
		"epochs": epochs,
		"dataset_size": len(train_loader),
		"batch_size": batch_size,
		"sequence_length": sequence_length,
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
				step=epoch,
			)

	print("\nTraining complete!")

if __name__ == "__main__":
	main()
