from tqdm import tqdm
import torch
import click
from drifter_dataloader_sequential import create_drifter_dataloader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import numpy as np
from pathlib import Path

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

model = torch.load("model.pth", weights_only=False)
model.train()

criterion = MSELoss()
opt = Adam(model.parameters())

component_weights = {
	"velocity": 10.0,
	"orientation": 1.0,
	"position": 1.0,
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


def visualize_trajectory(
	images, true_states, pred_states, seq_len, writer, step
):
	"""
	Visualize predicted vs true trajectory for a single sequence.

	Args:
		images: Image sequence [seq_len, C, H, W]
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


def validate(
	model,
	test_dataloader,
	criterion,
	component_weights,
	device,
	writer,
	step,
	reference_sample=None,
):
	"""
	Validate the model on test set.

	Args:
		model: The model to validate
		test_dataloader: Test data loader
		criterion: Loss function
		component_weights: Weights for each output component
		device: Device to run on
		writer: TensorBoard writer
		step: Global step
		reference_sample: Optional fixed sample for trajectory visualization

	Returns:
		Average validation loss
	"""
	model.eval()
	total_loss = 0.0
	total_samples = 0
	per_component_loss = {key: 0.0 for key in component_weights.keys()}

	with torch.no_grad():
		for batch_idx, (images, states, seq_lens) in enumerate(test_dataloader):
			predictions, pred_as_dict = model(images.to(device), seq_lens)

			batch_loss = 0.0
			for key, value in pred_as_dict.items():
				if key in states:
					component_loss = criterion(value, states[key].to(device))
					weighted_loss = component_loss * component_weights[key]
					per_component_loss[key] += component_loss.item()
					batch_loss += weighted_loss

			total_loss += batch_loss.item()
			total_samples += 1

			# Visualize reference trajectory on first batch if available
			if batch_idx == 0 and reference_sample is not None:
				ref_images, ref_states, ref_seq_len = reference_sample
				ref_predictions, ref_pred_dict = model(
					ref_images.to(device), [ref_seq_len]
				)

				# Extract first sample from batch
				visualize_trajectory(
					ref_images[0],
					{k: v[0] for k, v in ref_states.items()},
					{k: v[0] for k, v in ref_pred_dict.items()},
					ref_seq_len,
					writer,
					step,
				)

	# Log validation metrics
	avg_loss = total_loss / total_samples
	writer.add_scalar("validation/loss", avg_loss, step)

	for key, value in per_component_loss.items():
		avg_component_loss = value / total_samples
		writer.add_scalar(f"validation/loss_{key}", avg_component_loss, step)

	model.train()
	return avg_loss


@click.command()
@click.option(
	"--train_db", type=str, required=True, help="Path to training database"
)
@click.option(
	"--batch_size", type=int, default=8, help="Batch size for training"
)
@click.option(
	"--epochs", type=int, default=50, help="Number of epochs to train"
)
@click.option(
	"--log_dir",
	type=str,
	default=None,
	help="TensorBoard log directory",
)
@click.option(
	"--validate_every",
	type=int,
	default=100,
	help="Run validation every N batches",
)
@click.option(
	"--test_split",
	type=float,
	default=0.2,
	help="Fraction of data for test set",
)
@click.option(
	"--allow-mid-episode/--no-allow-mid-episode",
	default=True,
	help="Allow sequences to start mid-episode (default: True)",
)
def main(
	train_db,
	epochs,
	batch_size,
	log_dir,
	validate_every,
	test_split,
	allow_mid_episode,
):
	if log_dir is None:
		log_dir = f"runs/vision_model/{datetime.now():%Y%m%d-%H%M%S}"

	writer = SummaryWriter(log_dir)

	# Create output directory for models
	output_dir = Path("outputs/vision_model")
	output_dir.mkdir(parents=True, exist_ok=True)
	best_model_path = output_dir / "best_model.pth"

	# Create train and test dataloaders
	train_dataloader, test_dataloader = create_drifter_dataloader(
		db_path=train_db,
		min_seq_len=40,
		max_seq_len=75,
		batch_size=batch_size,
		shuffle=True,
		num_workers=0,
		test_split=test_split,
		allow_mid_episode=allow_mid_episode,
	)

	print(f"Train batches: {len(train_dataloader)}")
	print(f"Test batches: {len(test_dataloader)}")

	# Get a reference sample from test set for visualization
	reference_sample = None
	for images, states, seq_lens in test_dataloader:
		# Take first sample from first batch as reference
		reference_sample = (
			images[0:1],  # Keep batch dimension
			{k: v[0:1] for k, v in states.items()},
			seq_lens[0].item(),
		)
		print(f"Reference trajectory length: {seq_lens[0].item()}")
		break

	best_val_loss = float("inf")
	global_step = 0

	for epoch in range(epochs):
		# Training loop
		model.train()
		for idx, (images, states, seq_lens) in tqdm(
			enumerate(train_dataloader),
			total=len(train_dataloader),
			desc=f"Epoch {epoch + 1}/{epochs}",
		):
			predictions, pred_as_dict = model(images.to(dev), seq_lens)

			loss = 0.0
			per_output_loss = {}
			for key, value in pred_as_dict.items():
				if key in states:
					this_loss = (
						criterion(value, states[key].to(dev))
						* component_weights[key]
					)
					per_output_loss[key] = this_loss
					loss += this_loss.float()

			# Log training losses
			for key, value in per_output_loss.items():
				writer.add_scalar(
					f"train/loss_{key}", value.item(), global_step
				)
			writer.add_scalar("train/loss", loss.item(), global_step)

			opt.zero_grad()
			loss.backward()
			opt.step()

			global_step += 1

			# Periodic validation
			if global_step % validate_every == 0:
				val_loss = validate(
					model,
					test_dataloader,
					criterion,
					component_weights,
					dev,
					writer,
					global_step,
					reference_sample,
				)
				print(
					f"\nValidation at step {global_step}: loss = {val_loss:.6f}"
				)

				# Save best model
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					torch.save(model, best_model_path)
					print(
						f"Saved best model with validation loss: {val_loss:.6f}"
					)

		# Validate at end of each epoch
		val_loss = validate(
			model,
			test_dataloader,
			criterion,
			component_weights,
			dev,
			writer,
			global_step,
			reference_sample,
		)
		print(f"\nEpoch {epoch + 1} validation: loss = {val_loss:.6f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save(model, best_model_path)
			print(f"Saved best model with validation loss: {val_loss:.6f}")

	writer.close()
	print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
	print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
	main()
