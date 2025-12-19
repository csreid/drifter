from tqdm import tqdm
import torch
import click
from env_vision_model import EnvModel
from drifter_dataloader_sequential import (
	create_sequence_dataloader as create_dataloader,
)
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import mlflow

mlflow.set_tracking_uri("http://localhost:6006")

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

#model = mlflow.pytorch.load_model(f"models:/best_id_model/latest")
model = torch.load('model.pth', weights_only=False)
model.train()

criterion = MSELoss()
opt = Adam(model.parameters())

#writer = SummaryWriter()

def do_logging():
	sample_est = model(sample_imgs.to(dev), sample_seqlens)
	sample_position_est = sample_est["position"]
	true_sample_position = sample_states["position"]

	fig, ax = plt.subplots()

	est_x = sample_position_est[0, :, 0].detach().cpu().numpy()
	est_y = sample_position_est[0, :, 1].detach().cpu().numpy()
	true_x = true_sample_position[0, :, 0].detach().cpu().numpy()
	true_y = true_sample_position[0, :, 1].detach().cpu().numpy()

	n_pts = len(true_x)
	colors = np.arange(n_pts)

	ax.plot(
		est_x,
		est_y,
		marker="o",
		linestyle="--",
		label="Estimated positions",
	)

	scatter = ax.scatter(
		true_x, true_y, c=colors, cmap="plasma", label="True positions"
	)

	ax.set_xbound(-20, 20)
	ax.set_ybound(-20, 20)

	fig.colorbar(scatter, ax=ax, label="Timestep")

	ax.legend()

	mlflow.log_figure(
		fig,
		"trajectory_step{epoch * len(dataloader) + idx}.png"
	)
	plt.close(fig)

#	writer.add_video(
#		"Sampled Trajectory", sample_imgs, epoch * len(dataloader) + idx
#	)


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
def main(train_db, epochs, batch_size):
	dataloader = create_dataloader(
		db_path=train_db,
		batch_size=batch_size,
		shuffle=True,
		num_workers=4,
		min_seq_len=40,
		max_seq_len=75,
	)
	for epoch in range(epochs):
		for idx, (images, states, seq_lens) in tqdm(
			enumerate(dataloader), total=len(dataloader)
		):
			totalidx = epoch * len(dataloader) + idx

			predictions = model(images.to(dev), seq_lens)

			loss = 0.0
			per_output_loss = {}
			for key, value in predictions.items():
				if key in states:
					this_loss = criterion(value, states[key].to(dev))
					per_output_loss[key] = this_loss
					loss += this_loss

			mlflow.log_metrics(
				per_output_loss,
				step=totalidx
			)
			mlflow.log_metric("loss", loss, step=totalidx)

			opt.zero_grad()
			loss.backward()
			opt.step()

if __name__ == '__main__':
	main()
