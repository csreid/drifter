from tqdm import tqdm
import torch
from env_vision_model import EnvModel
from drifter_dataloader_sequential import (
	create_sequence_dataloader as create_dataloader,
)
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

# Create the dataloader
dataloader = create_dataloader(
	db_path="drifter_data.db",
	batch_size=4,
	shuffle=True,
	num_workers=4,
	min_seq_len=40,
	max_seq_len=75,
)
sample_dataloader = create_dataloader(
	db_path="drifter_data.db",
	batch_size=1,
	shuffle=True,
	num_workers=4,
	min_seq_len=40,
)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

model = EnvModel(hidden_size=1024).to(dev)
criterion = MSELoss()
opt = Adam(model.parameters())

writer = SummaryWriter()

sample_imgs, sample_states, sample_seqlens = next(iter(sample_dataloader))


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

	writer.add_figure(
		"Estimated vs True positions",
		fig,
		epoch * len(dataloader) + idx,
	)
	plt.close(fig)

	writer.add_video(
		"Sampled Trajectory", sample_imgs, epoch * len(dataloader) + idx
	)


for epoch in range(20):
	for idx, (images, states, seq_lens) in tqdm(
		enumerate(dataloader), total=len(dataloader)
	):
		predictions = model(images.to(dev), seq_lens)

		loss = 0.0
		per_output_loss = {}
		for key, value in predictions.items():
			this_loss = criterion(value, states[key].to(dev))

			per_output_loss[key] = this_loss
			loss += this_loss

		writer.add_scalars(
			"loss_components",
			per_output_loss,
			epoch * len(dataloader) + idx,
		)
		writer.add_scalar("Loss", loss, epoch * len(dataloader) + idx)

		opt.zero_grad()
		loss.backward()
		opt.step()

	with torch.no_grad():
		do_logging()
