from tqdm import tqdm
import torch
from env_vision_model import EnvModel
from drifter_dataloader_sequential import (
	create_sequence_dataloader as create_dataloader,
)
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# Create the dataloader
dataloader = create_dataloader(
	db_path="drifter_data.db", batch_size=32, shuffle=True, num_workers=4
)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

model = EnvModel().to(dev)
criterion = MSELoss()
opt = Adam(model.parameters())

writer = SummaryWriter()
# Use in training loop
for epoch in range(10):
	for idx, (images, states, seq_lens) in tqdm(enumerate(dataloader), total=len(dataloader)):
		# images: (B, C, H, W) - camera images
		# states: dict with keys:
		#   - 'position': (B, 3)
		#   - 'orientation': (B, 4) - quaternion
		#   - 'velocity': (B, 3)
		#   - 'local_goal': (B, 3)
		#   - 'goal': (B, 3)
		predictions = model(images.to(dev), seq_lens)

		loss = 0.0
		per_output_loss = {}
		for key, value in predictions.items():
			this_loss = criterion(value, states[key].to(dev))

			per_output_loss[key] = this_loss
			loss += this_loss

		# loss = criterion(predictions, states['velocity'].to(dev))
		writer.add_scalars(
			"loss_components",
			per_output_loss,
			epoch * len(dataloader) + idx,
		)
		writer.add_scalar("Loss", loss, epoch * len(dataloader) + idx)

		opt.zero_grad()
		loss.backward()
		opt.step()
