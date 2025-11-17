from tqdm import tqdm
import torch
from env_vision_model import EnvModel
from drifter_dataloader_sequential import create_sequence_dataloader as create_dataloader
from torch.nn import MSELoss
from torch.optim import Adam

# Create the dataloader
dataloader = create_dataloader(
	db_path="drifter_data.db", batch_size=32, shuffle=True, num_workers=4
)

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = EnvModel().to(dev)
criterion = MSELoss()
opt = Adam(model.parameters())

# Use in training loop
for images, states, seq_lens in tqdm(dataloader):
	# images: (B, C, H, W) - camera images
	# states: dict with keys:
	#   - 'position': (B, 3)
	#   - 'orientation': (B, 4) - quaternion
	#   - 'velocity': (B, 3)
	#   - 'local_goal': (B, 3)
	#   - 'goal': (B, 3)
	predictions = model(images.to(dev))

	loss = 0.
	for key, value in predictions.items():
		loss += criterion(value, states[key].to(dev))

	#loss = criterion(predictions, states['velocity'].to(dev))

	opt.zero_grad()
	loss.backward()
	opt.step()
