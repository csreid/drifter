from env_vision_model import EnvModel
from drifter_dataloader import create_dataloader
from torch.nn import MSELoss
from torch.optim import Adam

# Create the dataloader
dataloader = create_dataloader(
	db_path="drifter_data.db", batch_size=32, shuffle=True, num_workers=4
)

model = EnvModel()
criterion = MSELoss()
opt = Adam(model.parameters())
# Use in training loop
for images, states in dataloader:
	# images: (B, C, H, W) - camera images
	# states: dict with keys:
	#   - 'position': (B, 3)
	#   - 'orientation': (B, 4) - quaternion
	#   - 'velocity': (B, 3)
	#   - 'local_goal': (B, 3)
	#   - 'goal': (B, 3)
	predictions = model(images)
	loss = criterion(predictions, states)

	opt.zero_grad()
	loss.backward()
	opt.step()
