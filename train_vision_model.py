from tqdm import tqdm
import torch
import click
from drifter_dataloader_sequential import (
	create_sequence_dataloader as create_dataloader,
)
from torch.nn import MSELoss
from torch.optim import Adam
import mlflow

mlflow.set_tracking_uri("http://localhost:6006")

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

model = torch.load("model.pth", weights_only=False)
model.train()

criterion = MSELoss()
opt = Adam(model.parameters())

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

			mlflow.log_metrics(per_output_loss, step=totalidx)
			mlflow.log_metric("loss", loss, step=totalidx)

			opt.zero_grad()
			loss.backward()
			opt.step()


if __name__ == "__main__":
	main()
