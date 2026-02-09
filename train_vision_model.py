from tqdm import tqdm
import torch
import click
from drifter_dataloader_sequential import (
	create_sequential_dataloader as create_dataloader,
	DrifterSequenceDataset,
)
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

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
@click.option(
	"--log_dir",
	type=str,
	default="runs/vision_model",
	help="TensorBoard log directory",
)
def main(train_db, epochs, batch_size, log_dir):
	writer = SummaryWriter(log_dir)

	dataset = DrifterSequenceDataset(
		db_path=train_db,
		min_seq_len=40,
		max_seq_len=75,
	)
	dataloader = create_dataloader(
		dataset=dataset,
		batch_size=batch_size,
		shuffle=True,
	)
	for epoch in range(epochs):
		for idx, (images, states, seq_lens) in tqdm(
			enumerate(dataloader), total=len(dataloader)
		):
			totalidx = epoch * len(dataloader) + idx

			predictions, pred_as_dict = model(images.to(dev), seq_lens)

			loss = 0.0
			per_output_loss = {}
			for key, value in pred_as_dict.items():
				if key in states:
					this_loss = criterion(value, states[key].to(dev))
					per_output_loss[key] = this_loss
					loss += this_loss

			# Log per-output losses
			for key, value in per_output_loss.items():
				writer.add_scalar(f"loss/{key}", value.item(), totalidx)

			# Log total loss
			writer.add_scalar("loss/total", loss.item(), totalidx)

			opt.zero_grad()
			loss.backward()
			opt.step()

	writer.close()


if __name__ == "__main__":
	main()
