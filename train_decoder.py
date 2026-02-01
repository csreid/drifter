from drifter_dataloader_sequential import (
	create_sequential_dataloader as create_dataloader,
	DrifterSequenceDataset,
)


def fit_batch(model, optimizer, criterion, batch, device):
	imgs, state, seqlens = batch
	state = state.to(device)
	seqlens = seqlens.int()

	optimizer.zero_grad()
	state_pred_tensor, _ = model.get_state(imgs, seqlens)
	loss = criterion(state_pred_tensor, state)

	loss.backward()
	optimizer.step()

	return loss

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
	model.train()

	total_loss = 0.0
	total_samples = 0

	pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

	for batch_idx, (h_t, a_t, h_t_next, seqlens) in enumerate(pbar):
		loss = fit_batch(model, (h_t, a_t, h_t_next, seqlens))

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



def main():
	pass
