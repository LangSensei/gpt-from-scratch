import torch
from torch.utils.data import DataLoader
from tokenizer import Tokenizer
from model import Model
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class ModelTrainer:
    def __init__(self, model: Model, optimizer: torch.optim.AdamW, tokenizer: Tokenizer):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, eval_freq: int, eval_iter: int, start_context: str) -> tuple[list[float], list[float], list[int]]:
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            
            for input_batch, target_batch in train_loader:
                self.optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = self._calc_loss_batch(input_batch, target_batch)
                loss.backward() # Calculate loss gradients
                self.optimizer.step() # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self._evaluate_model(train_loader, val_loader, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Print a sample text after each epoch
            self._generate_and_print_sample(start_context)

        return train_losses, val_losses, track_tokens_seen

    def plot_losses(self, epochs_seen: torch.Tensor, tokens_seen: list[int], train_losses: list[float], val_losses: list[float]) -> None:
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room
        plt.show()

    def _evaluate_model(self, train_loader: DataLoader, val_loader: DataLoader, eval_iter: int) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            train_loss = self._calc_loss_loader(train_loader, num_batches=eval_iter)
            val_loss = self._calc_loss_loader(val_loader, num_batches=eval_iter)
        self.model.train()
        return train_loss, val_loss

    def _generate_and_print_sample(self, start_context: str) -> None:
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = Tokenizer.text_to_token_ids(start_context, self.tokenizer).to(self.device)
        with torch.no_grad():
            token_ids = self.model.generate(
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size
            )
        decoded_text = Tokenizer.token_ids_to_text(token_ids, self.tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
        self.model.train()

    def _calc_loss_loader(self, data_loader: DataLoader, num_batches: int = None) -> float:
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self._calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def _calc_loss_batch(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> float:
        input_batch, target_batch = input_batch.to(self.device), target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss