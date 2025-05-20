import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from torch import nn, optim, pdist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from q1_network_structure import Encoder, Decoder
from sklearn.manifold import TSNE
import seaborn as sns



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
val_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
batch_size = 128
num_epochs = 20

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Training Function
# -------------------------------
def train_autoencoder(latent_dim, channels, encoder_path, model_name):
    """
    Trains a decoder using a fixed (pretrained and frozen) encoder on the MNIST dataset. Returns the encoder,
     trained decoder, and training/validation losses.
    """
    encoder = Encoder(latent_dim=latent_dim, channels=channels).to(device)
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    decoder = Decoder(latent_dim=latent_dim, channels=channels).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        decoder.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z, idx1, idx2, shape = encoder(images)
            recon = decoder(z, idx1, idx2, shape)
            loss = loss_fn(recon, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                z, idx1, idx2, shape = encoder(images)
                recon = decoder(z, idx1, idx2, shape)
                val_loss += loss_fn(recon, images).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return encoder, decoder, train_losses, val_losses

# -------------------------------
# Show Reconstructions (Fixed Inputs)
# -------------------------------
def show_comparative_reconstructions(models, fixed_inputs, device, model_keys, num_inputs_to_show=None):
    """
    Display reconstructions of a fixed set of inputs across multiple models.

    Args:
        models: list of tuples (encoder, decoder)
        fixed_inputs: tensor of shape (N, C, H, W)
        device: torch device
        model_keys: list of model names
        num_inputs_to_show: int or None, number of inputs to show. If None, show all inputs.
    """
    if num_inputs_to_show is None:
        num_inputs_to_show = fixed_inputs.size(0)
    else:
        num_inputs_to_show = min(num_inputs_to_show, fixed_inputs.size(0))

    fixed_inputs = fixed_inputs[:num_inputs_to_show]

    num_models = len(models)
    num_images = fixed_inputs.size(0)

    total_rows = num_models + 1
    total_cols = num_images + 1
    fig, axes = plt.subplots(total_rows, total_cols, figsize=(total_cols * 2, total_rows * 2))

    # First row: inputs
    axes[0, 0].axis('off')
    for col in range(num_images):
        axes[0, col + 1].set_title(f"Input {col + 1}", fontsize=10)
        axes[0, col + 1].imshow(fixed_inputs[col].squeeze().cpu(), cmap='gray')
        axes[0, col + 1].axis('off')
    axes[0, 0].text(0.5, 0.5, "Input", ha='center', va='center', fontsize=10)

    for row_idx, (key, (encoder, decoder)) in enumerate(zip(model_keys, models), start=1):
        encoder.eval()
        decoder.eval()
        inputs = fixed_inputs.to(device)

        with torch.no_grad():
            z, idx1, idx2, shape = encoder(inputs)
            recon = decoder(z, idx1, idx2, shape)

        recon_cpu = recon.cpu()

        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].text(0.5, 0.5, key, ha='center', va='center', fontsize=10)

        for col in range(num_images):
            axes[row_idx, col + 1].imshow(recon_cpu[col].squeeze(), cmap='gray')
            axes[row_idx, col + 1].axis('off')

    plt.suptitle("Comparison of Reconstructions by Model", fontsize=14)
    plt.tight_layout()
    plt.show()


def show_digit_class_reconstructions(models, model_keys, val_dataset, digit, num_samples, device):
    """
    Display multiple instances of the same digit and their reconstructions for each model.
    """
    # Extract samples of the target digit
    images = []
    for img, label in val_dataset:
        if label == digit:
            images.append(img)
            if len(images) >= num_samples:
                break
    if len(images) < num_samples:
        print(f"Warning: Only found {len(images)} instances of digit {digit}")

    inputs = torch.stack(images).to(device)

    num_models = len(models)
    total_rows = num_models + 1
    total_cols = num_samples + 1
    fig, axes = plt.subplots(total_rows, total_cols, figsize=(total_cols * 2, total_rows * 2))

    # First row: inputs
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, 0.5, f"Digit {digit}", ha='center', va='center', fontsize=10)
    for col in range(num_samples):
        axes[0, col + 1].imshow(inputs[col].squeeze().cpu(), cmap='gray')
        axes[0, col + 1].axis('off')
        axes[0, col + 1].set_title(f"Sample {col + 1}", fontsize=10)

    # Following rows: reconstructions
    for row_idx, (key, (encoder, decoder)) in enumerate(zip(model_keys, models), start=1):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z, idx1, idx2, shape = encoder(inputs)
            recon = decoder(z, idx1, idx2, shape)

        recon_cpu = recon.cpu()
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].text(0.5, 0.5, key, ha='center', va='center', fontsize=10)

        for col in range(num_samples):
            axes[row_idx, col + 1].imshow(recon_cpu[col].squeeze(), cmap='gray')
            axes[row_idx, col + 1].axis('off')

    plt.suptitle(f"Reconstructions of Digit {digit} Across Models", fontsize=14)
    plt.tight_layout()
    plt.show()

# -------------------------------
# Main
# -------------------------------
encoder_configs = [
    {"name": "Q1 encoder", "path": "encoder_d=16,c=16.pth"},
    {"name": "Q2 encoder", "path": "encoder_all.pth"}
]

latent_dim = 16
channels = 16

losses = {}
models = []
model_keys = []

for config in encoder_configs:
    name = config["name"]
    path = config["path"]

    encoder, decoder, train_l, val_l = train_autoencoder(latent_dim, channels, path, name)
    losses[name] = {"train": train_l, "val": val_l}

    torch.save(decoder.state_dict(), f"decoder_{name.replace(' ', '_')}.pth")

    models.append((encoder, decoder))
    model_keys.append(name)

fixed_inputs, _ = next(iter(val_loader))
fixed_inputs = fixed_inputs[:8]

show_comparative_reconstructions(models, fixed_inputs, device, model_keys)
show_digit_class_reconstructions(models, model_keys, val_dataset, digit=4, num_samples=8, device=device)

plt.figure(figsize=(10, 5))
colors = plt.cm.tab10.colors
for i, key in enumerate(model_keys):
    color = colors[i % len(colors)]
    plt.plot(losses[key]["train"], label=f"Train {key}", color=color)
    plt.plot(losses[key]["val"], linestyle="--", label=f"Val {key}", color=color)

plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Training and Validation Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
