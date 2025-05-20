import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from q1_network_structure import Encoder, Decoder

# -------------------------------
# Configuration
# -------------------------------
num_epochs = 20
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Dataset (Official split)
# -------------------------------
transform = transforms.ToTensor()
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Training Function
# -------------------------------
def train_autoencoder(latent_dim, channels):
    encoder = Encoder(latent_dim=latent_dim, channels=channels).to(device)
    decoder = Decoder(latent_dim=latent_dim, channels=channels).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    loss_fn = nn.L1Loss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            z, idx1, idx2, shape = encoder(images)
            recon = decoder(z, idx1, idx2, shape)
            loss = loss_fn(recon, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        encoder.eval()
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

        print(f"[latent={latent_dim}, channels={channels}] "
              f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return encoder, decoder, train_losses, val_losses

# -------------------------------
# Show Reconstructions (Fixed Inputs)
# -------------------------------
def show_comparative_reconstructions(models, fixed_inputs, device, model_keys, loss_fn):
    num_models = len(models)
    num_images = fixed_inputs.size(0)

    total_rows = num_models + 1  # one for input row
    total_cols = num_images + 1  # one for label column
    fig, axes = plt.subplots(total_rows, total_cols, figsize=(total_cols * 2, total_rows * 2))

    # First row: labels
    axes[0, 0].axis('off')  # top-left corner blank
    for col in range(num_images):
        axes[0, col + 1].set_title(f"Input {col + 1}", fontsize=10)
        axes[0, col + 1].imshow(fixed_inputs[col].squeeze().cpu(), cmap='gray')
        axes[0, col + 1].axis('off')
    axes[0, 0].text(0.5, 0.5, "Input", ha='center', va='center', fontsize=10)

    # Remaining rows: model reconstructions
    for row_idx, (key, (encoder, decoder)) in enumerate(zip(model_keys, models), start=1):
        encoder.eval()
        decoder.eval()
        inputs = fixed_inputs.to(device)

        with torch.no_grad():
            z, idx1, idx2, shape = encoder(inputs)
            recon = decoder(z, idx1, idx2, shape)

        recon_cpu = recon.cpu()

        # First column: model label
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].text(0.5, 0.5, key, ha='center', va='center', fontsize=10)

        # Columns: reconstructed images
        for col in range(num_images):
            axes[row_idx, col + 1].imshow(recon_cpu[col].squeeze(), cmap='gray')
            axes[row_idx, col + 1].axis('off')

    plt.suptitle("Comparison of Reconstructions by Model (Rows) vs Input Examples (Columns)", fontsize=14)
    plt.tight_layout()
    plt.show()



# -------------------------------
# Train All Models
# -------------------------------
configs = [(4, 4), (4, 16), (16, 4), (16, 16)]
losses = {}
models = []
model_keys = []

for latent_dim, channels in configs:
    encoder, decoder, train_l, val_l = train_autoencoder(latent_dim, channels)
    key = f"d={latent_dim},c={channels}"
    losses[key] = {'train': train_l, 'val': val_l}

    # Save trained models
    torch.save(encoder.state_dict(), f"encoder_{key}.pth")
    torch.save(decoder.state_dict(), f"decoder_{key}.pth")

    models.append((encoder, decoder))
    model_keys.append(key)

# -------------------------------
# Visualize Reconstructions (Fixed Inputs)
# -------------------------------
fixed_inputs, _ = next(iter(val_loader))
fixed_inputs = fixed_inputs[:6]  # Use first 6 images

show_comparative_reconstructions(models, fixed_inputs, device, model_keys, nn.L1Loss())

# -------------------------------
# Plot All Loss Curves (Log Scale)
# -------------------------------
plt.figure(figsize=(12, 6))
colors = plt.cm.tab10.colors  # 10 distinct colors

for i, key in enumerate(model_keys):
    color = colors[i % len(colors)]
    plt.plot(losses[key]['train'], label=f"Train {key}", color=color)
    plt.plot(losses[key]['val'], linestyle='--', label=f"Val {key}", color=color)

plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Train and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
