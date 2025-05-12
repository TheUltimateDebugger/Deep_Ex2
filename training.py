import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from network_structure import Encoder, Decoder

# -------------------------------
# Configuration
# -------------------------------
latent_dim = 16
channels = 4
num_epochs = 20
batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -------------------------------
# Dataset
# -------------------------------
transform = transforms.ToTensor()
full_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))  # 48000
test_size = len(full_dataset) - train_size  # 12000
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    return train_losses, val_losses

# -------------------------------
# Train All 4 Models
# -------------------------------
configs = [(4, 4), (4, 16), (16, 4), (16, 16)]
losses = {}

for latent_dim, channels in configs:
    train_l, val_l = train_autoencoder(latent_dim, channels)
    key = f"d={latent_dim},c={channels}"
    losses[key] = {'train': train_l, 'val': val_l}

# -------------------------------
# Plot All Loss Curves (Log Scale)
# -------------------------------
plt.figure(figsize=(12, 6))
colors = plt.cm.tab10.colors  # 10 distinct colors
model_keys = list(losses.keys())

for i, key in enumerate(model_keys):
    color = colors[i % len(colors)]
    plt.plot(losses[key]['train'], label=f"Train {key}", color=color)
    plt.plot(losses[key]['val'], linestyle='--', label=f"Val {key}", color=color)

plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Train and Validation Loss (log scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

