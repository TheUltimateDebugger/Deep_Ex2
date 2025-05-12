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
channels = 4  # try also 16 for large model
num_epochs = 50
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
def train_autoencoder(latent_dim):
    encoder = Encoder(latent_dim=latent_dim, channels=channels).to(device)
    decoder = Decoder(latent_dim=latent_dim, channels=channels).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=1e-2)
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

        # Validation
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

        print(f"Latent {latent_dim} | Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

# -------------------------------
# Train Both Models
# -------------------------------
train_16, val_16 = train_autoencoder(latent_dim=16)
train_4, val_4 = train_autoencoder(latent_dim=4)

# -------------------------------
# Plot Loss Curves
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_16, label='Train Loss (d=16)')
plt.plot(val_16, label='Val Loss (d=16)')
plt.plot(train_4, label='Train Loss (d=4)')
plt.plot(val_4, label='Val Loss (d=4)')
plt.title('Training and Validation Loss for d=16 and d=4')
plt.xlabel('Epoch')
plt.ylabel('L1 Loss')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.show()
