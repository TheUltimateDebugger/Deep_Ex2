import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from network_structure import Encoder, Decoder

# -------------------------------
# Configuration
# -------------------------------
latent_dim = 16
channels = 4  # try also 16 for large model
num_epochs = 10
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Dataset
# -------------------------------
transform = transforms.ToTensor()
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
encoder = Encoder(latent_dim=latent_dim, channels=channels).to(device)
decoder = Decoder(latent_dim=latent_dim, channels=channels).to(device)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=1e-3)
loss_fn = nn.L1Loss()

# -------------------------------
# Training Loop
# -------------------------------
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# -------------------------------
# Optional: Visualize Reconstruction
# -------------------------------
import numpy as np

encoder.eval()
decoder.eval()
with torch.no_grad():
    images, _ = next(iter(train_loader))
    images = images.to(device)
    z, idx1, idx2, shape = encoder(images)
    recon = decoder(z, idx1, idx2, shape)

    # Show 6 images
    fig, axs = plt.subplots(2, 6, figsize=(12, 4))
    for i in range(6):
        axs[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axs[0, i].set_title("Original")
        axs[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
        axs[1, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()
