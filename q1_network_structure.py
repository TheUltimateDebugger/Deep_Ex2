import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=16, channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.fc = nn.Linear((channels * 2) * 7 * 7, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, idx1 = self.pool1(x)
        x = F.relu(self.conv2(x))
        x, idx2 = self.pool2(x)
        shape = x.size()
        x = torch.flatten(x, 1)
        z = self.fc(x)
        return z, idx1, idx2, shape

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, channels=4):
        super().__init__()
        self.fc = nn.Linear(latent_dim, (channels * 2) * 7 * 7)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, z, idx1, idx2, shape):
        x = self.fc(z)
        x = x.view(shape)
        x = self.unpool2(x, idx2)
        x = F.relu(self.deconv2(x))
        x = self.unpool1(x, idx1)
        x = torch.sigmoid(self.deconv1(x))
        return x
