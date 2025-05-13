import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierHead(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=64, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),                 # Non-linear activation
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z):
        return self.model(z)
