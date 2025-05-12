import torch.nn as nn
import torch.nn.functional as F


class MLPBasicClassifier(nn.Module):
    """
    the basic classifier with the requested layers
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(180, 180)
        self.fc2 = nn.Linear(180, 180)
        self.output = nn.Linear(180, 7)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.output(x)


class MLPBetterClassifier(nn.Module):
    """
    the improved classifier we made
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(180, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 7)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.output(x)


class LinearClassifier(nn.Module):
    """
    the improved classifier with all the none linear components removed
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(180, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 7)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return self.output(x)
