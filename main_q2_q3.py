import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import random
import q3_training
from network_structure import Encoder
from q2_classification_model import ClassifierHead
from train_q2 import train_model, plot_results, show_predictions_dual

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Shared setup ---
transform = transforms.ToTensor()
full_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
class_names = [str(i) for i in range(10)]

# --------------------------
# COMPARISON 1: All Data vs 100-Sample
# --------------------------

# All data model (train encoder)
encoder_all = Encoder(latent_dim=16, channels=16)
classifier_all = ClassifierHead(latent_dim=16, num_classes=10)
optimizer_all = optim.Adam(list(encoder_all.parameters()) + list(classifier_all.parameters()), lr=1e-3)
train_loader_all = DataLoader(train_dataset, batch_size=128, shuffle=True)

train_losses_all, train_accs_all, test_losses_all, test_accs_all = train_model(
    encoder=encoder_all,
    classifier_head=classifier_all,
    train_loader=train_loader_all,
    test_loader=val_loader,
    optimizer=optimizer_all,
    criterion=nn.CrossEntropyLoss(),
    epochs=30,
    device=device
)

# 100-sample model (train encoder)
subset_indices = random.sample(range(len(train_dataset)), 100)
small_train_dataset = Subset(train_dataset, subset_indices)
small_train_loader = DataLoader(small_train_dataset, batch_size=32, shuffle=True)

encoder_100 = Encoder(latent_dim=16, channels=16)
classifier_100 = ClassifierHead(latent_dim=16, num_classes=10)
optimizer_100 = optim.Adam(list(encoder_100.parameters()) + list(classifier_100.parameters()), lr=1e-3)

train_losses_100, train_accs_100, test_losses_100, test_accs_100 = train_model(
    encoder=encoder_100,
    classifier_head=classifier_100,
    train_loader=small_train_loader,
    test_loader=val_loader,
    optimizer=optimizer_100,
    criterion=nn.CrossEntropyLoss(),
    epochs=30,
    device=device
)

# Plot: All data vs 100-sample
plot_results(
    train1=train_losses_all, acc1=train_accs_all,
    test1=test_losses_all, test_acc1=test_accs_all,
    train2=train_losses_100, acc2=train_accs_100,
    test2=test_losses_100, test_acc2=test_accs_100,
    label1="All Data",
    label2="100 Samples",
    title_suffix="(Both Training Encoder)"
)

# 👀 Predictions: All data vs 100
show_predictions_dual(
    encoder1=encoder_all, classifier1=classifier_all,
    encoder2=encoder_100, classifier2=classifier_100,
    test_loader=val_loader,
    class_names=class_names,
    num_examples=10,
    device=device,
    explanation="Top: Full Training | Bottom: 100-sample Training"
)


# --------------------------
#  COMPARISON 2: All Data (Train Encoder) vs Frozen Encoder
# --------------------------

# Load frozen encoder
encoder_frozen = Encoder(latent_dim=16, channels=16)
encoder_frozen.load_state_dict(torch.load("encoder_d=16,c=16.pth", map_location=device))
classifier_frozen = ClassifierHead(latent_dim=16, num_classes=10)

# Freeze encoder
for param in encoder_frozen.parameters():
    param.requires_grad = False

optimizer_frozen = optim.Adam(classifier_frozen.parameters(), lr=1e-3)

train_losses_frozen, train_accs_frozen, test_losses_frozen, test_accs_frozen = q3_training.train_model_q3(
    encoder=encoder_frozen,
    classifier_head=classifier_frozen,
    train_loader=train_loader_all,
    test_loader=val_loader,
    optimizer=optimizer_frozen,
    criterion=nn.CrossEntropyLoss(),
    epochs=30)


# Plot: Train encoder vs Freeze encoder (same data size)
plot_results(
    train1=train_losses_all, acc1=train_accs_all,
    test1=test_losses_all, test_acc1=test_accs_all,
    train2=train_losses_frozen, acc2=train_accs_frozen,
    test2=test_losses_frozen, test_acc2=test_accs_frozen,
    label1="Train Encoder",
    label2="Frozen Encoder",
    title_suffix="(Same Dataset)"
)

# 👀 Predictions: Train encoder vs frozen
show_predictions_dual(
    encoder1=encoder_all, classifier1=classifier_all,
    encoder2=encoder_frozen, classifier2=classifier_frozen,
    test_loader=val_loader,
    class_names=class_names,
    num_examples=8,
    device=device,
    explanation="Top: Full Training | Bottom: frozen encoder"
)

# --------------------------
# COMPARISON 3: 100-sample Training (Train Encoder vs Frozen Encoder)
# --------------------------

# Load frozen encoder
encoder_100_frozen = Encoder(latent_dim=16, channels=16)
encoder_100_frozen.load_state_dict(torch.load("encoder_d=16,c=16.pth", map_location=device))

# Freeze encoder
for param in encoder_100_frozen.parameters():
    param.requires_grad = False

classifier_100_frozen = ClassifierHead(latent_dim=16, num_classes=10)
optimizer_100_frozen = optim.Adam(classifier_100_frozen.parameters(), lr=1e-3)

train_losses_100_frozen, train_accs_100_frozen, test_losses_100_frozen, test_accs_100_frozen = q3_training.train_model_q3(
    encoder=encoder_100_frozen,
    classifier_head=classifier_100_frozen,
    train_loader=small_train_loader,
    test_loader=val_loader,
    optimizer=optimizer_100_frozen,
    criterion=nn.CrossEntropyLoss(),
    epochs=30
)

# Plot: Small Data (trainable vs frozen encoder)
plot_results(
    train1=train_losses_100, acc1=train_accs_100,
    test1=test_losses_100, test_acc1=test_accs_100,
    train2=train_losses_100_frozen, acc2=train_accs_100_frozen,
    test2=test_losses_100_frozen, test_acc2=test_accs_100_frozen,
    label1="Train Encoder (100 Samples)",
    label2="Frozen Encoder (100 Samples)",
    title_suffix="(Small Dataset)"
)

#  Predictions: Small dataset training (trainable vs frozen encoder)
show_predictions_dual(
    encoder1=encoder_100, classifier1=classifier_100,
    encoder2=encoder_100_frozen, classifier2=classifier_100_frozen,
    test_loader=val_loader,
    class_names=class_names,
    num_examples=8,
    device=device,
    explanation="Top: Trained Encoder (100 Samples) | Bottom: Frozen Encoder (100 Samples)"
)
