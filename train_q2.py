import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from network_structure import Encoder, Decoder
from q2_classification_model import ClassifierHead
import random
import matplotlib.pyplot as plt

def plot_results(train1, acc1, test1, test_acc1, train2, acc2, test2, test_acc2, label1="Full", label2="Small", title_suffix=""):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train1, label=f"{label1} Train Loss")
    plt.plot(test1, label=f"{label1} Test Loss")
    plt.plot(train2, label=f"{label2} Train Loss", linestyle='--')
    plt.plot(test2, label=f"{label2} Test Loss", linestyle='--')
    plt.title(f"Loss {title_suffix}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(acc1, label=f"{label1} Train Acc")
    plt.plot(test_acc1, label=f"{label1} Test Acc")
    plt.plot(acc2, label=f"{label2} Train Acc", linestyle='--')
    plt.plot(test_acc2, label=f"{label2} Test Acc", linestyle='--')
    plt.title(f"Accuracy {title_suffix}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(encoder, classifier_head, train_loader, test_loader, optimizer, criterion, epochs=10, device='cpu'):
    encoder.to(device)
    classifier_head.to(device)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        encoder.train()
        classifier_head.train()
        correct, total, running_loss = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            z, _, _, _ = encoder(x)
            logits = classifier_head(z)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(correct / total)

        # Evaluation
        encoder.eval()
        classifier_head.eval()
        correct, total, test_loss = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                z, _, _, _ = encoder(x)
                logits = classifier_head(z)
                loss = criterion(logits, y)
                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        test_losses.append(test_loss / len(test_loader))
        test_accs.append(correct / total)

        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

    return train_losses, train_accs, test_losses, test_accs


def show_predictions_dual(encoder1, classifier1, encoder2, classifier2, test_loader, class_names=None, num_examples=8, device='cpu', explanation =""):
    encoder1.eval()
    classifier1.eval()
    encoder2.eval()
    classifier2.eval()

    encoder1.to(device)
    classifier1.to(device)
    encoder2.to(device)
    classifier2.to(device)

    # Get a batch from the test set
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            z1, _, _, _ = encoder1(x)
            logits1 = classifier1(z1)
            preds1 = torch.argmax(logits1, dim=1)

            z2, _, _, _ = encoder2(x)
            logits2 = classifier2(z2)
            preds2 = torch.argmax(logits2, dim=1)
            break  # Only one batch

    # Plot predictions
    plt.figure(figsize=(15, 6))
    for i in range(min(num_examples, len(x))):
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(x[i].cpu().squeeze(), cmap='gray')
        title = f"Full: {preds1[i].item()}"
        if class_names:
            title += f"\n(True: {class_names[y[i].item()]})"
        else:
            title += f"\n(True: {y[i].item()})"
        plt.title(title)
        plt.axis('off')

        plt.subplot(2, num_examples, num_examples + i + 1)
        plt.imshow(x[i].cpu().squeeze(), cmap='gray')
        title = f"100-sample: {preds2[i].item()}"
        if class_names:
            title += f"\n(True: {class_names[y[i].item()]})"
        else:
            title += f"\n(True: {y[i].item()})"
        plt.title(title)
        plt.axis('off')

    plt.suptitle(f"{explanation}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

