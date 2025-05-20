import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from q1_network_structure import Encoder, Decoder
from q2_classification_model import ClassifierHead
import random
import matplotlib.pyplot as plt


def train_model_q3(encoder, classifier_head, train_loader, test_loader, optimizer, criterion, epochs=10, device='cpu'):
    encoder.to(device)
    classifier_head.to(device)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
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

        print(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

    return train_losses, train_accs, test_losses, test_accs


def show_predictions_single(encoder, classifier, test_loader, class_names=None, num_examples=8, device='cpu'):
    encoder.eval()
    classifier.eval()

    encoder.to(device)
    classifier.to(device)

    # Get a batch from the test set
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            z, _, _, _ = encoder(x)
            logits = classifier(z)
            preds = torch.argmax(logits, dim=1)
            break  # Only one batch

    # Plot predictions
    plt.figure(figsize=(12, 4))
    for i in range(min(num_examples, len(x))):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(x[i].cpu().squeeze(), cmap='gray')
        if class_names:
            title = f"{class_names[preds[i].item()]}\n(True: {class_names[y[i].item()]})"
        else:
            title = f"{preds[i].item()}\n(True: {y[i].item()})"
        plt.title(title)
        plt.axis('off')

    plt.suptitle("Model Predictions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


if __name__ == '__main__':
    encoder = Encoder(latent_dim=16, channels=16)
    encoder.load_state_dict(torch.load('encoder_d=16,c=16.pth', map_location='cpu'))
    classifier_head = ClassifierHead(latent_dim=16, num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(classifier_head.parameters()), lr=1e-3)
    batch_size = 128

    transform = transforms.ToTensor()
    full_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))  # 48000
    test_size = len(full_dataset) - train_size  # 12000
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False

    # Make sure optimizer only updates classifier parameters
    optimizer = optim.Adam(classifier_head.parameters(), lr=1e-3)

    # Now call the training function
    train_losses, train_accs, test_losses, test_accs = train_model_q3(
        encoder=encoder,
        classifier_head=classifier_head,
        train_loader=train_loader,
        test_loader=val_loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        epochs=10
    )

    show_predictions_single(
        encoder=encoder,
        classifier=classifier_head,
        test_loader=val_loader,
        class_names=[str(i) for i in range(10)],
        num_examples=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
