import os
import time
import joblib
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# =============================
# 1. Model TinyCNN
# =============================
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(self.layer3(self.layer2(self.layer1(x))))


# =============================
# 2. Early Stopper
# =============================
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0005, path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.min_val_loss = float("inf")

    def check(self, val_loss, model):
        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return False
        else:
            self.counter += 1

        return self.counter >= self.patience


def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            preds = torch.argmax(output, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return correct / total if total > 0 else 0

# =============================
# 3. Training Function
# =============================
def train_model(dataset_path, selected_classes, epochs, batch_size, model_save_name, status_callback=None):
    IMG_SIZE = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    # FILTER CLASS
    idx_map = {cls: i for i, cls in enumerate(selected_classes)}
    filtered_samples = [
        (path, idx_map[full_dataset.classes[label]])
        for path, label in full_dataset.samples
        if full_dataset.classes[label] in selected_classes
    ]

    full_dataset.samples = filtered_samples
    full_dataset.classes = selected_classes

    # SPLIT 70/15/15
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # INIT MODEL
    model = TinyCNN(num_classes=len(selected_classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    stopper = EarlyStopper(path=f"{model_save_name}.pth")

    # STREAMLIT
    progress = st.progress(0)
    graph_loss = st.empty()

    history = {"train_loss": [], "val_loss": []}

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(epochs):
        if status_callback:
            status_callback(f"Training epoch {epoch+1}/{epochs}...")

        model.train()
        running_loss = 0

        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                output = model(img)
                val_loss += criterion(output, label).item()

        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        train_acc = calculate_accuracy(model, train_loader, device)
        val_acc = calculate_accuracy(model, val_loader, device)

        history["train_loss"].append(running_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        graph_loss.line_chart(history)

        progress.progress((epoch + 1) / epochs)
        st.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if stopper.check(val_loss, model):
            st.warning("⛔ Early stopping triggered!")
            break

    # LOAD BEST MODEL
    model.load_state_dict(torch.load(f"{model_save_name}.pth"))

    # SAVE AS JOBLIB
    joblib.dump({
        "state_dict": model.state_dict(),
        "classes": selected_classes
    }, f"{model_save_name}.joblib")

    st.success(f"Model saved as {model_save_name}.joblib")
    return model
