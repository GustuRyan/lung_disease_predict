import streamlit as st
from torchvision.datasets import DatasetFolder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch, torch.nn as nn, torch.optim as optim
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from cnn3ipynb import TinyCNN


def train_dynamic_classes(
    selected_classes,
    dataset_root,
    epochs,
    batch_size,
    progress_callback=None,
    status_callback=None,
):
    IMG_SIZE = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.ImageFolder(dataset_root, transform=transform)

    # =========================
    # FILTER CLASS
    # =========================
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

    filtered_samples = [
        (path, class_to_idx[dataset.classes[label]])
        for path, label in dataset.samples
        if dataset.classes[label] in selected_classes
    ]

    dataset.samples = filtered_samples
    dataset.classes = selected_classes

    # =========================
    # SPLIT DATASET
    # =========================
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # =========================
    # MODEL
    # =========================
    model = TinyCNN(num_classes=len(selected_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(epochs):
        if status_callback:
            status_callback(f"Training epoch {epoch+1}/{epochs}...")

        model.train()
        running_loss = 0
        correct, total_data = 0, 0

        for img, label in train_loader:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct += (preds == label).sum().item()
            total_data += label.size(0)

        train_acc = correct / total_data

        # ================= VALIDATION =================
        model.eval()
        val_loss, correct, total_data = 0, 0, 0

        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                output = model(img)
                val_loss += criterion(output, label).item()

                preds = torch.argmax(output, dim=1)
                correct += (preds == label).sum().item()
                total_data += label.size(0)

        val_acc = correct / total_data

        history["train_loss"].append(running_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if progress_callback:
            progress_callback(epoch, epochs)

    # =========================
    # EVALUATION (TEST SET)
    # =========================
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            preds = torch.argmax(output, dim=1)

            y_true.extend(label.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(
        y_true,
        y_pred,
        target_names=selected_classes,
        output_dict=True
    )

    cm = confusion_matrix(y_true, y_pred)

    # =========================
    # RETURN LENGKAP
    # =========================
    return history, model, selected_classes, report, cm
