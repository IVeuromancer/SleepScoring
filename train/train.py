# sleep_scoring/train.py

from model import SleepScoringModel
from preprocessing import load_and_preprocess_data
from dataset import create_dataloaders
from utils import compute_class_weights, plot_training_results

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np
from sklearn.metrics import accuracy_score

# ---- Load Data ----
X_tensor, y_tensor, EEG_files = load_and_preprocess_data()
train_loader, val_loader, test_loader = create_dataloaders(X_tensor, y_tensor, EEG_files)

# ---- Training Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SleepScoringModel().to(device)

label_counts = Counter(y_tensor.numpy())
total = len(train_loader.dataset)
weights = compute_class_weights(label_counts, total, device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0003)

# ---- Training Loop ----
epochs = 100
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        loss_sum += loss.item() * xb.size(0)

    acc = correct / total
    avg_loss = loss_sum / total
    train_losses.append(avg_loss)
    train_accs.append(acc)

    # Validation
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            pred = out.argmax(1)
            preds.extend(pred.cpu().numpy())
            labels.extend(yb.cpu().numpy())
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            loss_sum += loss.item() * xb.size(0)

    acc_val = correct / total
    avg_loss_val = loss_sum / total
    val_losses.append(avg_loss_val)
    val_accs.append(acc_val)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {avg_loss_val:.4f} | Train Acc: {acc:.4f} | Val Acc: {acc_val:.4f}")
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:", cm.diagonal() / cm.sum(axis=1))

plot_training_results(train_losses, val_losses, train_accs, val_accs)

# ---- Testing Evaluation ----
model.eval()
all_pred_labels = []
all_true_labels = []

with torch.no_grad():
    for batch_X_test, batch_y_test in test_loader:
        batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device)
        outputs_test = model(batch_X_test)
        _, predicted_test = torch.max(outputs_test, 1)
        all_pred_labels.extend(predicted_test.cpu().numpy())
        all_true_labels.extend(batch_y_test.cpu().numpy())

all_pred_labels = np.array(all_pred_labels)
all_true_labels = np.array(all_true_labels)

accuracy_overall = accuracy_score(all_true_labels, all_pred_labels)
print(f"Overall Testing Accuracy: {accuracy_overall:.4f}")

for i in range(3):
    class_accuracy = accuracy_score(all_true_labels == i, all_pred_labels == i)
    print(f"Class {i} Accuracy: {class_accuracy:.4f}")

# ---- Save Model ----
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "../saved_models/sleep_scoring_model.pth")
print("Model saved to saved_models/sleep_scoring_model.pth")
