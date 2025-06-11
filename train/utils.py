import torch
import matplotlib.pyplot as plt

def compute_class_weights(label_counts, total, device):
    weights = [1.0 / (label_counts[i] / total) for i in range(3)]
    return torch.tensor(weights, dtype=torch.float32, device=device)

def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()
