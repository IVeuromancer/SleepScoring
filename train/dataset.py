from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset

def create_dataloaders(X_tensor, y_tensor, EEG_files):
    num_mouse = len(EEG_files)
    epochs_per_mouse = X_tensor.shape[0] // (num_mouse * 3)
    datasets = [TensorDataset(X_tensor[i*epochs_per_mouse:(i+1)*epochs_per_mouse],
                              y_tensor[i*epochs_per_mouse:(i+1)*epochs_per_mouse])
                for i in range(num_mouse * 3)]

    train_sets, val_sets, test_sets = [], [], []
    for d in datasets:
        n = len(d)
        tr, va = int(n*0.7), int(n*0.15)
        train_sets.append(Subset(d, range(tr)))
        val_sets.append(Subset(d, range(tr, tr+va)))
        test_sets.append(Subset(d, range(tr+va, n)))

    return (
        DataLoader(ConcatDataset(train_sets), batch_size=256, shuffle=False),
        DataLoader(ConcatDataset(val_sets), batch_size=256, shuffle=False),
        DataLoader(ConcatDataset(test_sets), batch_size=256, shuffle=False),
    )
