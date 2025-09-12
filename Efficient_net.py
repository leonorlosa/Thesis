import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tifffile as tiff

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef

import optuna

# --------------------
# Config
# --------------------
device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

NUM_CLASSES = 2
IMG_SIZE = 224
SEEDS = [53, 65, 88]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --------------------
# Dataset
# --------------------
class TiffDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = tiff.imread(img_path).astype(np.float32)

        # Expect 2D or 3D with a single channel; reduce to 2D tensor
        if image.ndim == 3:
            if 1 in image.shape:
                image = image.squeeze()
            else:
                image = image[0]
        if image.ndim != 2:
            raise ValueError(f"Unexpected image shape {image.shape} for {img_path} (need 2D or 3D single-channel).")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# --------------------
# Preprocessing helpers
# --------------------
def calc_dataset_mean_std(paths, sample_size=2000):
    sample = random.sample(paths, min(sample_size, len(paths)))
    means, stds = [], []
    for p in sample:
        try:
            im = tiff.imread(p).astype(np.float32)
            if im.ndim == 3:
                if 1 in im.shape:
                    im = im.squeeze()
                else:
                    im = im[0]
            if im.ndim != 2:
                continue
            means.append(im.mean())
            stds.append(im.std())
        except Exception:
            continue
    if not means:
        return 0.0, 1.0
    mean_val, std_val = float(np.mean(means)), float(np.mean(stds))
    return mean_val, max(std_val, 1e-9)

# --------------------
# Model
# --------------------
def get_efficientnet_b0(num_classes=NUM_CLASSES, dropout_p1=0.4, dropout_p2=0.6):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    # Change first conv to 1 input channel
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    nn.init.normal_(model.features[0][0].weight, mean=0.0, std=0.001)

    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p1),
        nn.Linear(in_features, 640),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p2),
        nn.Linear(640, num_classes),
    )
    return model

# --------------------
# Training (one fold)
# --------------------
def train_fold(model, train_loader, val_loader, epochs, patience, criterion, optimizer, scheduler, fold_num, output_dir, class_names):
    best_val_loss = float('inf')
    patience_ctr = 0
    best_model_path = os.path.join(output_dir, f'best_model_fold_{fold_num}.pth')

    best_metrics = {
        "best_val_acc": 0.0,
        "best_val_f1": 0.0,
        "best_val_mcc": 0.0,
        "val_report": None
    }

    for epoch in range(epochs):
        # Train
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        # Validate
        model.eval()
        v_correct, v_total, v_loss = 0, 0, 0.0
        gts, preds = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                _, pred = out.max(1)
                v_total += y.size(0)
                v_correct += (pred == y).sum().item()
                preds.extend(pred.cpu().numpy())
                gts.extend(y.cpu().numpy())

        val_loss = v_loss / len(val_loader.dataset)
        val_acc = 100.0 * v_correct / v_total if v_total > 0 else 0.0
        val_f1 = 100.0 * f1_score(gts, preds, average='weighted', zero_division=0) if len(np.unique(gts)) >= 2 else 0.0
        val_mcc = matthews_corrcoef(gts, preds) if len(np.unique(gts)) >= 2 else 0.0

        scheduler.step()

        if (epoch % 5 == 0) or (epoch == epochs - 1):
            print(f"[Fold {fold_num}] Epoch {epoch+1}/{epochs} | "
                  f"TrainLoss: {running_loss/len(train_loader.dataset):.4f} | "
                  f"ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.2f}% | F1: {val_f1:.2f}% | MCC: {val_mcc:.3f}")

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), best_model_path)
            best_metrics.update({
                "best_val_acc": val_acc,
                "best_val_f1": val_f1,
                "best_val_mcc": val_mcc,
                "val_report": classification_report(gts, preds, target_names=class_names, zero_division=0)
            })
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[Fold {fold_num}] Early stopping at epoch {epoch+1}.")
                break

    # Load best weights for this fold
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model, best_metrics

# --------------------
# Cross-validation
# --------------------
def run_cross_validation(paths, labels, n_splits, epochs, class_names, params, output_dir):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEEDS[0])
    fold_summaries = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), start=1):
        tr_paths = [paths[i] for i in tr_idx]
        va_paths = [paths[i] for i in va_idx]
        tr_lbls = [labels[i] for i in tr_idx]
        va_lbls = [labels[i] for i in va_idx]

        norm_mean, norm_std = calc_dataset_mean_std(tr_paths)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.Normalize(mean=[norm_mean], std=[norm_std]),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            transforms.Normalize(mean=[norm_mean], std=[norm_std]),
        ])

        ds_tr = TiffDataset(tr_paths, tr_lbls, transform=train_transform)
        ds_va = TiffDataset(va_paths, va_lbls, transform=val_transform)

        class_freq = Counter(tr_lbls)
        w_per_class = {c: 1.0 / n for c, n in class_freq.items()}
        sample_weights = [w_per_class[l] for l in tr_lbls]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_lbls), replacement=True)

        ld_tr = DataLoader(ds_tr, batch_size=16, sampler=sampler, num_workers=0)
        ld_va = DataLoader(ds_va, batch_size=16, shuffle=False, num_workers=0)

        model = get_efficientnet_b0(dropout_p1=params['dropout_p1'], dropout_p2=params['dropout_p2']).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        model, best_metrics = train_fold(
            model, ld_tr, ld_va, epochs=epochs, patience=15,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            fold_num=fold, output_dir=output_dir, class_names=class_names
        )

        fold_summaries.append(best_metrics)
        print(f"[Fold {fold}] Best @ min ValLoss -> "
              f"Acc: {best_metrics['best_val_acc']:.2f}%, "
              f"F1: {best_metrics['best_val_f1']:.2f}%, "
              f"MCC: {best_metrics['best_val_mcc']:.3f}")

    return fold_summaries

# --------------------
# Evaluation on test set
# --------------------
def evaluate_single_model_on_test_set(model, test_paths, test_labels, class_names, train_val_paths):
    mean_val, std_val = calc_dataset_mean_std(train_val_paths)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.Normalize(mean=[mean_val], std=[std_val]),
    ])

    ds_test = TiffDataset(test_paths, test_labels, transform=test_transform)
    loader = DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=0)

    model.eval()
    gts, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            gts.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())

    report = classification_report(gts, preds, target_names=class_names, zero_division=0)
    acc = 100.0 * (np.array(gts) == np.array(preds)).mean()
    f1 = 100.0 * f1_score(gts, preds, average='weighted', zero_division=0) if len(np.unique(gts)) >= 2 else 0.0
    mcc = matthews_corrcoef(gts, preds) if len(np.unique(gts)) >= 2 else 0.0

    print("\n--- Single Model Test Report ---")
    print(report)
    print(f"Accuracy: {acc:.2f}% | Weighted F1: {f1:.2f}% | MCC: {mcc:.3f}")

    return {"accuracy": acc, "f1": f1, "mcc": mcc}

def evaluate_ensemble_on_test_set(models, test_paths, test_labels, class_names, train_val_paths):
    mean_val, std_val = calc_dataset_mean_std(train_val_paths)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.Normalize(mean=[mean_val], std=[std_val]),
    ])

    ds_test = TiffDataset(test_paths, test_labels, transform=test_transform)
    loader = DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=0)

    for m in models:
        m.eval()

    gts, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.zeros((x.size(0), NUM_CLASSES), device=device)
            for m in models:
                probs += torch.softmax(m(x), dim=1)
            probs /= len(models)
            _, pred = probs.max(1)
            gts.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())

    report = classification_report(gts, preds, target_names=class_names, zero_division=0)
    acc = 100.0 * (np.array(gts) == np.array(preds)).mean()
    f1 = 100.0 * f1_score(gts, preds, average='weighted', zero_division=0) if len(np.unique(gts)) >= 2 else 0.0
    mcc = matthews_corrcoef(gts, preds) if len(np.unique(gts)) >= 2 else 0.0

    print("\n--- Ensemble Test Report ---")
    print(report)
    print(f"Accuracy: {acc:.2f}% | Weighted F1: {f1:.2f}% | MCC: {mcc:.3f}")

    return {"accuracy": acc, "f1": f1, "mcc": mcc}

# --------------------
# Optuna objective (quick split inside)
# --------------------
def objective(trial, train_paths, train_labels):
    params = {
        "lr": trial.suggest_float("lr", 2e-4, 1.5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-4, 5e-2, log=True),
        "dropout_p1": trial.suggest_float("dropout_p1", 0.25, 0.48),
        "dropout_p2": trial.suggest_float("dropout_p2", 0.18, 0.55),
    }

    tr_p, va_p, tr_l, va_l = train_test_split(
        train_paths, train_labels, test_size=0.25, random_state=SEEDS[0], stratify=train_labels
    )

    mean_val, std_val = calc_dataset_mean_std(tr_p)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize([mean_val], [std_val]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        transforms.Normalize([mean_val], [std_val]),
    ])

    ds_tr = TiffDataset(tr_p, tr_l, transform=train_transform)
    ds_va = TiffDataset(va_p, va_l, transform=val_transform)

    w_per_class = {c: 1.0 / n for c, n in Counter(tr_l).items()}
    sample_weights = [w_per_class[l] for l in tr_l]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_l), replacement=True)

    ld_tr = DataLoader(ds_tr, batch_size=16, sampler=sampler, num_workers=0)
    ld_va = DataLoader(ds_va, batch_size=16, shuffle=False, num_workers=0)

    model = get_efficientnet_b0(dropout_p1=params["dropout_p1"], dropout_p2=params["dropout_p2"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    model.train()
    for _ in range(15):  # short warmup for speed
        for x, y in ld_tr:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    gts, preds = [], []
    with torch.no_grad():
        for x, y in ld_va:
            x, y = x.to(device), y.to(device)
            _, pred = model(x).max(1)
            gts.extend(y.cpu().numpy())
            preds.extend(pred.cpu().numpy())

    return matthews_corrcoef(gts, preds) if len(np.unique(gts)) >= 2 else 0.0

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    base_data_path = Path("C:/Users/ExtreMedSyncRGB/Documents/Leonor")
    cell_dirs = [base_data_path / "MIP control", base_data_path / "MIP IC50"]
    class_names = ["A549 (Control)", "A549 + PTX"]

    paths, labels = [], []
    for lbl, d in enumerate(cell_dirs):
        imgs = list(d.glob("*.tif"))
        paths.extend(imgs)
        labels.extend([lbl] * len(imgs))
        print(f"Found {len(imgs)} images for class '{class_names[lbl]}'")
    if not paths:
        raise RuntimeError("No images found. Check 'base_data_path' and 'cell_dirs'.")

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=SEEDS[0], stratify=labels
    )
    print(f"\nTotal: {len(paths)} | Train/Val: {len(train_val_paths)} | Test: {len(test_paths)}")

    TOTAL_EPOCHS_FOR_CV = 50
    N_SPLITS = 5
    N_TRIALS = 50

    all_runs_single = []
    all_runs_ensemble = []

    for i, seed in enumerate(SEEDS, start=1):
        print(f"\n{'#'*20} RUN {i}/{len(SEEDS)} | Seed {seed} {'#'*20}")
        set_seed(seed)
        output_dir = f"run_{i}_seed{seed}"
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Optuna
        print("\n==> Hyperparameter search (Optuna)")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda trial: objective(trial, train_val_paths, train_val_labels), n_trials=N_TRIALS)
        best_params = study.best_params
        print(f"Best params: {best_params}")

        # Step 2: Cross-validation
        print("\n==> K-fold cross-validation")
        fold_summaries = run_cross_validation(
            train_val_paths, train_val_labels, n_splits=N_SPLITS, epochs=TOTAL_EPOCHS_FOR_CV,
            class_names=class_names, params=best_params, output_dir=output_dir
        )

        # Fold stats
        accs = [f["best_val_acc"] for f in fold_summaries]
        f1s = [f["best_val_f1"] for f in fold_summaries]
        mccs = [f["best_val_mcc"] for f in fold_summaries]
        print("\nCV summary (best @ min ValLoss across folds):")
        print(f"Acc: {np.mean(accs):.2f} ± {np.std(accs):.2f}%")
        print(f"F1 : {np.mean(f1s):.2f} ± {np.std(f1s):.2f}%")
        print(f"MCC: {np.mean(mccs):.3f} ± {np.std(mccs):.3f}")

        # Step 3: Single best fold -> test
        best_fold_idx = int(np.argmax(accs)) + 1
        best_path = os.path.join(output_dir, f"best_model_fold_{best_fold_idx}.pth")
        print(f"\n==> Single-model test (best fold: {best_fold_idx})")
        if os.path.exists(best_path):
            single_model = get_efficientnet_b0(dropout_p1=best_params["dropout_p1"], dropout_p2=best_params["dropout_p2"]).to(device)
            single_model.load_state_dict(torch.load(best_path, map_location=device))
            sm = evaluate_single_model_on_test_set(single_model, test_paths, test_labels, class_names, train_val_paths)
        else:
            print("Best fold model not found; skipping single-model test.")
            sm = {"accuracy": np.nan, "f1": np.nan, "mcc": np.nan}
        all_runs_single.append(sm)

        # Step 4: Ensemble (all folds) -> test
        print("\n==> Ensemble test (avg of fold models)")
        ensemble_models = []
        for f in range(1, N_SPLITS + 1):
            p = os.path.join(output_dir, f"best_model_fold_{f}.pth")
            if os.path.exists(p):
                m = get_efficientnet_b0(dropout_p1=best_params["dropout_p1"], dropout_p2=best_params["dropout_p2"]).to(device)
                m.load_state_dict(torch.load(p, map_location=device))
                ensemble_models.append(m)
        if ensemble_models:
            ens = evaluate_ensemble_on_test_set(ensemble_models, test_paths, test_labels, class_names, train_val_paths)
        else:
            print("No fold models found; skipping ensemble test.")
            ens = {"accuracy": np.nan, "f1": np.nan, "mcc": np.nan}
        all_runs_ensemble.append(ens)

    # Final summary across runs
    def _summ(metric_list, key):
        vals = [m[key] for m in metric_list if not np.isnan(m[key])]
        return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))

    sm_acc_mean, sm_acc_std = _summ(all_runs_single, "accuracy")
    sm_f1_mean, sm_f1_std = _summ(all_runs_single, "f1")
    sm_mcc_mean, sm_mcc_std = _summ(all_runs_single, "mcc")

    en_acc_mean, en_acc_std = _summ(all_runs_ensemble, "accuracy")
    en_f1_mean, en_f1_std = _summ(all_runs_ensemble, "f1")
    en_mcc_mean, en_mcc_std = _summ(all_runs_ensemble, "mcc")

    print("\n================ FINAL SUMMARY (across seeds) ================")
    print(f"Single model -> Acc: {sm_acc_mean:.2f} ± {sm_acc_std:.2f}% | "
          f"F1: {sm_f1_mean:.2f} ± {sm_f1_std:.2f}% | MCC: {sm_mcc_mean:.3f} ± {sm_mcc_std:.3f}")
    print(f"Ensemble     -> Acc: {en_acc_mean:.2f} ± {en_acc_std:.2f}% | "
          f"F1: {en_f1_mean:.2f} ± {en_f1_std:.2f}% | MCC: {en_mcc_mean:.3f} ± {en_mcc_std:.3f}")