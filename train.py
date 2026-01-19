import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data_loader import GraphSequenceDataset, collate_fn
from model_gcn_lstm import GCN_LSTM_Model


# CONFIG
DATA_DIR = "/home/member2/Dastin/Program/CREMA-D-NPZ"
BATCH_SIZE = 64
EPOCHS = 1000
LR = 1e-4
KFOLD = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
PLOT_DIR = "plots"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

label_map = {"ANG":0,"DIS":1,"FEA":2,"HAP":3,"NEU":4,"SAD":5}


# ACTOR-BASED TEST SPLIT
all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".npz")]
actors = sorted({os.path.basename(f).split("_")[0] for f in all_files})

random.shuffle(actors)
test_actors = actors[:int(0.2 * len(actors))]
trainval_actors = actors[int(0.2 * len(actors)):]

trainval_files = [f for f in all_files if os.path.basename(f).split("_")[0] in trainval_actors]
test_files     = [f for f in all_files if os.path.basename(f).split("_")[0] in test_actors]

print(f"Actors → Train+Val: {len(trainval_actors)} | Test: {len(test_actors)}")


# CHECKPOINT
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_f1):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_f1": best_f1
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_f1"]


# TRAIN/EVAL 
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, preds, gts = 0, [], []

    with torch.set_grad_enabled(is_train):
        for b in loader:
            xs = [x.to(DEVICE) for x in b["x"]]
            edge = b["edge_index"].to(DEVICE)
            y = b["label"].to(DEVICE)

            out = model(xs, edge)
            loss = criterion(out, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds += out.argmax(1).cpu().tolist()
            gts += y.cpu().tolist()

    total_loss /= len(gts)
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="macro")

    return total_loss, acc, f1


# PLOT FUNCTION
def save_plot(history, fold):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title(f"Loss (Fold {fold})")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.title(f"Accuracy (Fold {fold})")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fold_{fold}.png", dpi=200)
    plt.close()


# CROSS VALIDATION
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
best_model_state = None
best_f1_global = -1

for fold, (tr_idx, val_idx) in enumerate(kf.split(trainval_files), start=1):
    print(f"\n========== FOLD {fold} ==========")

    tr_files = [trainval_files[i] for i in tr_idx]
    val_files = [trainval_files[i] for i in val_idx]

    train_loader = DataLoader(GraphSequenceDataset(tr_files, label_map),
                              BATCH_SIZE, True, collate_fn=collate_fn)
    val_loader   = DataLoader(GraphSequenceDataset(val_files, label_map),
                              BATCH_SIZE, False, collate_fn=collate_fn)

    model = GCN_LSTM_Model(468, 32, 6).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc","train_f1","val_f1"]}

    csv_path = f"{LOG_DIR}/fold_{fold}.csv"
    ckpt_path = f"{CHECKPOINT_DIR}/fold_{fold}.pt"
    start_epoch, best_f1 = 0, -1

    if os.path.exists(ckpt_path):
        start_epoch, best_f1 = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        start_epoch += 1
        print(f"✔ Resume from epoch {start_epoch}")

        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for r in csv.DictReader(f):
                    for k in history:
                        history[k].append(float(r[k]))

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(history.keys())

    for epoch in range(start_epoch, EPOCHS):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1 = run_epoch(model, val_loader, criterion)

        for k,v in zip(history.keys(), [tr_loss,vl_loss,tr_acc,vl_acc,tr_f1,vl_f1]):
            history[k].append(v)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([history[k][-1] for k in history])

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, best_f1)

            if vl_f1 > best_f1_global:
                best_f1_global = vl_f1
                best_model_state = model.state_dict()

        scheduler.step()
        save_plot(history, fold)

        print(
            f"Epoch {epoch+1:04d}\n"
            f" Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Train F1: {tr_f1:.4f}\n"
            f" Val   Loss: {vl_loss:.4f} | Val   Acc: {vl_acc:.4f} | Val   F1: {vl_f1:.4f}"
        )


# FINAL TEST
print("\n=== FINAL TEST EVALUATION ===")
model.load_state_dict(best_model_state)
model.eval()

test_loader = DataLoader(GraphSequenceDataset(test_files, label_map),
                         BATCH_SIZE, False, collate_fn=collate_fn)

preds, gts = [], []
with torch.no_grad():
    for b in test_loader:
        xs = [x.to(DEVICE) for x in b["x"]]
        edge = b["edge_index"].to(DEVICE)
        y = b["label"].to(DEVICE)
        out = model(xs, edge)
        preds += out.argmax(1).cpu().tolist()
        gts += y.cpu().tolist()

print(classification_report(gts, preds, digits=4))

cm = confusion_matrix(gts, preds)
np.savetxt("confusion_matrix.csv", cm, delimiter=",", fmt="%d")

test_acc = accuracy_score(gts, preds)
test_f1  = f1_score(gts, preds, average="macro")

print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Macro-F1 : {test_f1:.4f}")