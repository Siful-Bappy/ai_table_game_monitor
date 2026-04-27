"""
MobileNet Card Classifier
=========================
Training and inference for the GKL card dataset.

Dataset layout (produced by augment_dataset.py):
    <dataset_dir>/
        train/  <class_name>/  aug_0000.jpg ...
        val/    <class_name>/  val_0000.jpg ...

Usage
-----
    # Step 1 – compute & cache dataset mean/std (run once)
    python mobilenet_card_classifier.py stats  dataset_gkl_cards/augmented

    # Step 2 – train
    python mobilenet_card_classifier.py train  dataset_gkl_cards/augmented

    # Step 3 – evaluate on val folder
    python mobilenet_card_classifier.py test   dataset_gkl_cards/augmented/val

    # Predict a single image
    python mobilenet_card_classifier.py test   path/to/card.jpg
"""

import os
import json
import random
import pickle
import shutil

import cv2
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import timm


NUM_CLASSES = 53
WORKING_DIR = "./working"
STATS_PATH  = os.path.join(WORKING_DIR, "dataset_stats.json")


# -----------------------------------------------------------------------
# Dataset statistics  (Step 1 — run once before training)
# -----------------------------------------------------------------------

def compute_dataset_stats(dataset_dir, max_samples=3000):
    """
    Compute per-channel mean and std from training images (BGR→RGB, [0,1]).
    Saves result to STATS_PATH so it only needs to run once.

    Returns
    -------
    mean, std : list[float] — three values each, in [0, 1]
    """
    img_paths = glob(os.path.join(dataset_dir, "train/*/*.jpg"))
    if not img_paths:
        raise ValueError(f"No training images found under {dataset_dir}/train/")

    random.shuffle(img_paths)
    img_paths = img_paths[:max_samples]

    mean = np.zeros(3, dtype=np.float64)
    sq   = np.zeros(3, dtype=np.float64)
    n    = 0

    print(f"Computing dataset stats from {len(img_paths)} images ...")
    for path in tqdm(img_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean += img.mean(axis=(0, 1))
        sq   += (img ** 2).mean(axis=(0, 1))
        n    += 1

    mean /= n
    std   = np.sqrt(np.maximum(sq / n - mean ** 2, 0))

    stats = {"mean": mean.tolist(), "std": std.tolist(), "n_samples": n}
    os.makedirs(WORKING_DIR, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"  Mean : {[round(v, 4) for v in stats['mean']]}")
    print(f"  Std  : {[round(v, 4) for v in stats['std']]}")
    print(f"  Saved → {STATS_PATH}")
    return stats["mean"], stats["std"]


def load_dataset_stats():
    """
    Load cached mean/std from STATS_PATH.
    Falls back to ImageNet defaults if the file does not exist yet.
    """
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            s = json.load(f)
        print(f"Loaded stats from {STATS_PATH}")
        print(f"  Mean : {[round(v, 4) for v in s['mean']]}")
        print(f"  Std  : {[round(v, 4) for v in s['std']]}")
        return s["mean"], s["std"]

    print(f"WARNING: {STATS_PATH} not found — run 'stats' command first.")
    print("Falling back to ImageNet defaults.")
    # return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return [0.687, 0.726, 0.766], [0.338, 0.310, 0.264]



# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "mobilenetv1_125.ra4_e3600_r224_in1k",
            num_classes=NUM_CLASSES,
            global_pool="avg",
            pretrained=True,
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------------------------------------------------
# Dataset & DataLoader
# -----------------------------------------------------------------------

class MyDataset(Dataset):
    """
    Loads JPEG images from a folder tree:  <root>/<class_name>/<img>.jpg
    Class index is derived from class_list (must match card_labels.json order).
    """
    def __init__(self, img_paths, class_list, tf=None):
        self.paths      = img_paths
        self.card_class = class_list
        self.transform  = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path     = self.paths[idx]
        img      = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        cls_name = os.path.basename(os.path.dirname(path))
        label    = self.card_class.index(cls_name)   # sort order must match!
        return img, label


class DeviceDL:
    """Wraps a DataLoader and moves every batch to GPU/CPU automatically."""
    def __init__(self, dl):
        self.dl     = dl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to(self, data):
        if isinstance(data, (list, tuple)):
            return [self._to(x) for x in data]
        return data.to(self.device)

    def __iter__(self):
        for batch in self.dl:
            yield self._to(batch)

    def __len__(self):
        return len(self.dl)


# -----------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------

class TrainHelper:
    def __init__(self, save_path=os.path.join(WORKING_DIR, "history.pickle"),
                 history=None):
        self.history   = history or []
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def accuracy(self, outputs, labels):
        pred = torch.argmax(outputs, dim=1)
        return torch.tensor(torch.sum(pred == labels).item() / len(pred))

    @torch.no_grad()
    def _validate_batch(self, model, batch):
        imgs, labels = batch
        out  = model(imgs)
        return {
            "val_loss": F.cross_entropy(out, labels).detach(),
            "val_acc":  self.accuracy(out, labels),
        }

    @torch.no_grad()
    def evaluation(self, model, loader):
        model.eval()
        results    = [self._validate_batch(model, b) for b in loader]
        epoch_loss = torch.stack([r["val_loss"] for r in results]).mean()
        epoch_acc  = torch.stack([r["val_acc"]  for r in results]).mean()
        return {
            "val_loss": round(epoch_loss.item(), 5),
            "val_acc":  round(epoch_acc.item(),  5),
        }

    def logging(self, epoch, result):
        print("Epoch {:>3}:  train_loss={:.4f}  val_loss={:.4f}  val_acc={:.4f}".format(
            epoch, result["train_loss"], result["val_loss"], result["val_acc"]))
        self.history.append(result)
        with open(self.save_path, "wb") as f:
            pickle.dump(self.history, f)


# -----------------------------------------------------------------------
# Train  (Step 2)
# -----------------------------------------------------------------------

def train(dataset_dir, checkpoint_path=None, epochs=100, batch_size=32, lr=1e-3):
    """
    Train MobileNet on the augmented GKL card dataset.

    Parameters
    ----------
    dataset_dir     : folder containing train/ and val/ sub-folders
    checkpoint_path : optional .pt file to resume training from
    epochs          : training epochs
    batch_size      : mini-batch size
    lr              : initial learning rate (Adam + CosineAnnealing)
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}\n")

    # ── Labels ──────────────────────────────────────────────────────────
    with open("card_labels.json", "r", encoding="utf-8") as f:
        card_label_list = json.load(f)
    print(f"Classes : {len(card_label_list)}")

    # ── Dataset mean / std ──────────────────────────────────────────────
    mean, std = load_dataset_stats()

    # ── Model ───────────────────────────────────────────────────────────
    model = MobileNet()
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Resumed from checkpoint: {checkpoint_path}")
    else:
        print("Starting fresh (pretrained ImageNet backbone).")
    model.to(DEVICE)

    # ── Transforms ──────────────────────────────────────────────────────
    # Images are already 224×224 from augment_dataset.py.
    # Resize to 256 first so RandomCrop gives spatial diversity.
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Image paths ─────────────────────────────────────────────────────
    train_paths = glob(os.path.join(dataset_dir, "train/*/*.jpg"))
    val_paths   = glob(os.path.join(dataset_dir, "val/*/*.jpg"))

    if not train_paths:
        raise ValueError(f"No images found in {dataset_dir}/train/")
    if not val_paths:
        raise ValueError(f"No images found in {dataset_dir}/val/")

    print(f"Train images : {len(train_paths)}")
    print(f"Val   images : {len(val_paths)}\n")

    # ── DataLoaders ─────────────────────────────────────────────────────
    num_workers = 4 if os.name != "nt" else 0   # 0 on Windows
    train_dl = DeviceDL(DataLoader(
        MyDataset(train_paths, card_label_list, tf=train_tf),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    ))
    val_dl = DeviceDL(DataLoader(
        MyDataset(val_paths, card_label_list, tf=val_tf),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    ))

    # ── Optimiser + scheduler ───────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(WORKING_DIR, exist_ok=True)
    helper       = TrainHelper()
    best_acc     = 0.0
    best_ckpt    = ""

    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", leave=False):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach())

        scheduler.step()

        result = helper.evaluation(model, val_dl)
        result["train_loss"] = torch.stack(train_losses).mean().item()

        if result["val_acc"] > best_acc:
            best_acc  = result["val_acc"]
            best_ckpt = os.path.join(WORKING_DIR, f"best_ep{epoch}_{best_acc:.4f}.pt")
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ New best → {best_ckpt}")

        helper.logging(epoch, result)

    # Copy best weights as canonical best.pt
    final_path = os.path.join(WORKING_DIR, "best.pt")
    if best_ckpt:
        shutil.copy(best_ckpt, final_path)
        print(f"\nBest model saved → {final_path}  (val_acc={best_acc:.4f})")


# -----------------------------------------------------------------------
# Inference / Export API
# -----------------------------------------------------------------------

def load_model(checkpoint_path=None, train_mode=False):
    """Load MobileNet weights. Creates a fresh model if checkpoint not found."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MobileNet()

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"Loaded weights from {checkpoint_path}")
    elif not train_mode:
        print(f"WARNING: checkpoint not found at '{checkpoint_path}'. Weights are random.")

    model.to(DEVICE)
    if not train_mode:
        model.eval()
    return model


def load_image_preprocessor():
    """Return the inference transform using cached dataset mean/std."""
    mean, std = load_dataset_stats()
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def classify_card(model, preprocessor, image_or_path):
    """
    Classify a single card image.

    Parameters
    ----------
    image_or_path : str (file path) | np.ndarray (BGR from cv2) | PIL.Image

    Returns
    -------
    (predicted_index, confidence_float)
    """
    device = next(model.parameters()).device

    if isinstance(image_or_path, str):
        img = Image.open(image_or_path).convert("RGB")
    elif isinstance(image_or_path, np.ndarray):
        # cv2 delivers BGR — convert to RGB for PIL / torchvision
        img = Image.fromarray(cv2.cvtColor(image_or_path, cv2.COLOR_BGR2RGB))
    else:
        img = image_or_path  # assume PIL Image already

    tensor = preprocessor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        idx   = torch.argmax(probs, dim=1).item()
    return idx, probs[0, idx].item()


def classify_cards(model, preprocessor, cards):
    """
    Classify a batch of card images (numpy BGR from cv2).

    Returns
    -------
    indices    : torch.Tensor shape (N,)
    confidence : torch.Tensor shape (N,)
    """
    device = next(model.parameters()).device
    batch  = torch.stack([
        preprocessor(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        for im in cards
    ]).to(device)

    with torch.no_grad():
        probs_all  = torch.softmax(model(batch), dim=1)
        indices    = torch.argmax(probs_all, dim=1)
        confidence = torch.max(probs_all, dim=1).values

    return indices.cpu(), confidence.cpu()


def test_all(model, dataset_dir, card_label_list):
    """Evaluate model accuracy on all images under dataset_dir/*/*.jpg."""
    preprocessor = load_image_preprocessor()
    test_paths   = glob(os.path.join(dataset_dir, "*/*.jpg"))
    if not test_paths:
        print(f"No images found under {dataset_dir}")
        return

    test_dl = DeviceDL(DataLoader(
        MyDataset(test_paths, card_label_list, tf=preprocessor),
        batch_size=64, shuffle=False,
    ))
    result = TrainHelper().evaluation(model, test_dl)
    print(f"Test result  :  val_loss={result['val_loss']}  val_acc={result['val_acc']}")


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def _usage():
        print("Usage:")
        print("  python mobilenet_card_classifier.py stats  <dataset_dir>")
        print("  python mobilenet_card_classifier.py train  <dataset_dir>")
        print("  python mobilenet_card_classifier.py test   <dataset_dir/val  |  image.jpg>")
        sys.exit(1)

    if len(sys.argv) < 3:
        _usage()

    cmd = sys.argv[1]

    if cmd == "stats":
        compute_dataset_stats(sys.argv[2])

    elif cmd == "train":
        dataset_dir     = sys.argv[2]
        checkpoint_path = os.path.join(WORKING_DIR, "best.pt")
        # Auto-compute stats if not cached yet
        if not os.path.exists(STATS_PATH):
            print("Stats not found — computing now ...\n")
            compute_dataset_stats(dataset_dir)
        train(dataset_dir, checkpoint_path=checkpoint_path)

    elif cmd == "test":
        with open("card_labels.json", "r", encoding="utf-8") as f:
            card_label_list = json.load(f)
        checkpoint_path = os.path.join(WORKING_DIR, "best.pt")
        model = load_model(checkpoint_path)

        target = sys.argv[2]
        if os.path.isdir(target):
            test_all(model, target, card_label_list)
        else:
            preprocessor    = load_image_preprocessor()
            idx, confidence = classify_card(model, preprocessor, target)
            print(f"Prediction : {card_label_list[idx]}  (confidence {confidence:.3f})")
            img = cv2.imread(target)
            cv2.imshow("card", img)
            cv2.waitKey(0)

    else:
        _usage()
