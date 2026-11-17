#!/usr/bin/env python3
"""Training pipeline for Markius v5 Temporal Convolutional Network (TCN).

Trains an invariant, causal 1D Dilated Residual Network on external tracking matches
(/workspace/shared/dataset) to output calibrated frame-level probabilities [P(H), P(A), P(D)].
Exports frozen weights to cache/markius_v5_tcn_weights.npz for pure NumPy inference.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starter.feature_extraction_v5 import (
    extract_features_from_dataset_npz,
    extract_features_from_competition,
    N_FEATURES,
)

LABEL_MAP = {"H": 0, "A": 1, "D": 2}


class CausalConv1d(nn.Module):
    """1D causal convolution with dilation ensuring no future information leakage."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.pad,
        )

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return out


class ResidualBlock(nn.Module):
    """Causal dilated residual block with LayerNorm and ReLU."""
    def __init__(self, channels=32, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + res)


class MarkiusTCN(nn.Module):
    """Lightweight 1D Dilated Residual Network with ~38,000 parameters and 255-frame (~10.2s) receptive field."""
    def __init__(self, in_features=16, hidden=32, n_classes=3):
        super().__init__()
        self.in_proj = nn.Conv1d(in_features, hidden, kernel_size=1)
        self.block1 = ResidualBlock(hidden, dilation=1)
        self.block2 = ResidualBlock(hidden, dilation=2)
        self.block3 = ResidualBlock(hidden, dilation=4)
        self.block4 = ResidualBlock(hidden, dilation=8)
        self.block5 = ResidualBlock(hidden, dilation=16)
        self.block6 = ResidualBlock(hidden, dilation=32)
        self.out_proj = nn.Conv1d(hidden, n_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 16, T)
        h = self.in_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        logits = self.out_proj(h)  # (B, 3, T)
        return logits


def load_match_data(npz_path):
    """Extract features and target integer labels (0=H, 1=A, 2=D, -1=X)."""
    feats, str_labels = extract_features_from_dataset_npz(npz_path)
    targets = np.full(len(str_labels), -1, dtype=np.int64)
    for k, v in LABEL_MAP.items():
        targets[str_labels == k] = v
    return feats, targets


class SequenceDataset:
    """Pre-loads matches and samples random continuous chunks of length T."""
    def __init__(self, file_paths, seq_len=256):
        self.seq_len = seq_len
        self.matches = []
        for p in file_paths:
            if not os.path.exists(p):
                continue
            feats, targets = load_match_data(p)
            self.matches.append((feats, targets))
        print(f"Loaded {len(self.matches)} matches. Total frames: {sum(len(m[0]) for m in self.matches):,}")

    def sample_batch(self, batch_size=32):
        batch_x = np.zeros((batch_size, N_FEATURES, self.seq_len), dtype=np.float32)
        batch_y = np.full((batch_size, self.seq_len), -1, dtype=np.int64)

        for b in range(batch_size):
            m_idx = np.random.randint(0, len(self.matches))
            feats, targets = self.matches[m_idx]
            max_start = len(feats) - self.seq_len
            if max_start <= 0:
                continue
            start = np.random.randint(0, max_start)
            end = start + self.seq_len

            # (seq_len, 16) -> (16, seq_len)
            bx = feats[start:end].T.copy()
            by = targets[start:end].copy()

            # Physical Team-Symmetric Data Augmentation: 50% probability
            # Swapping Team H <-> Team A enforces strict physical invariance
            if np.random.rand() > 0.5:
                # 4 (home_min_dist) <-> 5 (away_min_dist)
                bx[4], bx[5] = bx[5].copy(), bx[4].copy()
                # 6 (dist_margin = away - home) -> -dist_margin
                bx[6] = -bx[6]
                # 7 (home_min_kappa) <-> 8 (away_min_kappa)
                bx[7], bx[8] = bx[8].copy(), bx[7].copy()
                # 9 (p_voronoi_home) <-> 10 (p_voronoi_away)
                bx[9], bx[10] = bx[10].copy(), bx[9].copy()
                # 11 (n_home_3m) <-> 12 (n_away_3m)
                bx[11], bx[12] = bx[12].copy(), bx[11].copy()
                # 15 (norm_ball_x) -> -norm_ball_x
                bx[15] = -bx[15]

                # Target swap: 0 (H) <-> 1 (A), 2 (D) remains 2
                mask0 = by == 0
                mask1 = by == 1
                by[mask0] = 1
                by[mask1] = 0

            batch_x[b] = bx
            batch_y[b] = by

        return torch.from_numpy(batch_x), torch.from_numpy(batch_y)


def train_model(train_files, val_file, out_weights_path="cache/markius_v5_tcn_weights.npz", n_steps=1200):
    print("Initializing Markius TCN Model...")
    model = MarkiusTCN(in_features=N_FEATURES, hidden=32, n_classes=3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Markius TCN Model Parameters: {n_params:,}")

    print("Loading training dataset...")
    train_dataset = SequenceDataset(train_files, seq_len=256)

    print("Loading validation dataset...")
    val_feats, val_targets = load_match_data(val_file)
    val_x = torch.from_numpy(val_feats.T).unsqueeze(0)  # (1, 16, N)
    val_y = torch.from_numpy(val_targets).unsqueeze(0)  # (1, N)

    # Class weights to balance D (~42%) vs H/A (~29% each)
    class_weights = torch.tensor([1.2, 1.2, 0.8], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

    print(f"Starting TCN training ({n_steps} steps on 2 vCPUs)...")
    best_val_acc = 0.0
    start_time = time.time()

    model.train()
    for step in range(1, n_steps + 1):
        x_batch, y_batch = train_dataset.sample_batch(batch_size=32)
        optimizer.zero_grad()
        logits = model(x_batch)  # (B, 3, T)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 200 == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                # Validate on slice of val match (first 30k frames)
                v_slice_x = val_x[:, :, :30000]
                v_slice_y = val_y[:, :30000]
                val_logits = model(v_slice_x)
                val_loss = criterion(val_logits, v_slice_y).item()
                preds = torch.argmax(val_logits, dim=1)
                valid_mask = v_slice_y != -1
                acc = (preds[valid_mask] == v_slice_y[valid_mask]).float().mean().item()

            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{n_steps:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2%} | Time: {elapsed:.1f}s")
            best_val_acc = max(best_val_acc, acc)
            model.train()

    print(f"Training completed in {time.time() - start_time:.1f}s! Best Val Accuracy: {best_val_acc:.2%}")

    # Export weights for Pure NumPy Forward Pass
    export_weights_to_numpy(model, out_weights_path)


def export_weights_to_numpy(model, out_path):
    """Export PyTorch model weights to a NumPy archive for dependency-free inference."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    state_dict = model.state_dict()
    weights_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
    np.savez_compressed(out_path, **weights_dict)
    file_size_kb = os.path.getsize(out_path) / 1024.0
    print(f"Successfully exported frozen TCN weights to {out_path} ({file_size_kb:.1f} KB).")


if __name__ == "__main__":
    import glob
    all_dataset_matches = sorted(glob.glob("/workspace/shared/dataset/*.npz"))
    val_match = "/workspace/shared/dataset/metrica_game3.npz"
    train_matches = [p for p in all_dataset_matches if p != val_match]
    print(f"Found {len(train_matches)} training matches and holdout validation: {val_match}")
    train_model(train_matches, val_match, out_weights_path="cache/markius_v5_tcn_weights.npz", n_steps=1500)
