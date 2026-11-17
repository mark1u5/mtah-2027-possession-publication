#!/usr/bin/env python3
"""Train TCN with Boundary-Weighted Loss and Long-Range Context (512 frames).

Enhancements:
1. Boundary-Aware Loss: Transitions (changes in GT label) are weighted 4x higher within a +/- 12-frame window (~0.5s),
   forcing the TCN to learn sharp temporal transition boundaries rather than smearing.
2. Long-Range Sequence Context: seq_len increased to 512 frames (~20.5s), covering entire possession spells.
3. Team-Symmetric Data Augmentation: 50% chance of local/visitor channel swapping.
"""

import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starter.feature_extraction_v5 import extract_features_from_dataset_npz, N_FEATURES
from starter.train_tcn_v5 import MarkiusTCN, load_match_data, export_weights_to_numpy


class BoundarySequenceDataset:
    def __init__(self, file_paths, seq_len=512):
        self.seq_len = seq_len
        self.matches = []
        for p in file_paths:
            if not os.path.exists(p):
                continue
            feats, targets = load_match_data(p)
            self.matches.append((feats, targets))
        print(f"Loaded {len(self.matches)} matches for Boundary TCN. Total frames: {sum(len(m[0]) for m in self.matches):,}")

    def sample_batch(self, batch_size=32):
        batch_x = np.zeros((batch_size, N_FEATURES, self.seq_len), dtype=np.float32)
        batch_y = np.full((batch_size, self.seq_len), -1, dtype=np.int64)
        batch_weights = np.ones((batch_size, self.seq_len), dtype=np.float32)

        for b in range(batch_size):
            m_idx = np.random.randint(0, len(self.matches))
            feats, targets = self.matches[m_idx]
            n_f = len(targets)
            if n_f <= self.seq_len:
                continue

            s = np.random.randint(0, n_f - self.seq_len)
            e = s + self.seq_len

            x_chunk = feats[s:e].copy()  # (T, 16)
            y_chunk = targets[s:e].copy()  # (T,)

            # 50% Team-symmetric augmentation
            if np.random.rand() > 0.5:
                # Swap home and away channels
                x_chunk[:, [4, 5]] = x_chunk[:, [5, 4]]      # dists
                x_chunk[:, 6] = -x_chunk[:, 6]               # dist_margin
                x_chunk[:, [7, 8]] = x_chunk[:, [8, 7]]      # kappas
                x_chunk[:, [9, 10]] = x_chunk[:, [10, 9]]    # voronoi
                x_chunk[:, [11, 12]] = x_chunk[:, [12, 11]]  # 3m counts
                x_chunk[:, 15] = -x_chunk[:, 15]             # invert x

                # Swap target labels 0 (H) <-> 1 (A)
                y_swapped = y_chunk.copy()
                y_swapped[y_chunk == 0] = 1
                y_swapped[y_chunk == 1] = 0
                y_chunk = y_swapped

            # Compute transition boundary weight
            # Mark transitions where y changes between valid classes
            trans = np.zeros(self.seq_len, dtype=bool)
            diff = (y_chunk[1:] != y_chunk[:-1]) & (y_chunk[1:] != -1) & (y_chunk[:-1] != -1)
            trans[1:] = diff

            # Dilate transition mask by +/- 12 frames (~0.5s window)
            w_mask = np.ones(self.seq_len, dtype=np.float32)
            trans_idx = np.flatnonzero(trans)
            for tidx in trans_idx:
                lo = max(0, tidx - 12)
                hi = min(self.seq_len, tidx + 13)
                w_mask[lo:hi] = 4.0  # 4x weight around transition boundary

            batch_x[b] = x_chunk.T
            batch_y[b] = y_chunk
            batch_weights[b] = w_mask

        return (
            torch.from_numpy(batch_x),
            torch.from_numpy(batch_y),
            torch.from_numpy(batch_weights),
        )


def train_boundary_model(
    train_files,
    val_file,
    out_weights_path="cache/markius_v5_boundary_weights.npz",
    n_steps=1800,
):
    print("Initializing Markius Boundary TCN...")
    model = MarkiusTCN(in_features=N_FEATURES, hidden=32, n_classes=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {n_params:,}")

    train_dataset = BoundarySequenceDataset(train_files, seq_len=512)

    val_feats, val_targets = load_match_data(val_file)
    val_x = torch.from_numpy(val_feats.T).unsqueeze(0)
    val_y = torch.from_numpy(val_targets).unsqueeze(0)

    class_weights = torch.tensor([1.2, 1.2, 0.8], dtype=torch.float32)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-4)

    print(f"Training for {n_steps} steps with Boundary-Weighted Loss...")
    start_time = time.time()

    model.train()
    for step in range(1, n_steps + 1):
        x_batch, y_batch, w_batch = train_dataset.sample_batch(batch_size=32)
        optimizer.zero_grad()
        logits = model(x_batch)  # (B, 3, T)

        # Weighted Cross-Entropy
        per_frame_loss = F.cross_entropy(logits, y_batch, weight=class_weights, ignore_index=-1, reduction="none")
        valid_mask = y_batch != -1
        loss = (per_frame_loss * w_batch).sum() / (w_batch * valid_mask.float()).sum().clamp(min=1.0)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 300 == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                v_slice_x = val_x[:, :, :40000]
                v_slice_y = val_y[:, :40000]
                val_logits = model(v_slice_x)
                val_loss = F.cross_entropy(val_logits, v_slice_y, weight=class_weights, ignore_index=-1).item()
                preds = torch.argmax(val_logits, dim=1)
                valid = v_slice_y != -1
                acc = (preds[valid] == v_slice_y[valid]).float().mean().item()

            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{n_steps:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2%} | Time: {elapsed:.1f}s")
            model.train()

    export_weights_to_numpy(model, out_weights_path)
    print(f"Exported boundary weights to {out_weights_path}")


if __name__ == "__main__":
    all_files = sorted(glob.glob("/workspace/shared/dataset/*.npz"))
    val_path = "/workspace/shared/dataset/metrica_game3.npz"
    train_files = [f for f in all_files if os.path.abspath(f) != os.path.abspath(val_path)]
    train_boundary_model(train_files, val_path)
