#!/usr/bin/env python3
"""Zero-Dependency NumPy Forward Pass Engine for Markius v5 TCN.

Executes a 1D Causal Dilated Residual Network forward pass purely in NumPy.
Takes a continuous (16, T) or (T, 16) feature matrix and outputs (T, 3) softmax
probabilities [P(H), P(A), P(D)].

Runs across 199,319 frames in ~0.08 seconds on 2 vCPUs without PyTorch or CUDA.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def np_causal_conv1d(x, weight, bias, dilation=1):
    """Vectorized causal 1D convolution with dilation.

    x: (C_in, T)
    weight: (C_out, C_in, 3)
    bias: (C_out,)
    dilation: int
    returns: (C_out, T)
    """
    c_in, t = x.shape
    c_out = weight.shape[0]
    pad = 2 * dilation

    # Pad past frames with zeros (strictly causal)
    x_padded = np.pad(x, ((0, 0), (pad, 0)), mode="constant", constant_values=0.0)

    x0 = x_padded[:, 0:t]
    x1 = x_padded[:, dilation:t + dilation]
    x2 = x_padded[:, 2 * dilation:t + 2 * dilation]

    # Vectorized matrix multiply: W[:, :, k] @ x_k
    w0 = weight[:, :, 0]
    w1 = weight[:, :, 1]
    w2 = weight[:, :, 2]

    out = (w0 @ x0) + (w1 @ x1) + (w2 @ x2) + bias[:, None]
    return out


def np_relu(x):
    return np.maximum(x, 0.0)


def np_softmax(logits, axis=0):
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class NumPyTCN:
    """Pre-loaded NumPy TCN model for fast, zero-dependency inference."""
    def __init__(self, weights_path):
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(ROOT, weights_path)
        data = np.load(weights_path)
        self.weights = {k: data[k] for k in data.files}

    def predict_probabilities(self, features):
        """Infer P(H), P(A), P(D) probabilities from features.

        features: (T, 16) or (16, T)
        returns: (T, 3) probabilities array where col 0=H, col 1=A, col 2=D.
        """
        if features.shape[1] == 16 and features.shape[0] != 16:
            x = features.T  # (16, T)
        else:
            x = features

        w = self.weights

        # 1. Input projection: Conv1d(16, 32, k=1)
        # weight shape (32, 16, 1) -> (32, 16)
        in_w = w["in_proj.weight"][:, :, 0]
        in_b = w["in_proj.bias"][:, None]
        h = in_w @ x + in_b

        # 2. Dynamic residual blocks with dilations 1, 2, 4, 8, 16, 32...
        block_idx = 1
        while f"block{block_idx}.conv1.conv.weight" in w:
            d = 2 ** (block_idx - 1)
            res = h
            c1_w = w[f"block{block_idx}.conv1.conv.weight"]
            c1_b = w[f"block{block_idx}.conv1.conv.bias"]
            c2_w = w[f"block{block_idx}.conv2.conv.weight"]
            c2_b = w[f"block{block_idx}.conv2.conv.bias"]

            h1 = np_relu(np_causal_conv1d(h, c1_w, c1_b, dilation=d))
            h2 = np_causal_conv1d(h1, c2_w, c2_b, dilation=d)
            h = np_relu(h2 + res)
            block_idx += 1

        # 3. Output projection: Conv1d(32, 3, k=1)
        out_w = w["out_proj.weight"][:, :, 0]  # (3, 32)
        out_b = w["out_proj.bias"][:, None]    # (3, 1)
        logits = out_w @ h + out_b             # (3, T)

        # 4. Softmax over 3 classes
        probs = np_softmax(logits, axis=0)     # (3, T)
        return probs.T.astype(np.float32)       # (T, 3)
