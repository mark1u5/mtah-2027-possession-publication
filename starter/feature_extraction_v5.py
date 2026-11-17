#!/usr/bin/env python3
"""Unified Feature Extraction Pipeline for Markius v5 (TCN).

Extracts a continuous, invariant 16-channel feature matrix from tracking data.
Works identically on:
1. Unified external dataset .npz files (/workspace/shared/dataset/*.npz)
2. Challenge match BAR-ATH kinematics (cache/kinematics_v2.npz)
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FEATURE_NAMES = [
    "v_ball",            # 0: Ball speed (m/s)
    "a_ball",            # 1: Ball acceleration (m/s^2)
    "ball_missing",      # 2: Ball missing flag (0 or 1)
    "ball_gap_s",        # 3: Time since ball last seen (s)
    "home_min_dist",     # 4: Distance to nearest home player (m)
    "away_min_dist",     # 5: Distance to nearest away player (m)
    "dist_margin",       # 6: away_min_dist - home_min_dist (m)
    "home_min_kappa",    # 7: Home minimum reach ratio (tau / reach)
    "away_min_kappa",    # 8: Away minimum reach ratio (tau / reach)
    "p_voronoi_home",    # 9: Pointwise pitch control for home [0, 1]
    "p_voronoi_away",    # 10: Pointwise pitch control for away [0, 1]
    "n_home_3m",         # 11: Number of home players within 3m
    "n_away_3m",         # 12: Number of away players within 3m
    "mean_player_spd",   # 13: Average speed of visible players on pitch (m/s)
    "dist_boundary",     # 14: Distance to nearest pitch boundary (m)
    "norm_ball_x",       # 15: Normalized ball x position [-0.5, 0.5]
]

N_FEATURES = len(FEATURE_NAMES)  # 16


def reconstruct_ball_simple(ball_xy, fps=25.0, max_gap_frames=125):
    """Linear aerodynamic-capped ball imputation over occlusions."""
    bx = ball_xy[:, 0].copy()
    by = ball_xy[:, 1].copy()
    n_frames = len(bx)

    is_missing = np.isnan(bx)
    diff = np.diff(np.concatenate(([0], is_missing.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    gap_len = np.zeros(n_frames, dtype=np.float32)
    for s, e in zip(starts, ends):
        length = e - s
        gap_len[s:e] = np.arange(1, length + 1, dtype=np.float32)
        if length <= max_gap_frames and s > 0 and e < n_frames:
            p0_x, p0_y = bx[s - 1], by[s - 1]
            p1_x, p1_y = bx[e], by[e]
            if not np.isnan(p0_x) and not np.isnan(p1_x):
                dist = np.hypot(p1_x - p0_x, p1_y - p0_y)
                total_t = (length + 1) / fps
                if (dist / total_t) <= 35.0:
                    steps = np.arange(1, length + 1, dtype=np.float32) / (length + 1)
                    bx[s:e] = p0_x + steps * (p1_x - p0_x)
                    by[s:e] = p0_y + steps * (p1_y - p0_y)
    return bx, by, is_missing, gap_len


def extract_features_from_dataset_npz(npz_path):
    """Extract standard 16-channel feature matrix and labels from a unified .npz match."""
    data = np.load(npz_path)
    fps = float(data["fps"]) if "fps" in data else 25.0
    dt = 1.0 / fps
    length = float(data["pitch_length"]) if "pitch_length" in data else 105.0
    width = float(data["pitch_width"]) if "pitch_width" in data else 68.0

    ball_xy = data["ball_xy"]
    n_frames = len(ball_xy)
    players_xy = data["players_xy"]        # (N, S, 2)
    players_team = data["players_team"]    # (S,) 0=home, 1=away
    players_vis = data["players_visible"]  # (N, S) bool
    labels = data["label"]                # (N,)

    bx, by, is_missing, gap_len = reconstruct_ball_simple(ball_xy, fps=fps)

    # 1. Ball Kinematics
    v_bx = np.gradient(np.nan_to_num(bx, nan=length / 2.0)) / dt
    v_by = np.gradient(np.nan_to_num(by, nan=width / 2.0)) / dt
    v_ball = np.clip(np.hypot(v_bx, v_by), 0.0, 35.0)

    a_bx = np.gradient(v_bx) / dt
    a_by = np.gradient(v_by) / dt
    a_ball = np.clip(np.hypot(a_bx, a_by), 0.0, 40.0)

    ball_gap_s = np.clip(gap_len / fps, 0.0, 5.0)

    # 2. Player Speeds & Positions
    px = players_xy[:, :, 0]
    py = players_xy[:, :, 1]

    # Mask non-visible players with NaN
    px = np.where(players_vis, px, np.nan)
    py = np.where(players_vis, py, np.nan)

    vx = np.gradient(np.nan_to_num(px, nan=0.0), axis=0) / dt
    vy = np.gradient(np.nan_to_num(py, nan=0.0), axis=0) / dt
    player_speeds = np.hypot(vx, vy)
    player_speeds[~players_vis] = np.nan

    with np.errstate(all="ignore"):
        mean_player_spd = np.nanmean(player_speeds, axis=1)
    mean_player_spd = np.clip(np.nan_to_num(mean_player_spd, nan=1.5), 0.0, 10.0)

    # 3. Proximities & Voronoi Potentials
    dx = bx[:, None] - px
    dy = by[:, None] - py
    dists = np.hypot(dx, dy)
    dists_safe = np.maximum(dists, 0.05)

    u_x = dx / dists_safe
    u_y = dy / dists_safe
    v_app = np.clip(vx * u_x + vy * u_y, 0.0, 8.5)
    reach = 0.85 + 0.25 * v_app
    kappa = dists / np.maximum(reach, 0.1)

    home_mask = (players_team == 0)[None, :] & players_vis
    away_mask = (players_team == 1)[None, :] & players_vis

    # Minimum distances
    dists_clean = np.where(np.isfinite(dists), dists, np.inf)
    home_d = np.where(home_mask, dists_clean, np.inf)
    away_d = np.where(away_mask, dists_clean, np.inf)
    home_min_dist = np.clip(np.min(home_d, axis=1), 0.0, 15.0)
    away_min_dist = np.clip(np.min(away_d, axis=1), 0.0, 15.0)
    home_min_dist = np.where(np.isfinite(home_min_dist), home_min_dist, 15.0)
    away_min_dist = np.where(np.isfinite(away_min_dist), away_min_dist, 15.0)
    dist_margin = np.clip(away_min_dist - home_min_dist, -10.0, 10.0)

    # Minimum kappa
    kappa_clean = np.where(np.isfinite(kappa), kappa, np.inf)
    home_k = np.where(home_mask, kappa_clean, np.inf)
    away_k = np.where(away_mask, kappa_clean, np.inf)
    home_min_kappa = np.clip(np.min(home_k, axis=1), 0.0, 5.0)
    away_min_kappa = np.clip(np.min(away_k, axis=1), 0.0, 5.0)
    home_min_kappa = np.where(np.isfinite(home_min_kappa), home_min_kappa, 5.0)
    away_min_kappa = np.where(np.isfinite(away_min_kappa), away_min_kappa, 5.0)

    # Voronoi potency
    tau_arr = dists / 8.2
    potency = np.nan_to_num(np.exp(-tau_arr / 0.35) + 1.6 * np.exp(-(dists ** 2) / (2.0 * (1.35 ** 2))), nan=0.0)
    psi_home = np.sum(potency * home_mask, axis=1)
    psi_away = np.sum(potency * away_mask, axis=1)
    eps_loose = np.exp(-1.60 / 0.35)
    denom = psi_home + psi_away + eps_loose
    p_voronoi_home = psi_home / denom
    p_voronoi_away = psi_away / denom

    # Player counts within 3m
    n_home_3m = np.sum((dists <= 3.0) & home_mask, axis=1).astype(np.float32)
    n_away_3m = np.sum((dists <= 3.0) & away_mask, axis=1).astype(np.float32)

    # Boundary distance & normalized position
    dist_x = np.minimum(bx, length - bx)
    dist_y = np.minimum(by, width - by)
    dist_boundary = np.clip(np.minimum(dist_x, dist_y), -5.0, 20.0)
    dist_boundary = np.nan_to_num(dist_boundary, nan=0.0)

    norm_ball_x = np.clip(bx / length - 0.5, -0.5, 0.5)
    norm_ball_x = np.nan_to_num(norm_ball_x, nan=0.0)

    # Assemble feature matrix (N, 16)
    features = np.column_stack([
        v_ball,
        a_ball,
        is_missing.astype(np.float32),
        ball_gap_s,
        home_min_dist,
        away_min_dist,
        dist_margin,
        home_min_kappa,
        away_min_kappa,
        p_voronoi_home,
        p_voronoi_away,
        n_home_3m,
        n_away_3m,
        mean_player_spd,
        dist_boundary,
        norm_ball_x,
    ]).astype(np.float32)

    return features, labels


def extract_features_from_competition(kinematics, pitch=(105.3, 68.0), fps=25.0):
    """Extract standard 16-channel feature matrix from competition match kinematics."""
    bx = kinematics["ball_x_recon"]
    by = kinematics["ball_y_recon"]
    n_frames = len(bx)
    length, width = pitch
    dt = 1.0 / fps

    v_ball = np.clip(np.nan_to_num(kinematics["v_ball"], nan=0.0), 0.0, 35.0)
    a_ball = np.clip(np.nan_to_num(kinematics["a_ball"], nan=0.0), 0.0, 40.0)
    is_missing = np.nan_to_num(kinematics["ball_is_imputed"], nan=1.0).astype(np.float32)
    ball_gap_s = np.clip(np.nan_to_num(kinematics["ball_gap_len"], nan=0.0) / fps, 0.0, 5.0)

    home_min_dist = np.clip(np.nan_to_num(kinematics["home_min_dist"], nan=15.0), 0.0, 15.0)
    away_min_dist = np.clip(np.nan_to_num(kinematics["away_min_dist"], nan=15.0), 0.0, 15.0)
    dist_margin = np.clip(away_min_dist - home_min_dist, -10.0, 10.0)

    home_min_kappa = np.clip(np.nan_to_num(kinematics["home_min_kappa"], nan=5.0), 0.0, 5.0)
    away_min_kappa = np.clip(np.nan_to_num(kinematics["away_min_kappa"], nan=5.0), 0.0, 5.0)

    # Voronoi pitch control
    px = kinematics["px"]
    py = kinematics["py"]
    team_of_track = kinematics["team_of_track"]

    dx = bx[:, None] - px
    dy = by[:, None] - py
    dists = np.hypot(dx, dy)
    tau_arr = dists / 8.2
    potency = np.nan_to_num(np.exp(-tau_arr / 0.35) + 1.6 * np.exp(-(dists ** 2) / (2.0 * (1.35 ** 2))), nan=0.0)

    home_mask = (team_of_track == 0)[None, :] & (~np.isnan(px))
    away_mask = (team_of_track == 1)[None, :] & (~np.isnan(px))

    psi_home = np.sum(potency * home_mask, axis=1)
    psi_away = np.sum(potency * away_mask, axis=1)
    eps_loose = np.exp(-1.60 / 0.35)
    denom = psi_home + psi_away + eps_loose
    p_voronoi_home = psi_home / denom
    p_voronoi_away = psi_away / denom

    n_home_3m = np.sum((dists <= 3.0) & home_mask, axis=1).astype(np.float32)
    n_away_3m = np.sum((dists <= 3.0) & away_mask, axis=1).astype(np.float32)

    mean_player_spd = np.clip(np.nan_to_num(kinematics["mean_speed"], nan=1.5), 0.0, 10.0)

    dist_x = np.minimum(bx, length - bx)
    dist_y = np.minimum(by, width - by)
    dist_boundary = np.clip(np.minimum(dist_x, dist_y), -5.0, 20.0)
    dist_boundary = np.nan_to_num(dist_boundary, nan=0.0)

    norm_ball_x = np.clip(bx / length - 0.5, -0.5, 0.5)
    norm_ball_x = np.nan_to_num(norm_ball_x, nan=0.0)

    features = np.column_stack([
        v_ball,
        a_ball,
        is_missing,
        ball_gap_s,
        home_min_dist,
        away_min_dist,
        dist_margin,
        home_min_kappa,
        away_min_kappa,
        p_voronoi_home,
        p_voronoi_away,
        n_home_3m,
        n_away_3m,
        mean_player_spd,
        dist_boundary,
        norm_ball_x,
    ]).astype(np.float32)

    return features
