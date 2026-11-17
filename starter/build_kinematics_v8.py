#!/usr/bin/env python3
"""Build and verify kinematics_v8.npz with individual player tracking and relative velocity."""

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import games

HOME = 0
AWAY = 1
NOBODY = -1


def compute_velocity_1d(x, dt=1.0 / 25.0):
    """Compute 1D derivative preserving edge frames across NaNs."""
    n = len(x)
    v = np.full(n, np.nan, dtype=np.float32)
    dx_f = np.diff(x) / dt
    f_valid = ~np.isnan(dx_f)

    # Central difference where both left and right forward diffs are valid
    c_valid = np.zeros(n, dtype=bool)
    c_valid[1:-1] = f_valid[:-1] & f_valid[1:]
    v[1:-1] = np.where(c_valid[1:-1], 0.5 * (dx_f[:-1] + dx_f[1:]), v[1:-1])

    # Forward difference at the left boundary of a valid segment
    f_only = np.zeros(n, dtype=bool)
    f_only[:-1] = f_valid & np.isnan(v[:-1])
    v[:-1] = np.where(f_only[:-1], dx_f, v[:-1])

    # Backward difference at the right boundary of a valid segment
    b_only = np.zeros(n, dtype=bool)
    b_only[1:] = f_valid & np.isnan(v[1:])
    v[1:] = np.where(b_only[1:], dx_f, v[1:])
    return v


def extract_or_load_kinematics_v8(raw_path, meta_path, n_frames, pitch, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached kinematics v8 from {cache_path}...")
        data = np.load(cache_path)
        return (
            data["ball_x_recon"].copy(), data["ball_y_recon"].copy(),
            data["ball_is_imputed"].copy(), data["ball_gap_len"].copy(),
            data["mean_speed"].copy(),
            data["home_min_dist"].copy(), data["away_min_dist"].copy(),
            data["home_min_kappa"].copy(), data["away_min_kappa"].copy(),
            data["home_best_player"].copy(), data["away_best_player"].copy(),
            data["home_rel_speed"].copy(), data["away_rel_speed"].copy(),
            data["team_of_track"].copy()
        )

    print("Parsing ATD metadata and track slots...")
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    team_of_track = np.array([0 if t_id == home_id else 1 for t_id in slots], dtype=np.int8)
    n_tracks = len(team_of_track)
    length, width = pitch

    px = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    ball_x_raw = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y_raw = np.full(n_frames, np.nan, dtype=np.float32)
    mean_speed = np.zeros(n_frames, dtype=np.float32)

    print(f"Parsing raw tracking coordinates from {raw_path}...")
    t0 = time.time()
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            vf = int(parts[0]) - 1
            if vf < 0 or vf >= n_frames:
                continue
            speeds = []
            for slot_idx, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1)
                px[vf, slot_idx] = float(group[:c1]) * length
                py[vf, slot_idx] = float(group[c1 + 1:c2]) * width
                sp_str = group[c2 + 1:]
                if sp_str[0] != "N":
                    speeds.append(float(sp_str))
            if speeds:
                mean_speed[vf] = np.mean(speeds)
            ball = parts[2]
            if ball[0] != "N":
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                ball_x_raw[vf] = float(ball[:c1]) * length
                ball_y_raw[vf] = float(ball[c1 + 1:c2]) * width
    print(f"Tracking coordinates parsed in {time.time() - t0:.2f}s.")

    # 1. Trajectory reconstruction of short ball gaps (<= 25 frames / 1.0s)
    # Using exact endpoint linspace: np.linspace(start, end, gap_len + 2)[1:-1]
    bx_recon = ball_x_raw.copy()
    by_recon = ball_y_raw.copy()
    ball_is_imputed = np.zeros(n_frames, dtype=bool)
    ball_gap_len = np.zeros(n_frames, dtype=np.int16)

    nan_b = np.isnan(bx_recon)
    diff = np.diff(np.concatenate(([0], nan_b.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    for s, e in zip(starts, ends):
        gap_len = e - s
        if gap_len <= 25 and s > 0 and e < n_frames:
            if not np.isnan(bx_recon[s - 1]) and not np.isnan(bx_recon[e]):
                if (-1.0 <= bx_recon[s - 1] <= length + 1.0) and (-1.0 <= bx_recon[e] <= length + 1.0):
                    bx_recon[s:e] = np.linspace(bx_recon[s - 1], bx_recon[e], gap_len + 2)[1:-1]
                    by_recon[s:e] = np.linspace(by_recon[s - 1], by_recon[e], gap_len + 2)[1:-1]
                    ball_is_imputed[s:e] = True
                    ball_gap_len[s:e] = gap_len

    # Ball velocity components
    v_bx = compute_velocity_1d(bx_recon)
    v_by = compute_velocity_1d(by_recon)

    # 2. Vector approach velocities, dynamic envelopes, and individual control indices
    home_tracks = np.where(team_of_track == HOME)[0]
    away_tracks = np.where(team_of_track == AWAY)[0]

    home_min_dist = np.full(n_frames, np.nan, dtype=np.float32)
    away_min_dist = np.full(n_frames, np.nan, dtype=np.float32)
    home_min_kappa = np.full(n_frames, 1e9, dtype=np.float32)
    away_min_kappa = np.full(n_frames, 1e9, dtype=np.float32)
    home_best_player = np.full(n_frames, -1, dtype=np.int16)
    away_best_player = np.full(n_frames, -1, dtype=np.int16)
    home_rel_speed = np.full(n_frames, np.nan, dtype=np.float32)
    away_rel_speed = np.full(n_frames, np.nan, dtype=np.float32)

    R0 = 0.85
    tau_stride = 0.25

    for team_name, tracks, min_d_arr, min_k_arr, best_p_arr, rel_sp_arr in [
        ("Home", home_tracks, home_min_dist, home_min_kappa, home_best_player, home_rel_speed),
        ("Away", away_tracks, away_min_dist, away_min_kappa, away_best_player, away_rel_speed),
    ]:
        print(f"Computing vector kinematics and individual control for {team_name}...")
        sub_px = px[:, tracks]
        sub_py = py[:, tracks]

        sub_vx = np.gradient(sub_px, axis=0) * 25.0
        sub_vy = np.gradient(sub_py, axis=0) * 25.0

        dx = sub_px - bx_recon[:, None]
        dy = sub_py - by_recon[:, None]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 0.05)
        r_to_b_x = - dx / dist_safe
        r_to_b_y = - dy / dist_safe

        v_app = sub_vx * r_to_b_x + sub_vy * r_to_b_y
        v_app_fwd = np.clip(np.nan_to_num(v_app, nan=0.0), 0.0, 8.5)

        R_dyn = R0 + tau_stride * v_app_fwd
        kappa = dist / R_dyn

        # Find best player index per frame
        kappa_clean = np.where(np.isnan(kappa), 1e9, kappa)
        best_slot = np.argmin(kappa_clean, axis=1)

        m_dist = np.take_along_axis(dist, best_slot[:, None], axis=1).squeeze(1)
        m_k = np.take_along_axis(kappa, best_slot[:, None], axis=1).squeeze(1)
        best_p = tracks[best_slot]

        best_vx = np.take_along_axis(sub_vx, best_slot[:, None], axis=1).squeeze(1)
        best_vy = np.take_along_axis(sub_vy, best_slot[:, None], axis=1).squeeze(1)

        rel_vx = best_vx - v_bx
        rel_vy = best_vy - v_by
        r_speed = np.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)

        all_nan = np.all(np.isnan(kappa), axis=1)
        best_p[all_nan] = -1
        m_k[all_nan] = 1e9
        m_dist[all_nan] = np.nan
        r_speed[all_nan] = np.nan

        min_d_arr[:] = m_dist
        min_k_arr[:] = m_k
        best_p_arr[:] = best_p
        rel_sp_arr[:] = r_speed

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving kinematics v8 cache to {cache_path}...")
        np.savez_compressed(
            cache_path,
            bx_recon=bx_recon, by_recon=by_recon,
            ball_is_imputed=ball_is_imputed, ball_gap_len=ball_gap_len,
            mean_speed=mean_speed,
            home_min_dist=home_min_dist, away_min_dist=away_min_dist,
            home_min_kappa=home_min_kappa, away_min_kappa=away_min_kappa,
            home_best_player=home_best_player, away_best_player=away_best_player,
            home_rel_speed=home_rel_speed, away_rel_speed=away_rel_speed,
            team_of_track=team_of_track
        )
        print("Cache saved successfully.")

    return (
        bx_recon, by_recon, ball_is_imputed, ball_gap_len, mean_speed,
        home_min_dist, away_min_dist, home_min_kappa, away_min_kappa,
        home_best_player, away_best_player, home_rel_speed, away_rel_speed,
        team_of_track
    )


if __name__ == "__main__":
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], "BAR-ATH")
    n_frames = games.source_frames("tactical", "BAR-ATH")
    pitch = games.GAMES["BAR-ATH"]["pitch"]
    cache_path = os.path.join(ROOT, "cache", "kinematics_v8.npz")
    extract_or_load_kinematics_v8(raw_path, meta_path, n_frames, pitch, cache_path=cache_path)
