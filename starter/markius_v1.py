#!/usr/bin/env python3
"""Markius v1 — Autonomous, Physically Grounded Possession Pipeline.

Refactored from v10 to ensure clean, rule-compliant execution.
"""

import argparse
import os
import sys
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import games

HOME, AWAY, NOBODY = 0, 1, -1


def get_team_mapping(meta_path):
    """Map tracking slots to teams (HOME=0, AWAY=1) from EPTS metadata."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if t_id == home_id else AWAY for t_id in slots], dtype=np.int8)


def compute_velocity_1d(x, dt=1.0 / 25.0):
    """Compute central difference velocity with boundary forward/backward fallbacks."""
    n = len(x)
    v = np.full(n, np.nan, dtype=np.float32)
    dx_f = np.diff(x) / dt
    f_valid = ~np.isnan(dx_f)
    v[:-1][f_valid] = dx_f[f_valid]
    dx_b = np.diff(x) / dt
    b_valid = ~np.isnan(dx_b)
    v[1:][b_valid] = (v[1:][b_valid] + dx_b[b_valid]) * 0.5 if np.any(f_valid) else dx_b[b_valid]
    dx_c = (x[2:] - x[:-2]) / (2.0 * dt)
    c_valid = ~np.isnan(dx_c)
    v[1:-1][c_valid] = dx_c[c_valid]
    return v


def filter_min_duration(mask, min_len):
    """Filter out active intervals shorter than min_len frames."""
    out = mask.copy()
    diff = np.diff(np.concatenate(([0], out.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    for s, e in zip(starts, ends):
        if e - s < min_len:
            out[s:e] = False
    return out


def extract_or_load_kinematics(raw_path, meta_path, n_frames, pitch, cache_path=None):
    """Extract or load player and ball kinematics with autonomous glitch filtering."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading kinematics from cache: {cache_path}")
        data = np.load(cache_path)
        return {
            "ball_x_raw": data["ball_x_raw"],
            "ball_y_raw": data["ball_y_raw"],
            "ball_x_recon": data["ball_x_recon"],
            "ball_y_recon": data["ball_y_recon"],
            "ball_is_imputed": data["ball_is_imputed"],
            "ball_gap_len": data["ball_gap_len"],
            "mean_speed": data["mean_speed"],
            "home_min_dist": data["home_min_dist"],
            "away_min_dist": data["away_min_dist"],
            "home_min_kappa": data["home_min_kappa"],
            "away_min_kappa": data["away_min_kappa"],
            "home_best_player": data["home_best_player"],
            "away_best_player": data["away_best_player"],
            "home_rel_speed": data["home_rel_speed"],
            "away_rel_speed": data["away_rel_speed"],
            "team_of_track": data["team_of_track"],
        }

    length, width = pitch
    team_of_track = get_team_mapping(meta_path)
    n_tracks = len(team_of_track)

    px = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    mean_speed = np.zeros(n_frames, dtype=np.float32)
    ball_x_raw = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y_raw = np.full(n_frames, np.nan, dtype=np.float32)

    print(f"Parsing raw tracking data from {raw_path}...")
    with open(raw_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            vf = int(parts[0]) - 1
            if vf >= n_frames:
                break
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

    # 1. Non-physical Transport Velocity Glitch Filter (> 12 m/s jump)
    v_max_transport = 12.0
    for slot in range(n_tracks):
        valid_idx = np.flatnonzero(~np.isnan(px[:, slot]))
        if len(valid_idx) < 2:
            continue
        diff_idx = np.diff(valid_idx)
        for i, gap in enumerate(diff_idx):
            if gap > 1:
                f_prev = valid_idx[i]
                f_curr = valid_idx[i + 1]
                dt = (f_curr - f_prev) / 25.0
                d_jump = np.hypot(px[f_curr, slot] - px[f_prev, slot], py[f_curr, slot] - py[f_prev, slot])
                if (d_jump / dt) > v_max_transport:
                    end_burst = f_curr
                    while end_burst < n_frames - 1 and not np.isnan(px[end_burst + 1, slot]):
                        end_burst += 1
                    px[f_curr:end_burst + 1, slot] = np.nan
                    py[f_curr:end_burst + 1, slot] = np.nan

    # 2. Piecewise linear interpolation for short ball gaps (<= 25 frames / 1.0s)
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

    v_bx = compute_velocity_1d(bx_recon)
    v_by = compute_velocity_1d(by_recon)

    features = {}
    r0 = 0.85
    tau_stride = 0.25
    max_reach_spd = 8.5

    for team_code, name in [(HOME, "home"), (AWAY, "away")]:
        slots = np.where(team_of_track == team_code)[0]
        t_px = px[:, slots]
        t_py = py[:, slots]
        tracks = slots.astype(np.int16)

        dx = bx_recon[:, None] - t_px
        dy = by_recon[:, None] - t_py
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 0.05)
        u_x = dx / dist_safe
        u_y = dy / dist_safe

        vx = np.gradient(t_px, axis=0) * 25.0
        vy = np.gradient(t_py, axis=0) * 25.0

        v_app = vx * u_x + vy * u_y
        v_app_fwd = np.clip(np.nan_to_num(v_app, nan=0.0), 0.0, max_reach_spd)
        reach = r0 + tau_stride * v_app_fwd
        kappa = dist / np.maximum(reach, 0.1)

        sub_k_clean = np.where(np.isnan(kappa), 1e9, kappa)
        best_slot = np.argmin(sub_k_clean, axis=1)
        min_kappa = np.take_along_axis(kappa, best_slot[:, None], axis=1).squeeze(1)
        min_dist = np.take_along_axis(dist, best_slot[:, None], axis=1).squeeze(1)
        best_player = tracks[best_slot]

        best_vx = np.take_along_axis(vx, best_slot[:, None], axis=1).squeeze(1)
        best_vy = np.take_along_axis(vy, best_slot[:, None], axis=1).squeeze(1)
        rel_vx = best_vx - v_bx
        rel_vy = best_vy - v_by
        rel_speed = np.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)

        all_nan = np.all(np.isnan(kappa), axis=1)
        best_player[all_nan] = -1
        min_kappa[all_nan] = 1e9
        min_dist[all_nan] = np.nan
        rel_speed[all_nan] = np.nan

        features[name] = {
            "min_dist": min_dist,
            "min_kappa": min_kappa,
            "best_player": best_player.astype(np.int16),
            "rel_speed": rel_speed.astype(np.float32),
        }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            ball_x_raw=ball_x_raw,
            ball_y_raw=ball_y_raw,
            ball_x_recon=bx_recon,
            ball_y_recon=by_recon,
            ball_is_imputed=ball_is_imputed,
            ball_gap_len=ball_gap_len,
            mean_speed=mean_speed,
            home_min_dist=features["home"]["min_dist"],
            away_min_dist=features["away"]["min_dist"],
            home_min_kappa=features["home"]["min_kappa"],
            away_min_kappa=features["away"]["min_kappa"],
            home_best_player=features["home"]["best_player"],
            away_best_player=features["away"]["best_player"],
            home_rel_speed=features["home"]["rel_speed"],
            away_rel_speed=features["away"]["rel_speed"],
            team_of_track=team_of_track,
        )

    return {
        "ball_x_raw": ball_x_raw,
        "ball_y_raw": ball_y_raw,
        "ball_x_recon": bx_recon,
        "ball_y_recon": by_recon,
        "ball_is_imputed": ball_is_imputed,
        "ball_gap_len": ball_gap_len,
        "mean_speed": mean_speed,
        "home_min_dist": features["home"]["min_dist"],
        "away_min_dist": features["away"]["min_dist"],
        "home_min_kappa": features["home"]["min_kappa"],
        "away_min_kappa": features["away"]["min_kappa"],
        "home_best_player": features["home"]["best_player"],
        "away_best_player": features["away"]["best_player"],
        "home_rel_speed": features["home"]["rel_speed"],
        "away_rel_speed": features["away"]["rel_speed"],
        "team_of_track": team_of_track,
    }


def load_smart_tagging_phases(st_path, n_frames):
    tree = ET.parse(st_path)
    st_home = np.zeros(n_frames, dtype=bool)
    st_away = np.zeros(n_frames, dtype=bool)
    st_set_pieces = np.zeros(n_frames, dtype=bool)

    offensive_codes = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL"}

    for inst in tree.getroot().iter("instance"):
        s = max(0, min(n_frames - 1, round(float(inst.find("start").text) * 25)))
        e = max(0, min(n_frames, round(float(inst.find("end").text) * 25)))
        c = inst.find("code").text
        team = None
        for l in inst.iter("label"):
            if l.find("group") is not None and l.find("text") is not None:
                if l.find("group").text == "Team":
                    team = l.find("text").text
        if c in offensive_codes:
            if team == "Barcelona":
                st_home[s:e] = True
            elif team == "Atletic Club":
                st_away[s:e] = True
        elif c == "SET PIECES":
            st_set_pieces[s:e] = True

    return st_home, st_away, st_set_pieces


def predict_possession_markius(
    kinematics,
    st_home,
    st_away,
    st_set_pieces,
    n_frames,
    half_blocks,
    kickoff_2_frame,
    kappa_thresh=0.95,
    max_gap=25.0,
    penalty_slope=0.4,
    K_turnover=11,
    max_transit=2,
    same_player_ratio=0.50,
    v_rel_thresh=3.5,
    speed_window=60,
    dead_speed_thresh=1.40,
    min_dead_len=90,
):
    bx_recon = kinematics["ball_x_recon"]
    by_recon = kinematics["ball_y_recon"]
    ball_is_imputed = kinematics["ball_is_imputed"]
    ball_gap_len = kinematics["ball_gap_len"]
    mean_speed = kinematics["mean_speed"]
    home_min_kappa = kinematics["home_min_kappa"]
    away_min_kappa = kinematics["away_min_kappa"]
    home_best_player = kinematics["home_best_player"]
    away_best_player = kinematics["away_best_player"]
    home_rel_speed = kinematics["home_rel_speed"]
    away_rel_speed = kinematics["away_rel_speed"]

    half1_start, half1_end = half_blocks[0][1], half_blocks[0][2]
    half2_start, half2_end = half_blocks[1][1], half_blocks[1][2]

    # Continuous confidence penalty on imputed frames
    w_gap = np.where(ball_is_imputed, 1.0 - penalty_slope * (ball_gap_len / max_gap), 1.0).astype(np.float32)
    h_eff = home_min_kappa / w_gap
    a_eff = away_min_kappa / w_gap

    # Dead ball detection
    is_dead = np.zeros(n_frames, dtype=bool)
    is_dead[half2_start:kickoff_2_frame] = True

    ball_outside = (~np.isnan(bx_recon)) & ((bx_recon < -0.2) | (bx_recon > 105.5) | (by_recon < -0.2) | (by_recon > 68.2))
    ball_outside = filter_min_duration(ball_outside, 15)
    is_dead |= ball_outside

    smooth_speed = np.convolve(mean_speed, np.ones(speed_window) / speed_window, mode="same")
    missing_stoppage = np.isnan(bx_recon) & (smooth_speed < 1.9)
    missing_stoppage = filter_min_duration(missing_stoppage, min_dead_len)
    is_dead |= missing_stoppage

    ultra_low_speed = (smooth_speed < dead_speed_thresh) & (smooth_speed > 0) & (~(st_home | st_away))
    ultra_low_speed = filter_min_duration(ultra_low_speed, min_dead_len)
    is_dead |= ultra_low_speed

    v_bx = np.diff(bx_recon, prepend=bx_recon[0]) * 25.0
    v_by = np.diff(by_recon, prepend=by_recon[0]) * 25.0
    v_ball = np.hypot(v_bx, v_by)

    dead_diff = np.diff(np.concatenate(([0], is_dead.view(np.int8), [0])))
    dead_ends = np.flatnonzero(dead_diff == -1)
    restart_zone = np.zeros(n_frames, dtype=bool)
    for de in dead_ends:
        restart_zone[de:min(n_frames, de + 15)] = True
    restart_zone |= (abs(np.arange(n_frames) - half1_start) <= 30)
    restart_zone |= (abs(np.arange(n_frames) - kickoff_2_frame) <= 30)
    restart_zone |= st_set_pieces

    h_ctrl = (~np.isnan(h_eff)) & (h_eff <= kappa_thresh) & (~is_dead)
    a_ctrl = (~np.isnan(a_eff)) & (a_eff <= kappa_thresh) & (~is_dead)

    state_seq = np.full(n_frames, NOBODY, dtype=np.int8)
    curr_state = NOBODY
    opp_cand = NOBODY
    opp_frames = 0
    opp_start_frame = -1
    transit_frames = 0
    cand_history = deque(maxlen=K_turnover)

    eps_tie = 1e-4

    for f in range(n_frames):
        if is_dead[f]:
            curr_state = NOBODY
            opp_cand = NOBODY
            opp_frames = 0
            opp_start_frame = -1
            transit_frames = 0
            cand_history.clear()
            state_seq[f] = NOBODY
            continue

        hc = h_ctrl[f]
        ac = a_ctrl[f]

        if hc and not ac:
            inst = HOME
        elif ac and not hc:
            inst = AWAY
        elif hc and ac:
            if h_eff[f] < a_eff[f] - eps_tie:
                inst = HOME
            elif a_eff[f] < h_eff[f] - eps_tie:
                inst = AWAY
            else:
                inst = curr_state if curr_state != NOBODY else NOBODY
        else:
            inst = NOBODY

        if curr_state == NOBODY:
            if inst != NOBODY:
                if opp_cand == inst:
                    opp_frames += 1
                else:
                    opp_cand = inst
                    opp_frames = 1
                    opp_start_frame = f

                is_restart_impulse = restart_zone[f] or (v_ball[f] >= 2.0)
                required_init = 1 if is_restart_impulse else 3
                if opp_frames >= required_init:
                    curr_state = opp_cand
                    state_seq[opp_start_frame:f + 1] = curr_state
                    opp_cand = NOBODY
                    opp_frames = 0
            else:
                opp_cand = NOBODY
                opp_frames = 0
            state_seq[f] = curr_state

        else:
            opp = AWAY if curr_state == HOME else HOME
            if inst == opp:
                best_p = away_best_player[f] if opp == AWAY else home_best_player[f]
                rel_s = away_rel_speed[f] if opp == AWAY else home_rel_speed[f]

                if opp_cand == opp:
                    opp_frames += 1
                else:
                    opp_cand = opp
                    opp_frames = 1
                    opp_start_frame = f
                    cand_history.clear()

                cand_history.append((f, best_p, rel_s))
                transit_frames = 0

                if len(cand_history) == K_turnover:
                    players = [p[1] for p in cand_history if p[1] >= 0]
                    unique_p, counts = np.unique(players, return_counts=True) if players else ([], [])
                    dom_frac = (np.max(counts) / len(players)) if len(players) > 0 else 0.0
                    same_player = dom_frac >= same_player_ratio

                    valid_pairs = [(p[0], p[2]) for p in cand_history if not np.isnan(p[2])]
                    moves_with_player = (np.median([p[1] for p in valid_pairs]) <= v_rel_thresh) if valid_pairs else False

                    if same_player or moves_with_player:
                        curr_state = opp
                        backdate_frame = min(valid_pairs, key=lambda p: p[1])[0] if valid_pairs else cand_history[0][0]
                        state_seq[backdate_frame:f + 1] = curr_state
                        opp_cand = NOBODY
                        opp_frames = 0
                        cand_history.clear()
                    else:
                        state_seq[f] = curr_state
                else:
                    state_seq[f] = curr_state

            elif inst == NOBODY:
                if opp_cand != NOBODY:
                    transit_frames += 1
                    if transit_frames > max_transit:
                        opp_cand = NOBODY
                        opp_frames = 0
                        transit_frames = 0
                        cand_history.clear()
                state_seq[f] = curr_state

            else:
                opp_cand = NOBODY
                opp_frames = 0
                transit_frames = 0
                cand_history.clear()
                state_seq[f] = curr_state

    final_labels = np.full(n_frames, "D", dtype="<U1")
    final_labels[state_seq == HOME] = "H"
    final_labels[state_seq == AWAY] = "A"

    final_labels[:half1_start] = "X"
    final_labels[half1_end:half2_start] = "X"
    final_labels[half2_end:] = "X"

    return final_labels


def run_pipeline(game="BAR-ATH", out_csv="evaluation/results/markius_v1.csv", data_dir=None):
    """Run full Markius v1 pipeline and write formatted submission CSV."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir)
    st_path = games.asset_paths(game, data_dir)["smart_tagging"]
    n_frames = games.source_frames("tactical", game)
    pitch = games.GAMES[game]["pitch"]
    half_blocks = games.half_video_blocks(game)

    k2_cfg = games.GAMES[game]["kickoff_2nd"]
    kickoff_2_frame = (k2_cfg["minute"] * 60 + k2_cfg["second"]) * 25 + k2_cfg["frame"]

    cache_path = os.path.join(ROOT, "cache", "kinematics_v10.npz")
    print(f"Reading tracking data for {game}...")
    kinematics = extract_or_load_kinematics(raw_path, meta_path, n_frames, pitch, cache_path=cache_path)

    print(f"Loading Smart Tagging for {game}...")
    st_home, st_away, st_set_pieces = load_smart_tagging_phases(st_path, n_frames)

    print("Executing Markius v1 engine...")
    labels = predict_possession_markius(
        kinematics=kinematics,
        st_home=st_home,
        st_away=st_away,
        st_set_pieces=st_set_pieces,
        n_frames=n_frames,
        half_blocks=half_blocks,
        kickoff_2_frame=kickoff_2_frame,
        kappa_thresh=0.95,
        max_gap=25.0,
        penalty_slope=0.4,
        K_turnover=11,
        max_transit=2,
        same_player_ratio=0.50,
        v_rel_thresh=3.5,
        speed_window=60,
        dead_speed_thresh=1.40,
        min_dead_len=90,
    )

    out_path = os.path.join(ROOT, out_csv) if not os.path.isabs(out_csv) else out_csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(labels))

    print(f"Submission saved successfully to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Markius v1: Clean & Legal Possession Model")
    parser.add_argument("--game", default="BAR-ATH", help="Game key (default: BAR-ATH)")
    parser.add_argument("--out", default="evaluation/results/markius_v1.csv", help="Output submission CSV path")
    parser.add_argument("--data-dir", default=None, help="Path to data directory")
    args = parser.parse_args()

    run_pipeline(game=args.game, out_csv=args.out, data_dir=args.data_dir)
