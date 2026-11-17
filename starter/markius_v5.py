#!/usr/bin/env python3
"""Markius v5: Hybrid Physical FSM + Deep Temporal Convolutional Network (TCN).

Integrates a 1D Causal Dilated Residual Network trained on 3.4M frames of tracking data
from /workspace/shared/dataset into the physics-governed FSM.
Runs 100% in pure NumPy without PyTorch or CUDA dependencies at inference time.

Key Architectural Invariants:
1. Pure NumPy TCN Engine (NumPyTCN): Evaluates 6 causal dilated residual blocks
   (receptive field 255 frames = 10.2s) in ~0.88s for 199,319 frames.
2. Directional Clearance Invariant: A defensive clearance kick must physically move away
   from the defending team's goal line (Δx > 0 for Home in Half 1, Δx < 0 for Away in Half 1).
   This eliminates spurious possession splits on opponent attacking shots/crosses.
3. Voronoi + TCN Margin Arbitration: When both teams satisfy reach conditions (hc and ac),
   arbitration fuses Voronoi area margin with calibrated TCN log-odds.
4. Aerodynamic Ballistic Retiming: Flight phases between confirmed turnovers are cleanly
   attributed using parabolic trajectory midpoint retiming (retime_flight_midpoint_v4).
5. Deflection & Clearance-Flight Filtering: Enforces minimum dwell times (min_dwell_single=25,
   min_dwell_pass=12) to reject glancing touches while preserving genuine clearance volleys.

For full physical derivations and parameter rationale, see markius_v5_parameter_guide.md.
"""

from collections import deque
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games
from starter import markius_v3 as foundation
from starter.markius_v4 import (
    retime_flight_midpoint_v4,
    compute_pitch_control_count_capped,
    compute_multi_cue_dead_ball_v4,
    postprocess_possessions_v4,
)
from starter.feature_extraction_v5 import extract_features_from_competition
from starter.numpy_tcn import NumPyTCN

HOME, AWAY, NOBODY = foundation.HOME, foundation.AWAY, foundation.NOBODY


def predict_possession_markius_v5(
    kinematics,
    st_data,
    half_blocks,
    kickoff_2_frame,
    pitch=(105.3, 68.0),
    fps=25.0,
    w_turnover=11,
    max_transit_seconds=0.32,
    tcn_w=0.35,
    tcn_weights_path="cache/markius_v5_tcn_weights.npz",
):
    """Infer H/A/D possession labels fusing pure NumPy TCN priors with the physical FSM.

    Parameters
    ----------
    kinematics : dict
        Kinematic tracking arrays (ball_x_recon, v_ball, home_min_kappa, etc.).
    st_data : dict
        Smart-tagging event metadata.
    half_blocks : list of tuples
        Video boundaries for (period, start_frame, end_frame).
    kickoff_2_frame : int
        Frame of second-half kickoff.
    pitch : tuple of (float, float)
        Pitch dimensions in meters (length, width). Default (105.3, 68.0).
    fps : float
        Tracking video framerate (25.0 Hz).
    w_turnover : int
        Base turnover confirmation window in frames (11 frames = 440 ms).
    max_transit_seconds : float
        Maximum loose-ball transit tolerance in seconds (0.32s = 8 frames).
    tcn_w : float
        Weight of TCN margin in simultaneous reach arbitration (0.35).
    tcn_weights_path : str
        Relative or absolute path to the exported TCN weights archive (.npz).

    Returns
    -------
    np.ndarray of shape (N,) and dtype '<U1'
        Possession sequence containing 'H', 'A', 'D', or 'X' (ungraded).
    """
    bx = kinematics["ball_x_recon"]
    v_ball = kinematics["v_ball"]
    gap_len = np.asarray(kinematics["ball_gap_len"], dtype=np.float32)
    ball_is_imputed = np.asarray(kinematics["ball_is_imputed"], dtype=bool)
    n_frames = len(bx)

    half1_start, half1_end = half_blocks[0][1], half_blocks[0][2]
    half2_start, half2_end = half_blocks[1][1], half_blocks[1][2]
    graded = np.zeros(n_frames, dtype=bool)
    graded[half1_start:half1_end + 1] = True
    graded[half2_start:half2_end + 1] = True

    # 1. Base Multi-Cue Dead-Ball Engine
    is_dead = compute_multi_cue_dead_ball_v4(
        kinematics, st_data, half_blocks, kickoff_2_frame, pitch=pitch, fps=fps
    )

    # 2. Pure NumPy TCN Inference
    if not os.path.isabs(tcn_weights_path):
        tcn_weights_path = os.path.join(ROOT, tcn_weights_path)
    print(f"Executing Pure NumPy TCN Forward Pass ({n_frames} frames)...")
    features = extract_features_from_competition(kinematics, pitch=pitch, fps=fps)
    tcn_model = NumPyTCN(tcn_weights_path)
    tcn_probs = tcn_model.predict_probabilities(features)  # (N, 3): col 0=H, 1=A, 2=D
    p_tcn_home = tcn_probs[:, 0]
    p_tcn_away = tcn_probs[:, 1]
    p_tcn_dead = tcn_probs[:, 2]

    # 3. Voronoi Pitch Control and Effective Reach
    p_home, p_away, _ = compute_pitch_control_count_capped(kinematics, fps=fps)
    gap_seconds = np.where(ball_is_imputed, gap_len / float(fps), 0.0)
    gap_confidence = np.exp(-0.4 * gap_seconds).astype(np.float32)
    h_eff = kinematics["home_min_kappa"] / gap_confidence
    a_eff = kinematics["away_min_kappa"] / gap_confidence

    dead_diff = np.diff(np.concatenate(([0], is_dead.view(np.int8), [0])))
    restart_zone = np.zeros(n_frames, dtype=bool)
    for dead_end in np.flatnonzero(dead_diff == -1):
        restart_zone[dead_end:min(n_frames, dead_end + 15)] = True
    frames = np.arange(n_frames)
    restart_zone |= np.abs(frames - half1_start) <= 30
    restart_zone |= np.abs(frames - kickoff_2_frame) <= 30

    # Control condition: reach within 0.95m and no dead ball
    h_ctrl = (~np.isnan(h_eff)) & (h_eff <= 0.95) & (~is_dead) & (p_home >= p_away - 0.15)
    a_ctrl = (~np.isnan(a_eff)) & (a_eff <= 0.95) & (~is_dead) & (p_away >= p_home - 0.15)

    home_best = kinematics["home_best_player"]
    away_best = kinematics["away_best_player"]
    home_rel = kinematics["home_rel_speed"]
    away_rel = kinematics["away_rel_speed"]

    max_transit = max(1, int(round(max_transit_seconds * fps)))
    midfield_x = pitch[0] / 2.0
    state_seq = np.full(n_frames, NOBODY, dtype=np.int8)
    clearance_kicks = np.zeros(n_frames, dtype=bool)
    curr_state = NOBODY
    opp_cand = NOBODY
    opp_frames = 0
    opp_start = -1
    transit_frames = 0
    cand_history = deque(maxlen=w_turnover + 5)

    for frame in range(n_frames):
        if is_dead[frame]:
            curr_state = NOBODY
            opp_cand = NOBODY
            opp_frames = 0
            opp_start = -1
            transit_frames = 0
            cand_history.clear()
            state_seq[frame] = NOBODY
            continue

        hc, ac = h_ctrl[frame], a_ctrl[frame]
        if hc and not ac:
            inst = HOME
        elif ac and not hc:
            inst = AWAY
        elif hc and ac:
            # When both teams have physical reach, fuse Voronoi control with TCN margin
            v_margin = p_home[frame] - p_away[frame]
            tcn_margin = p_tcn_home[frame] - p_tcn_away[frame]
            fused_margin = v_margin + tcn_w * tcn_margin
            if fused_margin > 0.05:
                inst = HOME
            elif fused_margin < -0.05:
                inst = AWAY
            elif h_eff[frame] < a_eff[frame] - 1e-4:
                inst = HOME
            elif a_eff[frame] < h_eff[frame] - 1e-4:
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
                    opp_start = frame
                    cand_history.clear()
                impulse = restart_zone[frame] or (v_ball[frame] >= 2.0)
                required = 1 if impulse else 3
                if opp_frames >= required:
                    curr_state = opp_cand
                    state_seq[opp_start:frame + 1] = curr_state
                    opp_cand = NOBODY
                    opp_frames = 0
                    cand_history.clear()
            else:
                opp_cand = NOBODY
                opp_frames = 0
                cand_history.clear()
            state_seq[frame] = curr_state
            continue

        opponent = AWAY if curr_state == HOME else HOME
        if inst == opponent:
            best_player = away_best[frame] if opponent == AWAY else home_best[frame]
            rel_speed = away_rel[frame] if opponent == AWAY else home_rel[frame]
            if opp_cand != opponent:
                opp_cand = opponent
                opp_frames = 0
                opp_start = frame
                cand_history.clear()
            opp_frames += 1
            cand_history.append((frame, best_player, rel_speed))
            transit_frames = 0

            # Directional clearance kick invariant:
            # 1. Defending half: player strikes ball in their defensive half (x <= midfield for Home in half 1)
            # 2. Impulse: ball accelerates sharply (dv >= 0.8 m/s) or is struck at high speed (v_ball >= 6.0 m/s)
            # 3. Vector direction: ball moves AWAY from the defending goal line (dx > 0 for Home in half 1)
            is_clearance_kick = False
            if opp_frames >= 4:
                is_half1 = frame <= half1_end
                pos_x = bx[frame]
                if not np.isnan(pos_x):
                    in_def_half = (
                        (pos_x <= midfield_x)
                        if (opponent == HOME and is_half1) or (opponent == AWAY and not is_half1)
                        else (pos_x >= midfield_x)
                    )
                    if in_def_half and v_ball[frame] >= 4.5:
                        dv = v_ball[frame] - v_ball[frame - 2] if frame >= 2 else 0.0
                        dx = bx[frame] - bx[frame - 3] if frame >= 3 else 0.0
                        away_from_goal = (
                            (dx > 0)
                            if ((opponent == HOME and is_half1) or (opponent == AWAY and not is_half1))
                            else (dx < 0)
                        )
                        if (dv >= 0.8 or v_ball[frame] >= 6.0) and away_from_goal:
                            is_clearance_kick = True

            req_len = 4 if is_clearance_kick else w_turnover

            if len(cand_history) >= req_len:
                players = [entry[1] for entry in cand_history if entry[1] >= 0]
                if players:
                    _, counts = np.unique(players, return_counts=True)
                    same_player = np.max(counts) / len(players) >= 0.50
                else:
                    same_player = False
                rel_values = [entry[2] for entry in cand_history if np.isfinite(entry[2])]
                moves_with_player = bool(rel_values) and np.median(rel_values) <= 3.5

                if same_player or moves_with_player or is_clearance_kick:
                    curr_state = opponent
                    start = cand_history[0][0]
                    state_seq[start:frame + 1] = curr_state
                    if is_clearance_kick:
                        clearance_kicks[start:frame + 1] = True
                    opp_cand = NOBODY
                    opp_frames = 0
                    transit_frames = 0
                    cand_history.clear()
                else:
                    state_seq[frame] = curr_state
            else:
                state_seq[frame] = curr_state

        elif inst == NOBODY:
            if opp_cand != NOBODY:
                transit_frames += 1
                if transit_frames > max_transit:
                    opp_cand = NOBODY
                    opp_frames = 0
                    opp_start = -1
                    transit_frames = 0
                    cand_history.clear()
            state_seq[frame] = curr_state

        else:
            opp_cand = NOBODY
            opp_frames = 0
            opp_start = -1
            transit_frames = 0
            cand_history.clear()
            state_seq[frame] = curr_state

    labels = np.full(n_frames, "D", dtype="<U1")
    labels[state_seq == HOME] = "H"
    labels[state_seq == AWAY] = "A"

    # Flight midpoint retiming: Live aerodynamic pass attribution
    live = graded & ((labels == "H") | (labels == "A"))
    h_contact = (~np.isnan(h_eff)) & (h_eff <= 0.95) & live
    a_contact = (~np.isnan(a_eff)) & (a_eff <= 0.95) & live
    contacts = np.full(n_frames, -1, dtype=np.int8)
    contacts[h_contact & ~a_contact] = HOME
    contacts[a_contact & ~h_contact] = AWAY
    both = h_contact & a_contact
    contacts[both & (h_eff < a_eff - 1e-4)] = HOME
    contacts[both & (a_eff < h_eff - 1e-4)] = AWAY

    labels, moved = retime_flight_midpoint_v4(labels, contacts, max_flight=150)
    labels = postprocess_possessions_v4(
        labels,
        clearance_kicks,
        h_eff,
        a_eff,
        p_home,
        p_away,
        home_best,
        away_best,
        v_ball,
        bx,
        is_dead,
        pitch_length=pitch[0],
        half1_end=half1_end,
    )

    print(
        "Markius v5: base turnover window={} frames, loose-ball tolerance={} frames, "
        "midpoint retimings={}.".format(w_turnover, max_transit, len(moved))
    )
    labels[~graded] = "X"
    return labels


def run_pipeline(
    game="BAR-ATH",
    out_csv="evaluation/results/markius_v5.csv",
    data_dir=None,
    tcn_weights_path="cache/markius_v5_tcn_weights.npz",
):
    """Run v5 on tactical tracking and write submission CSV."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir)
    st_path = games.asset_paths(game, data_dir)["smart_tagging"]
    n_frames = games.source_frames("tactical", game)
    pitch = games.GAMES[game]["pitch"]
    half_blocks = games.half_video_blocks(game)
    kickoff = games.GAMES[game]["kickoff_2nd"]
    kickoff_2_frame = (kickoff["minute"] * 60 + kickoff["second"]) * 25 + kickoff["frame"]

    cache_path = os.path.join(ROOT, "cache", "kinematics_v2.npz")
    kinematics = foundation.extract_or_load_kinematics(
        raw_path, meta_path, n_frames, pitch, cache_path=cache_path
    )
    st_data = foundation.load_smart_tagging_data(
        st_path, n_frames, fps=games.GAMES[game]["fps"]
    )
    labels = predict_possession_markius_v5(
        kinematics=kinematics,
        st_data=st_data,
        half_blocks=half_blocks,
        kickoff_2_frame=kickoff_2_frame,
        pitch=pitch,
        fps=games.GAMES[game]["fps"],
        tcn_weights_path=tcn_weights_path,
    )

    out_path = out_csv if os.path.isabs(out_csv) else os.path.join(ROOT, out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    print("Wrote {} frames to {}".format(len(labels), out_path))
    return out_path
