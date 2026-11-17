#!/usr/bin/env python3
"""Markius v4: robust control persistence and confidence-aware flight timing.

v4 keeps v3's tracking features, dead-ball cues, and pitch-control model, while fixing
three temporal edge cases in the possession layer:

* imputed-ball confidence decays exponentially and can never invert into negative reach;
* a challenger keeps its accumulated evidence through a short loose-ball interval;
* flight retiming cannot jump across an intervening possession change.
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

HOME, AWAY, NOBODY = foundation.HOME, foundation.AWAY, foundation.NOBODY


def retime_flight_midpoint_v4(labels, per_frame_contact, max_flight=150, lookback=250):
    """Move clean direct turnovers to flight midpoints without crossing another change."""
    out = labels.copy()
    n = len(labels)
    team_of = {"H": HOME, "A": AWAY}
    moved = []
    changes = [
        i for i in range(1, n)
        if labels[i] in team_of and labels[i - 1] in team_of and labels[i] != labels[i - 1]
    ]

    for frame in changes:
        old = team_of[labels[frame - 1]]
        new = team_of[labels[frame]]
        lo = max(0, frame - lookback)
        window = per_frame_contact[lo:min(n, frame + max_flight)]
        old_idx = np.flatnonzero(window == old)
        old_idx = old_idx[old_idx + lo < frame]
        if not len(old_idx):
            continue
        last_old = int(old_idx[-1]) + lo
        after = per_frame_contact[last_old + 1:min(n, frame + max_flight)]
        new_idx = np.flatnonzero(after == new)
        if not len(new_idx):
            continue
        first_new = last_old + 1 + int(new_idx[0])
        if first_new - last_old > max_flight:
            continue

        target = (last_old + 1 + first_new) // 2
        left, right = sorted((frame, target))
        if np.any((labels[left:right] == "D") | (labels[left:right] == "X")):
            continue

        old_label, new_label = labels[frame - 1], labels[frame]
        # Only alter the uninterrupted old/new spell belonging to this turnover.
        if target < frame:
            if not np.all(labels[target:frame] == old_label):
                continue
            out[target:frame] = new_label
        elif target > frame:
            if not np.all(labels[frame:target] == new_label):
                continue
            out[frame:target] = old_label
        moved.append(target - frame)

    return out, moved


def compute_pitch_control_count_capped(kinematics, fps=25.0, expected_team_size=11):
    """Compute v3 pitch control while limiting influence from duplicate team tracks.

    The expected roster size is a physical bound, not an estimate from possession labels.
    When the tracker reports more than 11 simultaneous slots for a team, its summed influence
    is scaled back to an 11-player equivalent. Missing slots are not upweighted.
    """
    px = kinematics["px"]
    py = kinematics["py"]
    bx = kinematics["ball_x_recon"]
    by = kinematics["ball_y_recon"]
    team_of_track = kinematics["team_of_track"]
    dt = 1.0 / fps

    vx = np.gradient(px, axis=0) / dt
    vy = np.gradient(py, axis=0) / dt
    dx = bx[:, None] - px
    dy = by[:, None] - py
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 0.05)
    ux, uy = dx / dist_safe, dy / dist_safe
    v_app = vx * ux + vy * uy
    v_app_fwd = np.clip(np.nan_to_num(v_app, nan=0.0), 0.0, 8.2)
    tau_react = 0.22 * (1.0 - 0.70 * (v_app_fwd / 8.2))
    react_x = px + vx * tau_react
    react_y = py + vy * tau_react
    d_remain = np.sqrt((bx[:, None] - react_x) ** 2 + (by[:, None] - react_y) ** 2)
    tau_arr = tau_react + np.maximum(0.0, d_remain - 0.75) / 8.2
    potency = np.nan_to_num(
        np.exp(-tau_arr / 0.35) + 1.6 * np.exp(-(dist ** 2) / (2.0 * 1.35 ** 2)),
        nan=0.0,
    )

    valid = np.isfinite(px) & np.isfinite(py)
    home = team_of_track == HOME
    away = team_of_track == AWAY
    n_home = np.sum(valid[:, home], axis=1)
    n_away = np.sum(valid[:, away], axis=1)
    home_scale = np.minimum(1.0, expected_team_size / np.maximum(n_home, 1))
    away_scale = np.minimum(1.0, expected_team_size / np.maximum(n_away, 1))
    psi_home = np.sum(potency * home[None, :], axis=1) * home_scale
    psi_away = np.sum(potency * away[None, :], axis=1) * away_scale
    loose = np.exp(-1.60 / 0.35)
    denom = psi_home + psi_away + loose
    return psi_home / denom, psi_away / denom, loose / denom


def compute_multi_cue_dead_ball_v4(
    kinematics, st_data, half_blocks, kickoff_2_frame, pitch=(105.3, 68.0), fps=25.0
):
    """v3 dead-ball cues with a coverage gate on weak missing-ball evidence."""
    bx = kinematics["ball_x_recon"]
    by = kinematics["ball_y_recon"]
    mean_speed = kinematics["mean_speed"]
    n_frames = len(bx)
    length, width = pitch
    dt = 1.0 / fps
    half1_start, half1_end = half_blocks[0][1], half_blocks[0][2]
    half2_start, half2_end = half_blocks[1][1], half_blocks[1][2]
    is_dead = np.zeros(n_frames, dtype=bool)

    # Exact half and halftime boundaries.
    is_dead[half2_start:kickoff_2_frame] = True

    outside = (~np.isnan(bx)) & (
        (bx < -0.2) | (bx > length + 0.2) | (by < -0.2) | (by > width + 0.2)
    )
    is_dead |= foundation.filter_min_duration(outside, 15)

    # Ball crossing a boundary immediately before a tracking disappearance.
    missing = np.isnan(bx)
    missing_edges = np.diff(np.concatenate(([0], missing.view(np.int8), [0])))
    for vanish in np.flatnonzero(missing_edges == 1):
        if vanish < 3:
            continue
        x_prev, y_prev = bx[vanish - 3:vanish], by[vanish - 3:vanish]
        if np.any(np.isnan(x_prev)) or np.any(np.isnan(y_prev)):
            continue
        vx = (x_prev[-1] - x_prev[0]) / (2.0 * dt)
        vy = (y_prev[-1] - y_prev[0]) / (2.0 * dt)
        if np.hypot(vx, vy) < 1.5:
            continue
        x0, y0 = x_prev[-1], y_prev[-1]
        cross_frames = 999
        if vx < -1.0 and x0 < 4.0:
            cross_frames = int(round(x0 / abs(vx) * fps))
        elif vx > 1.0 and (length - x0) < 4.0:
            cross_frames = int(round((length - x0) / vx * fps))
        elif vy < -1.0 and y0 < 4.0:
            cross_frames = int(round(y0 / abs(vy) * fps))
        elif vy > 1.0 and (width - y0) < 4.0:
            cross_frames = int(round((width - y0) / vy * fps))
        if cross_frames <= 25:
            exit_frame = min(n_frames - 1, vanish + cross_frames)
            is_dead[exit_frame:min(n_frames, exit_frame + 60)] = True

    # Missing speed samples become zero in the v3 aggregate. Smooth only frames with enough
    # observed speed values, then require support across the window before using it as evidence.
    player_count = np.sum(
        np.isfinite(kinematics["px"]) & np.isfinite(kinematics["py"]), axis=1
    )
    if "player_speeds" in kinematics:
        speed_count = np.sum(np.isfinite(kinematics["player_speeds"]), axis=1)
    else:
        speed_count = player_count
    speed_frame_valid = speed_count >= 6
    speed_kernel = np.ones(60, dtype=np.float32)
    speed_support = np.convolve(speed_frame_valid.astype(np.float32), speed_kernel, mode="same")
    speed_total = np.convolve(
        np.where(speed_frame_valid, mean_speed, 0.0).astype(np.float32), speed_kernel, mode="same"
    )
    smooth_speed = np.divide(
        speed_total, np.maximum(speed_support, 1.0), dtype=np.float32
    )
    reliable_speed = speed_support >= 30
    adequate_coverage = (player_count >= 6) & reliable_speed
    missing_stoppage = missing & (smooth_speed < 1.9) & adequate_coverage
    missing_stoppage = foundation.filter_min_duration(missing_stoppage, 90)
    miss_edges = np.diff(np.concatenate(([0], missing_stoppage.view(np.int8), [0])))
    starts = np.flatnonzero(miss_edges == 1)
    ends = np.flatnonzero(miss_edges == -1)
    for start, _ in zip(starts, ends):
        back = start
        while back > max(0, start - 60) and np.isnan(bx[back - 1]):
            back -= 1
        missing_stoppage[back:start] = True
    is_dead |= missing_stoppage

    offensive = st_data["st_home"] | st_data["st_away"]
    ultra_slow = (
        (smooth_speed < 1.40)
        & (smooth_speed > 0)
        & reliable_speed
        & (player_count >= 6)
        & (~offensive)
    )
    is_dead |= foundation.filter_min_duration(ultra_slow, 90)

    is_dead[:half1_start] = False
    is_dead[half1_end:half2_start] = False
    is_dead[half2_end:] = False
    return is_dead


def postprocess_possessions_v4(
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
    pitch_length=105.3,
    half1_end=999999,
    min_dwell_single=25,
    min_dwell_pass=12,
    terminal_max_frames=45,
):
    """Refine possession sequences using physical clearance and deflection principles.

    1. Clearance-flight return: when a defender executes a defensive clearance kick,
       their possession covers the strike itself. If the defending team does not retain
       proximity (> 1.5m) and the opponent dominates pitch control (p_opp > p_def),
       possession returns to the opponent after the flight.
    2. Deflection filtering: an opponent excursion sandwiched between the same incumbent team
       (opp -> curr -> opp) with no sustained dwell (>= 25 frames), no completed pass
       (>= 2 teammates with >= 12f control), and no defensive clearance kick reverts to the
       incumbent team, honoring CHALLENGE.md rules on non-intercepting touches.
    3. Terminal touches: brief non-clearance touches before a dead ball (< 45 frames) revert
       to the attacking incumbent team who had possession before the stoppage.
    """
    out = labels.copy()
    n = len(labels)
    midfield_x = pitch_length / 2.0

    def get_spells(lbls):
        sp = []
        i = 0
        while i < n:
            if lbls[i] in ("H", "A"):
                s = i
                t = lbls[i]
                while i < n and lbls[i] == t:
                    i += 1
                sp.append((s, i, t))
            else:
                i += 1
        return sp

    spells = get_spells(out)

    # 1. Clearance-flight return
    for s, e, t in spells:
        if np.any(clearance_kicks[s:min(n, s + 15)]):
            opp_t = "A" if t == "H" else "H"
            def_eff = h_eff[s:e] if t == "H" else a_eff[s:e]
            p_def = p_home[s:e] if t == "H" else p_away[s:e]
            p_opp = p_away[s:e] if t == "H" else p_home[s:e]

            lost = np.flatnonzero((def_eff > 1.5) & (p_opp > p_def))
            if len(lost) and lost[0] >= 6 and (e - (s + lost[0])) >= 25:
                split_frame = s + lost[0]
                out[split_frame:e] = opp_t

    spells = get_spells(out)

    # 2. Deflection and terminal touch filtering
    for idx in range(len(spells)):
        curr_s, curr_e, curr_t = spells[idx]
        prev_t = spells[idx - 1][2] if idx > 0 and spells[idx - 1][1] == curr_s else None
        next_t = spells[idx + 1][2] if idx < len(spells) - 1 and spells[idx + 1][0] == curr_e else None

        opp = "A" if curr_t == "H" else "H"
        is_sandwiched = (prev_t == opp and next_t == opp)
        is_terminal = (prev_t == opp and next_t is None and curr_e < n and is_dead[curr_e])

        if is_sandwiched or is_terminal:
            eff = h_eff[curr_s:curr_e] if curr_t == "H" else a_eff[curr_s:curr_e]
            opp_eff = a_eff[curr_s:curr_e] if curr_t == "H" else h_eff[curr_s:curr_e]
            best_p = home_best[curr_s:curr_e] if curr_t == "H" else away_best[curr_s:curr_e]
            vb = v_ball[curr_s:curr_e]
            b_x = bx[curr_s:curr_e]

            mask = (~np.isnan(eff)) & (eff <= 0.95) & (best_p >= 0)
            p_ids, p_counts = np.unique(best_p[mask], return_counts=True)
            has_sustained_dwell = np.any(p_counts >= min_dwell_single) if len(p_counts) else False

            teammates_with_control = 0
            if len(p_counts) >= 2:
                for pid, cnt in zip(p_ids, p_counts):
                    if cnt >= min_dwell_pass:
                        p_mask = mask & (best_p == pid)
                        p_excl = p_mask & (opp_eff > 0.95)
                        if np.sum(p_excl) >= 4:
                            teammates_with_control += 1
            has_completed_pass = (teammates_with_control >= 2)

            has_clearance = False
            is_half1 = curr_s <= half1_end
            for k in range(len(vb)):
                if vb[k] >= 5.0:
                    pos_x = b_x[k]
                    if not np.isnan(pos_x):
                        if curr_t == "H":
                            in_def_half = (pos_x <= midfield_x + 5.0) if is_half1 else (pos_x >= midfield_x - 5.0)
                        else:
                            in_def_half = (pos_x >= midfield_x - 5.0) if is_half1 else (pos_x <= midfield_x + 5.0)
                        if in_def_half:
                            has_clearance = True
                            break

            if is_sandwiched and not has_sustained_dwell and not has_completed_pass and not has_clearance:
                out[curr_s:curr_e] = opp
            elif is_terminal and not has_sustained_dwell and not has_completed_pass and not has_clearance and (curr_e - curr_s < terminal_max_frames):
                out[curr_s:curr_e] = opp

    return out


def predict_possession_markius_v4(
    kinematics,
    st_data,
    half_blocks,
    kickoff_2_frame,
    pitch=(105.3, 68.0),
    fps=25.0,
    w_turnover=11,
    max_transit_seconds=0.32,
):
    """Infer H/A/D possession labels with bounded evidence persistence.

    The state machine follows v3's control and restart rules. During a live challenger
    episode, up to ``max_transit_seconds`` without a clear controller is treated as a
    loose-ball gap: it pauses the evidence counter instead of erasing the candidate.
    An incumbent-control observation, a dead-ball cue, or a longer gap still resets it.
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

    is_dead = compute_multi_cue_dead_ball_v4(
        kinematics, st_data, half_blocks, kickoff_2_frame, pitch=pitch, fps=fps
    )
    p_home, p_away, _ = compute_pitch_control_count_capped(kinematics, fps=fps)

    # Exponential confidence is strictly positive. The v3 linear expression crossed zero
    # on gaps longer than 2.5 seconds, making distant players appear to have negative reach.
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
    # SET PIECES tags describe broad phases (often many seconds), not the restart instant.
    # Restart impulse therefore comes from a short post-dead window or observed ball speed.

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
    cand_history = deque(maxlen=w_turnover)

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
            if p_home[frame] > p_away[frame] + 0.05:
                inst = HOME
            elif p_away[frame] > p_home[frame] + 0.05:
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

            # Fast defensive clearance kick impulse: requires only 4 frames if striking in defending half
            is_clearance_kick = False
            if opp_frames >= 4:
                is_half1 = frame <= half1_end
                pos_x = bx[frame]
                if not np.isnan(pos_x):
                    in_def_half = (
                        (pos_x <= midfield_x + 5.0)
                        if (opponent == HOME and is_half1) or (opponent == AWAY and not is_half1)
                        else (pos_x >= midfield_x - 5.0)
                    )
                    if in_def_half and v_ball[frame] >= 4.5:
                        dv = v_ball[frame] - v_ball[frame - 2] if frame >= 2 else 0.0
                        if dv >= 0.8 or v_ball[frame] >= 6.0:
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
            # Positive evidence for the incumbent invalidates the takeover candidate.
            opp_cand = NOBODY
            opp_frames = 0
            opp_start = -1
            transit_frames = 0
            cand_history.clear()
            state_seq[frame] = curr_state

    labels = np.full(n_frames, "D", dtype="<U1")
    labels[state_seq == HOME] = "H"
    labels[state_seq == AWAY] = "A"

    # Contacts used for timing must also pass the corrected confidence weighting.
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
        "Markius v4: turnover window={} frames, loose-ball tolerance={} frames, "
        "midpoint retimings={}.".format(w_turnover, max_transit, len(moved))
    )
    labels[~graded] = "X"
    return labels


def run_pipeline(
    game="BAR-ATH",
    out_csv="evaluation/results/markius_v4.csv",
    data_dir=None,
):
    """Run v4 on the tactical tracking and write the indexed submission CSV."""
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
    labels = predict_possession_markius_v4(
        kinematics=kinematics,
        st_data=st_data,
        half_blocks=half_blocks,
        kickoff_2_frame=kickoff_2_frame,
        pitch=pitch,
        fps=games.GAMES[game]["fps"],
    )

    out_path = out_csv if os.path.isabs(out_csv) else os.path.join(ROOT, out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    print("Wrote {} frames to {}".format(len(labels), out_path))
    return out_path
