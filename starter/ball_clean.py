#!/usr/bin/env python3
"""Clean the automatic ball track before deciding possession.

The ground truth comes from corrected tracking; the input is automatic and uncorrected. Where the
automatic ball is in the wrong place, no possession rule can recover. This module removes the parts
of the ball track that cannot be the ball, using physics only:

  - jumps: the ball cannot move more than MAX_STEP metres between two frames (40 m/s, faster than
    any shot); a jump splits the track into tracklets;
  - islands: a tracklet shorter than MIN_TRACKLET frames, bounded by jumps on both sides, is a
    false detection between two pieces of the real ball and is removed;
  - stuck: a tracklet that stays within STUCK_RADIUS metres for STUCK_FRAMES or more, with no
    player within STUCK_CLEAR metres at any point, is a mark on the pitch rather than the ball.

It also rebuilds markius_v1's reach features from any ball track, so the same model can run on the
cleaned ball. The rebuild is checked against the cache before it is trusted.
"""

import numpy as np

FPS = 25
MAX_STEP = 40.0 / FPS   # 1.6 m per frame
MIN_TRACKLET = 10       # frames (0.4 s)
STUCK_RADIUS = 0.3      # metres
STUCK_FRAMES = 50       # frames (2 s)
STUCK_CLEAR = 5.0       # metres


def tracklets(bx, by, max_step=MAX_STEP):
    """[(start, stop)] maximal runs of present ball with no jump larger than max_step."""
    present = ~np.isnan(bx)
    step = np.hypot(np.diff(bx), np.diff(by))
    jump = np.concatenate(([False], np.nan_to_num(step, nan=0.0) > max_step))
    out, start = [], None
    for i in range(len(bx)):
        if not present[i]:
            if start is not None:
                out.append((start, i))
                start = None
            continue
        if start is None:
            start = i
        elif jump[i]:
            out.append((start, i))
            start = i
    if start is not None:
        out.append((start, len(bx)))
    return out


def clean(bx, by, d_nearest, cfg=None):
    """Return a cleaned copy of the ball track and a report of what was removed."""
    cfg = {"islands": True, "stuck": True, **(cfg or {})}
    bx, by = bx.copy(), by.copy()
    pieces = tracklets(bx, by)
    report = {"tracklets": len(pieces), "jumps": 0, "island_frames": 0, "islands": 0,
              "stuck_frames": 0, "stuck": 0}
    for j, (s, e) in enumerate(pieces):
        jumped_in = j > 0 and pieces[j - 1][1] == s
        jumped_out = j + 1 < len(pieces) and pieces[j + 1][0] == e
        report["jumps"] += int(jumped_in)
        if cfg["islands"] and e - s < MIN_TRACKLET and (jumped_in or jumped_out):
            bx[s:e] = np.nan
            by[s:e] = np.nan
            report["islands"] += 1
            report["island_frames"] += e - s
            continue
        if cfg["stuck"] and e - s >= STUCK_FRAMES:
            cx, cy = np.nanmedian(bx[s:e]), np.nanmedian(by[s:e])
            spread = np.nanmax(np.hypot(bx[s:e] - cx, by[s:e] - cy))
            near = d_nearest[s:e]
            if spread <= STUCK_RADIUS and np.all(np.isnan(near) | (near > STUCK_CLEAR)):
                bx[s:e] = np.nan
                by[s:e] = np.nan
                report["stuck"] += 1
                report["stuck_frames"] += e - s
    return bx, by, report


# -- markius_v1's reach features, rebuilt from any ball track ------------------------------------

def compute_velocity_1d(x, dt=1.0 / 25.0):
    """Copied from markius_v1 so the rebuild matches its cache."""
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


def interpolate_short_gaps(ball_x_raw, ball_y_raw, length):
    """markius_v1's reconstruction: linear across gaps of <= 25 frames, endpoints on the pitch."""
    n = len(ball_x_raw)
    bx = ball_x_raw.copy()
    by = ball_y_raw.copy()
    imputed = np.zeros(n, dtype=bool)
    gap_len = np.zeros(n, dtype=np.int16)
    nan_b = np.isnan(bx)
    diff = np.diff(np.concatenate(([0], nan_b.view(np.int8), [0])))
    for s, e in zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)):
        g = e - s
        if g <= 25 and s > 0 and e < n:
            if not np.isnan(bx[s - 1]) and not np.isnan(bx[e]):
                if (-1.0 <= bx[s - 1] <= length + 1.0) and (-1.0 <= bx[e] <= length + 1.0):
                    bx[s:e] = np.linspace(bx[s - 1], bx[e], g + 2)[1:-1]
                    by[s:e] = np.linspace(by[s - 1], by[e], g + 2)[1:-1]
                    imputed[s:e] = True
                    gap_len[s:e] = g
    return bx, by, imputed, gap_len


def reach_features(ball_x_raw, ball_y_raw, px, py, team_of_track, pitch, mean_speed):
    """The kinematics dict principled_v1.predict expects, from a given raw ball track."""
    length, _ = pitch
    bx, by, imputed, gap_len = interpolate_short_gaps(ball_x_raw, ball_y_raw, length)
    v_bx = compute_velocity_1d(bx)
    v_by = compute_velocity_1d(by)
    out = {"ball_x_raw": ball_x_raw, "ball_y_raw": ball_y_raw, "ball_x_recon": bx,
           "ball_y_recon": by, "ball_is_imputed": imputed, "ball_gap_len": gap_len,
           "mean_speed": mean_speed, "team_of_track": team_of_track}
    r0, tau_stride, max_reach_spd = 0.85, 0.25, 8.5
    for code, name in ((0, "home"), (1, "away")):
        slots = np.where(team_of_track == code)[0]
        t_px, t_py = px[:, slots], py[:, slots]
        dx = bx[:, None] - t_px
        dy = by[:, None] - t_py
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 0.05)
        u_x, u_y = dx / dist_safe, dy / dist_safe
        vx = np.gradient(t_px, axis=0) * 25.0
        vy = np.gradient(t_py, axis=0) * 25.0
        v_app = np.clip(np.nan_to_num(vx * u_x + vy * u_y, nan=0.0), 0.0, max_reach_spd)
        kappa = dist / np.maximum(r0 + tau_stride * v_app, 0.1)
        best = np.argmin(np.where(np.isnan(kappa), 1e9, kappa), axis=1)
        min_kappa = np.take_along_axis(kappa, best[:, None], axis=1).squeeze(1)
        bvx = np.take_along_axis(vx, best[:, None], axis=1).squeeze(1)
        bvy = np.take_along_axis(vy, best[:, None], axis=1).squeeze(1)
        rel = np.sqrt((bvx - v_bx) ** 2 + (bvy - v_by) ** 2)
        all_nan = np.all(np.isnan(kappa), axis=1)
        player = slots.astype(np.int16)[best]
        player[all_nan] = -1
        min_kappa[all_nan] = 1e9
        rel[all_nan] = np.nan
        out[f"{name}_min_kappa"] = min_kappa
        out[f"{name}_best_player"] = player
        out[f"{name}_rel_speed"] = rel.astype(np.float32)
    return out
