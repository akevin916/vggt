"""Pose trajectory metrics (ATE / RPE) via evo."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D


def extrinsics_w2c_to_tum(extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (S,3,4) OpenCV world-to-cam to TUM (S,7) xyz+wxyz and timestamps."""
    tum, stamps = [], []
    for i, ext in enumerate(extrinsics):
        w2c = np.vstack([ext, np.array([0, 0, 0, 1], dtype=np.float64)])
        c2w = np.linalg.inv(w2c)
        xyz = c2w[:3, 3]
        quat_xyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        tum.append(np.concatenate([xyz, wxyz]))
        stamps.append(float(i))
    tum = np.stack(tum, axis=0)
    tum[:, :3] -= tum[:, :3].mean(axis=0, keepdims=True)
    tt = np.expand_dims(np.array(stamps, dtype=np.float64), -1)
    return tum, tt


def _make_traj(traj: np.ndarray, tstamps: np.ndarray) -> PoseTrajectory3D:
    return PoseTrajectory3D(
        positions_xyz=traj[:, :3],
        orientations_quat_wxyz=traj[:, 3:],
        timestamps=tstamps,
    )


def eval_pose_metrics(
    pred_extrinsics: np.ndarray,
    gt_tum: np.ndarray,
    gt_timestamps: np.ndarray,
) -> Dict[str, float]:
    pred_tum, pred_ts = extrinsics_w2c_to_tum(pred_extrinsics)

    pred_traj = _make_traj(pred_tum, pred_ts)
    gt_traj = _make_traj(gt_tum, gt_timestamps)

    n = min(pred_traj.num_poses, gt_traj.num_poses)
    if pred_traj.num_poses != gt_traj.num_poses:
        pred_traj = PoseTrajectory3D(
            positions_xyz=pred_traj.positions_xyz[:n],
            orientations_quat_wxyz=pred_traj.orientations_quat_wxyz[:n],
            timestamps=gt_traj.timestamps[:n],
        )
        gt_traj = PoseTrajectory3D(
            positions_xyz=gt_traj.positions_xyz[:n],
            orientations_quat_wxyz=gt_traj.orientations_quat_wxyz[:n],
            timestamps=gt_traj.timestamps[:n],
        )
    else:
        pred_traj.timestamps = gt_traj.timestamps

    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    ate_result = main_ape.ape(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    rpe_rot_result = main_rpe.rpe(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )
    rpe_trans_result = main_rpe.rpe(
        gt_traj,
        pred_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
    )

    return {
        "ate": float(ate_result.stats["rmse"]),
        "rpe_trans": float(rpe_trans_result.stats["rmse"]),
        "rpe_rot": float(rpe_rot_result.stats["rmse"]),
    }


def max_pose_depth_delta(
    pred_a: Dict[str, np.ndarray],
    pred_b: Dict[str, np.ndarray],
) -> Dict[str, float]:
    dd = float(np.abs(pred_a["depth"] - pred_b["depth"]).max())
    dp = float(np.abs(pred_a["pose_enc"] - pred_b["pose_enc"]).max())
    return {"max_delta_depth": dd, "max_delta_pose_enc": dp}


# ---------------------------------------------------------------------------
# AF-SfMLearner snippet ATE (reference/AF-SfMLearner/evaluate_pose.py)
#
# The endoscopy pose tables (AF-SfMLearner Table 10, and every paper that cites its numbers)
# are NOT full-trajectory ATE. A window of ``track_length`` frames slides over the whole
# sequence; each window is re-anchored at its own origin, re-scaled by its own least-squares
# factor, and the per-window errors are averaged. Drift therefore never accumulates past 5
# frames, and the numbers come out an order of magnitude below a full-sequence ATE -- the two
# must never share a table column. ``eval_pose_metrics`` above is the full-sequence metric,
# which is what EndoSfM3D's own code computes; this is what its cited baselines computed.
#
# Ported verbatim, including the two quirks: the error divides by N rather than sqrt(N)
# (so it is not an RMSE despite the name), and the scale is fitted per window.
# ---------------------------------------------------------------------------

SNIPPET_TRACK_LENGTH = 5


def _local_xyz(c2w: np.ndarray, i: int, n: int) -> np.ndarray:
    """Camera centres of frames i..i+n-1 expressed in frame i's coordinates.

    Identical to AF's ``dump_xyz`` over the relative transforms of that window: it starts at
    the origin and composes T_j = c2w_j^-1 @ c2w_{j+1}, which telescopes to c2w_i^-1 @ c2w_j.
    Building it from the absolute poses instead sidesteps their pose-network sign convention.
    """
    inv_i = np.linalg.inv(c2w[i])
    return np.stack([(inv_i @ c2w[j])[:3, 3] for j in range(i, i + n)])


def _snippet_ate(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> float:
    """AF's ``compute_ate``: anchor, fit one scale, divide by N (not sqrt(N))."""
    pred = pred_xyz + (gt_xyz[0] - pred_xyz[0])[None, :]
    denom = np.sum(pred ** 2)
    scale = np.sum(gt_xyz * pred) / denom if denom > 0 else 1.0
    err = pred * scale - gt_xyz
    return float(np.sqrt(np.sum(err ** 2)) / gt_xyz.shape[0])


def _snippet_rot_err(gt_c2w: np.ndarray, pred_c2w: np.ndarray, i: int, n: int) -> float:
    """AF's ``compute_re``: mean residual rotation angle over the window."""
    tot = 0.0
    gi, pi = np.linalg.inv(gt_c2w[i]), np.linalg.inv(pred_c2w[i])
    for j in range(i, i + n):
        R_gt, R_pr = (gi @ gt_c2w[j])[:3, :3], (pi @ pred_c2w[j])[:3, :3]
        R = R_gt @ np.linalg.inv(R_pr)
        s = np.linalg.norm([R[0, 1] - R[1, 0], R[1, 2] - R[2, 1], R[0, 2] - R[2, 0]])
        tot += float(np.arctan2(s, np.trace(R) - 1))
    return tot / n


def snippet_pose_metrics(
    pred_extrinsics: np.ndarray,
    gt_extrinsics: np.ndarray,
    track_length: int = SNIPPET_TRACK_LENGTH,
) -> Dict[str, float]:
    """AF-SfMLearner snippet ATE/RE over every ``track_length`` window of the sequence.

    Both inputs are (S,3,4) or (S,4,4) OpenCV **world-to-cam**, one row per frame, in
    sequence order and frame-aligned with each other.
    """
    def to_c2w(E: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=np.float64)
        if E.shape[-2] == 3:
            bottom = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(E), 1, 1))
            E = np.concatenate([E, bottom], axis=1)
        return np.linalg.inv(E)

    gt_c2w, pred_c2w = to_c2w(gt_extrinsics), to_c2w(pred_extrinsics)
    if len(gt_c2w) != len(pred_c2w):
        raise ValueError(f"frame count mismatch: gt {len(gt_c2w)} vs pred {len(pred_c2w)}")
    # Window bookkeeping copied from AF rather than idealised. It iterates
    # ``for i in range(0, num_frames - 1)`` over num_frames = S-1 RELATIVE poses and slices
    # ``[i : i + track_length - 1]``, so the last few windows run off the end and are
    # TRUNCATED (4, 3, 2 relative poses) instead of being dropped. That yields S-2 windows,
    # not the S-track_length+1 full ones. Two windows' worth of difference is immaterial to
    # the mean, but matching it exactly is what lets us claim the number is comparable.
    n = int(track_length)
    n_rel = len(gt_c2w) - 1
    if n_rel < 2:
        raise ValueError(f"sequence of {len(gt_c2w)} frames is too short for snippet ATE")
    spans = [(i, min(i + n - 1, n_rel) - i + 1) for i in range(n_rel - 1)]

    ates = [_snippet_ate(_local_xyz(gt_c2w, i, k), _local_xyz(pred_c2w, i, k))
            for i, k in spans]
    res = [_snippet_rot_err(gt_c2w, pred_c2w, i, k) for i, k in spans]
    return dict(snippet_ate=float(np.mean(ates)), snippet_ate_std=float(np.std(ates)),
                snippet_rot=float(np.mean(res)), snippet_rot_std=float(np.std(res)),
                n_windows=len(spans), track_length=n)


def _snippet_spans(n_frames: int, track_length: int):
    """AF's window list: (start, length) per window, tail windows truncated, S-2 of them."""
    n_rel = n_frames - 1
    if n_rel < 2:
        raise ValueError(f"sequence of {n_frames} frames is too short for snippet ATE")
    return [(i, min(i + track_length - 1, n_rel) - i + 1) for i in range(n_rel - 1)]


def snippet_metrics_from_chunks(
    chunks,
    gt_extrinsics: np.ndarray,
    track_length: int = SNIPPET_TRACK_LENGTH,
) -> Dict[str, float]:
    """Snippet ATE for a sequence too long for one forward pass, WITHOUT stitching.

    Every window is re-anchored at its own origin and re-fitted for scale, so it never reads
    a global reference frame -- it only needs its own ``track_length`` frames to come from a
    single, self-consistent inference pass. Feeding it overlapping chunks and scoring each
    window inside whichever chunk fully contains it therefore reproduces the whole-sequence
    number exactly, with no seam error anywhere. (Contrast infer_sequence_stitched, whose
    per-seam Sim3 moved full-sequence ATE by up to +32% on an 80-frame probe --
    see diag/stitch_error.py.)

    ``chunks``: iterable of ``(start_index, extrinsics)``, each (n,3,4) or (n,4,4) world-to-cam
    for frames ``start_index .. start_index+n-1`` of the sequence. Consecutive chunks must
    overlap by at least ``track_length - 1`` frames or some window will have no home.
    """
    def to_c2w(E):
        E = np.asarray(E, dtype=np.float64)
        if E.shape[-2] == 3:
            E = np.concatenate([E, np.tile([[[0.0, 0.0, 0.0, 1.0]]], (len(E), 1, 1))], axis=1)
        return np.linalg.inv(E)

    gt_c2w = to_c2w(gt_extrinsics)
    chunk_c2w = [(int(st), to_c2w(E)) for st, E in chunks]
    spans = _snippet_spans(len(gt_c2w), int(track_length))

    ates, res, homeless = [], [], []
    for i, k in spans:
        # Prefer the chunk where the window sits furthest from either edge: VGGT's per-frame
        # quality degrades at a pass's boundaries, so a centred window is the better estimate.
        best, best_margin = None, -1
        for st, C in chunk_c2w:
            if st <= i and st + len(C) >= i + k:
                margin = min(i - st, st + len(C) - (i + k))
                if margin > best_margin:
                    best, best_margin = (st, C), margin
        if best is None:
            homeless.append(i)
            continue
        st, C = best
        ates.append(_snippet_ate(_local_xyz(gt_c2w, i, k), _local_xyz(C, i - st, k)))
        res.append(_snippet_rot_err_local(gt_c2w, i, C, i - st, k))

    if homeless:
        raise ValueError(f"{len(homeless)} windows (first at frame {homeless[0]}) are not "
                         f"fully inside any chunk; raise the overlap to >= {track_length - 1}")
    return dict(snippet_ate=float(np.mean(ates)), snippet_ate_std=float(np.std(ates)),
                snippet_rot=float(np.mean(res)), snippet_rot_std=float(np.std(res)),
                n_windows=len(spans), track_length=int(track_length))


def _snippet_rot_err_local(gt_c2w, gi, pred_c2w, pi, n):
    """compute_re over one window, with gt and pred indexed into their own arrays."""
    tot = 0.0
    g0, p0 = np.linalg.inv(gt_c2w[gi]), np.linalg.inv(pred_c2w[pi])
    for j in range(n):
        R_gt = (g0 @ gt_c2w[gi + j])[:3, :3]
        R_pr = (p0 @ pred_c2w[pi + j])[:3, :3]
        R = R_gt @ np.linalg.inv(R_pr)
        s = np.linalg.norm([R[0, 1] - R[1, 0], R[1, 2] - R[2, 1], R[0, 2] - R[2, 0]])
        tot += float(np.arctan2(s, np.trace(R) - 1))
    return tot / n
