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
