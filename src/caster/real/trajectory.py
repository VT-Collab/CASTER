import time

import numpy as np
from termcolor import colored

from caster.real.franka import Franka


LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def quat2rot(q):
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)

def rot_error(R_target, R_curr):
    R_err = R_target @ R_curr.T
    v = 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ], dtype=np.float64)
    tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    sin_theta = np.sin(theta)
    if sin_theta > 1e-4:
        return v * (theta / sin_theta)
    return v

def interpolate_tcp(trajectory, sample):
    lower = min(int(sample), len(trajectory) - 1)
    upper = min(lower + 1, len(trajectory) - 1)
    fraction = sample - lower
    pos = trajectory[lower, :3] * (1.0 - fraction) + trajectory[upper, :3] * fraction
    q_low = trajectory[lower, 3:]
    q_up = trajectory[upper, 3:]
    if np.dot(q_low, q_up) < 0.0:
        q_up = -q_up
    q = q_low * (1.0 - fraction) + q_up * fraction
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q = q / norm
    return np.concatenate([pos, q])

def load_trajectory(path, require_feasible=True):
    with np.load(path, allow_pickle=False) as archive:
        if require_feasible and "feasible" in archive and not bool(archive["feasible"]):
            raise ValueError("Trajectory is marked infeasible")
        if "joint_trajectory" in archive:
            trajectory = np.asarray(archive["joint_trajectory"], dtype=np.float64)
            traj_type = "joint"
        elif "tcp_trajectory" in archive:
            trajectory = np.asarray(archive["tcp_trajectory"], dtype=np.float64)
            traj_type = "tcp"
        else:
            raise KeyError(f"{path} does not contain joint_trajectory or tcp_trajectory")

    if trajectory.ndim != 2 or trajectory.shape[1] != 7 or len(trajectory) < 2:
        raise ValueError(f"Expected trajectory shape (T, 7), got {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("Trajectory contains non-finite values")

    if traj_type == "joint":
        if np.any(trajectory < LOWER) or np.any(trajectory > UPPER):
            raise ValueError("Trajectory exceeds the stock Panda joint limits")
    else:
        quats = trajectory[:, 3:]
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        if np.any(norms < 1e-4):
            raise ValueError("TCP trajectory contains zero-norm quaternion")
        trajectory = trajectory.copy()
        trajectory[:, 3:] /= norms

    return trajectory, traj_type


def load_joint_trajectory(path, require_feasible=True):
    trajectory, traj_type = load_trajectory(path, require_feasible=require_feasible)
    if traj_type != "joint":
        raise ValueError(f"Expected joint trajectory in {path}, got {traj_type}")
    return trajectory


def interpolate(trajectory, sample):
    lower = min(int(sample), len(trajectory) - 1)
    upper = min(lower + 1, len(trajectory) - 1)
    fraction = sample - lower
    return trajectory[lower] * (1.0 - fraction) + trajectory[upper] * fraction


def playback_rate(trajectory, source_hz, control, traj_type="joint"):
    requested = float(control["playback_rate"])
    if requested <= 0.0:
        raise ValueError("control.playback_rate must be positive")
    if not control["automatic_slowdown"]:
        return requested
    if traj_type == "tcp":
        sampled_speed = np.max(
            np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1) * source_hz
        )
        cart_limit = float(control.get("cartesian_velocity_limit", 0.25))
        safe_speed = cart_limit * float(control["speed_fraction"])
    else:
        sampled_speed = np.max(
            np.linalg.norm(np.diff(trajectory, axis=0), axis=1) * source_hz
        )
        safe_speed = float(control["velocity_limit"]) * float(control["speed_fraction"])
    return min(requested, safe_speed / max(sampled_speed, 1e-12))


ROLLOUT_FILES = {
    "feature": "optimized_trajectory.npz",
    "trajectory": "optimized_trajectory.npz",
    "replay": "trajectory.npz",
}


def list_rollouts(output_dir, method):
    """Find exported rollouts in the dataset's simulation output directory."""
    return sorted(
        path for path in output_dir.glob("demo_*")
        if (path / ROLLOUT_FILES[method]).is_file()
    )


def resolve_rollout(output_dir, method, query):
    """Select a demo number or an exact demo directory name."""
    query = str(query).strip()
    matches = [
        path for path in list_rollouts(output_dir, method)
        if path.name == query
        or (query.isdigit() and path.name.startswith(f"demo_{int(query)}_"))
    ]
    if not matches:
        raise FileNotFoundError(f"No exported rollout for '{query}' in {output_dir}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple rollouts for demo {query}: {names}. Select a directory name.")
    demo_dir = matches[0]
    grasp_dir = demo_dir if method == "replay" else output_dir
    return demo_dir / ROLLOUT_FILES[method], grasp_dir / "grasp_trajectory.npz"


def clamp_trajectory_min_tcp_z(trajectory, min_z=0.02):
    """Clamp trajectory waypoints so TCP Z does not descend below min_z."""
    clamped = np.array(trajectory, dtype=np.float64, copy=True)
    z_values = np.array([Franka.joint2pose(q)[0][2] for q in clamped])
    below = np.where(z_values < min_z)[0]
    if len(below) == 0:
        return clamped
    first_below = below[0]
    if first_below > 0:
        z_prev = z_values[first_below - 1]
        z_curr = z_values[first_below]
        denom = z_curr - z_prev
        alpha = (min_z - z_prev) / denom if abs(denom) > 1e-9 else 0.0
        boundary_q = (1.0 - alpha) * clamped[first_below - 1] + alpha * clamped[first_below]
    else:
        boundary_q = clamped[0]
    clamped[first_below:] = boundary_q
    return clamped


def replay(
    robot,
    conn,
    trajectory,
    control,
    source_hz,
    rate,
    *,
    move_to_start,
    min_tcp_z=None,
    camera_feed=None,
):
    if move_to_start:
        move_speed = float(control.get("move_to_start_speed", 0.30))
        robot.go2position(
            conn,
            trajectory[0],
            callback=camera_feed.update if camera_feed else None,
            speed_limit=move_speed,
        )

    start_error = np.linalg.norm(trajectory[0] - robot.readState(conn)["q"])
    if start_error > float(control["start_tolerance"]):
        raise RuntimeError(
            f"Start error is {start_error:.3f} rad. Move to the first waypoint or "
            "deliberately enable control.move_to_start."
        )

    start = time.monotonic()
    duration = (len(trajectory) - 1) / (source_hz * rate)
    try:
        while True:
            if camera_feed is not None:
                camera_feed.update()
            elapsed = time.monotonic() - start
            target = interpolate(
                trajectory,
                min(elapsed * source_hz * rate, len(trajectory) - 1),
            )
            state = robot.readState(conn)
            current_z = state["x"][2]
            error = target - state["q"]
            if np.linalg.norm(error) > float(control["tracking_abort_error"]):
                raise RuntimeError(
                    f"Joint tracking error ({np.linalg.norm(error):.3f} rad) exceeded the configured abort limit ({control['tracking_abort_error']} rad)"
                )
            qdot = float(control["position_gain"]) * error
            if min_tcp_z is not None and current_z <= min_tcp_z:
                J_z = state["J"][2, :]
                norm_sq = float(np.dot(J_z, J_z))
                if norm_sq > 1e-6:
                    v_z = float(np.dot(J_z, qdot))
                    desired_vz = max(0.0, 2.0 * (min_tcp_z - current_z))
                    if v_z < desired_vz:
                        qdot = qdot + (J_z / norm_sq) * (desired_vz - v_z)
            robot.send2robot(
                conn,
                qdot,
                limit=float(control["velocity_limit"]),
            )
            if elapsed >= duration:
                break
    finally:
        robot.send2robot(conn, np.zeros(7))


def replay_tcp(
    robot,
    conn,
    trajectory,
    control,
    source_hz,
    rate,
    *,
    move_to_start,
    min_tcp_z=None,
    camera_feed=None,
):
    if min_tcp_z is not None:
        trajectory = trajectory.copy()
        trajectory[:, 2] = np.maximum(trajectory[:, 2], min_tcp_z)

    if move_to_start:
        target_pos = trajectory[0, :3]
        target_rot = quat2rot(trajectory[0, 3:])
        start_move = time.monotonic()
        while time.monotonic() - start_move < 5.0:
            if camera_feed is not None:
                camera_feed.update()
            state = robot.readState(conn)
            curr_pos, curr_rot = Franka.joint2pose(state["q"])
            pos_err = target_pos - curr_pos
            rot_err = rot_error(target_rot, curr_rot)
            if np.linalg.norm(pos_err) < 0.005 and np.linalg.norm(rot_err) < 0.02:
                break
            v_cmd = np.clip(1.5 * pos_err, -0.05, 0.05)
            w_cmd = np.clip(1.5 * rot_err, -0.15, 0.15)
            xdot = np.concatenate([v_cmd, w_cmd])
            qdot = np.linalg.pinv(state["J"], rcond=1e-2) @ xdot
            robot.send2robot(conn, qdot, limit=float(control["velocity_limit"]))
            time.sleep(0.01)
        robot.send2robot(conn, np.zeros(7))

    start = time.monotonic()
    duration = (len(trajectory) - 1) / (source_hz * rate)
    kp_pos = float(control.get("cartesian_pos_gain", 2.0))
    kp_rot = float(control.get("cartesian_rot_gain", 1.5))
    try:
        while True:
            if camera_feed is not None:
                camera_feed.update()
            elapsed = time.monotonic() - start
            target = interpolate_tcp(
                trajectory,
                min(elapsed * source_hz * rate, len(trajectory) - 1),
            )
            target_pos = target[:3]
            target_rot = quat2rot(target[3:])

            state = robot.readState(conn)
            curr_pos, curr_rot = Franka.joint2pose(state["q"])
            current_z = curr_pos[2]

            pos_err = target_pos - curr_pos
            rot_err = rot_error(target_rot, curr_rot)

            abort_limit = float(control.get("cartesian_tracking_abort_error", 1.0))
            if np.linalg.norm(pos_err) > abort_limit:
                raise RuntimeError(
                    f"Cartesian tracking error {np.linalg.norm(pos_err):.3f} m exceeded abort limit ({abort_limit} m)"
                )

            v_cmd = kp_pos * pos_err
            w_cmd = kp_rot * rot_err
            if min_tcp_z is not None and current_z <= min_tcp_z:
                v_cmd[2] = max(0.0, 2.0 * (min_tcp_z - current_z))

            xdot = np.concatenate([v_cmd, w_cmd])
            qdot = np.linalg.pinv(state["J"], rcond=1e-2) @ xdot
            robot.send2robot(
                conn,
                qdot,
                limit=float(control["velocity_limit"]),
            )
            if elapsed >= duration:
                break
    finally:
        robot.send2robot(conn, np.zeros(7))

