"""Validate or replay recovered trajectories on a physical Franka robot."""

from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from termcolor import colored

from caster.real.camera import CameraFeed, prompt_with_camera, sleep_with_camera
from caster.real.franka import Franka
from caster.real.trajectory import (
    clamp_trajectory_min_tcp_z,
    list_rollouts,
    load_joint_trajectory,
    load_trajectory,
    playback_rate,
    replay,
    replay_tcp,
    resolve_rollout,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def make_grasp_control(control, grasp):
    result = dict(control)
    multiplier = float(grasp.get("speed_multiplier", 2.0))
    for key in ("playback_rate", "velocity_limit", "tracking_abort_error"):
        result[key] = (
            float(grasp[key]) if key in grasp else float(control[key]) * multiplier
        )
    return result


def describe_trajectory(path, trajectory, kind, source_hz, rate):
    positions = trajectory if kind == "joint" else trajectory[:, :3]
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1).max() * source_hz
    duration = (len(trajectory) - 1) / source_hz
    label = "Joint" if kind == "joint" else "TCP"
    speed_label = "sampled joint-space speed" if kind == "joint" else "linear speed"
    unit = "rad/s" if kind == "joint" else "m/s"
    print(colored(f"\n--- Demo ({label}): {path.name} ---", "magenta"))
    print(f"Path: {path}")
    print(f"Shape: {trajectory.shape}; duration: {duration:.3f}s")
    print(f"Maximum {speed_label}: {speed:.3f} {unit}")
    print(f"Playback rate: {rate:.4f}x; execution duration: {duration / rate:.3f}s")


def set_gripper(robot, conn, grasp, action, camera_feed):
    default_command = "o" if action == "open" else "c"
    robot.send2gripper(conn, str(grasp.get(f"{action}_command", default_command)))
    seconds = (
        grasp.get("open_settle_seconds", 1.0) if action == "open"
        else grasp["close_settle_seconds"]
    )
    sleep_with_camera(float(seconds), camera_feed=camera_feed)


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(config: DictConfig):
    scene_dir = (REPO_ROOT / Path(config.scene_dir).expanduser()).resolve()
    task_dir = scene_dir / f"cf_{config.cf_index}_{config.task_name}"
    folders = {
        "feature": "feature_cost_function",
        "trajectory": "trajectory_cost_function",
        "replay": "replay",
    }
    output_dir = task_dir / "optimization" / folders[config.method]
    grasp_cfg = config.grasp
    task_name = f"{scene_dir.name}_{task_dir.name}_{config.method}"
    release_at_end = grasp_cfg.release_at_end

    camera_cfg = config["camera"]
    camera_feed = (
        CameraFeed(
            address=camera_cfg.get("address", "tcp://127.0.0.1:5555"),
            preview_port=camera_cfg.get("preview_port", 8082),
        )
        if camera_cfg.get("enabled", True) else None
    )

    should_record = camera_cfg.record
    record_fps = camera_cfg.record_fps
    record_post_seconds = camera_cfg.record_post_seconds
    record_dir = REPO_ROOT / Path(camera_cfg.record_dir)

    robot = None
    robot_conn = None
    gripper_conn = None

    try:
        print(f"\nRollout directory: {output_dir}")
        print(f"Video prefix:   {colored(task_name, 'magenta', attrs=['bold'])}")
        if release_at_end:
            print(colored("Gripper end policy: release/open at end of trajectory", "cyan"))
        else:
            print(colored("Gripper end policy: keep closed at end of trajectory (holding)", "cyan", attrs=["bold"]))
        print("Available demos:")
        for demo_dir in list_rollouts(output_dir, config.method):
            print(f"  {demo_dir.name}")

        if config.execute:
            network = config["network"]
            robot = Franka(bind_address=str(network["bind_address"]))
            robot_port = int(network["robot_port"])
            gripper_port = int(network["gripper_port"])
            print(
                f"\nWaiting for robot on {network['bind_address']}:{robot_port} "
                f"and gripper on {network['bind_address']}:{gripper_port}..."
            )
            robot_conn, gripper_conn = robot.connect_robot_and_gripper(
                robot_port,
                gripper_port,
                callback=camera_feed.update if camera_feed else None,
            )
            print("Connected to robot and gripper.")
        else:
            print(colored("\n[DRY RUN MODE] Set execute=true to command the robot.", "yellow"))

        pending_demo = str(config.demo) if config.demo is not None else None
        while True:
            if pending_demo is not None:
                user_input = pending_demo
                pending_demo = None
            else:
                prompt_text = colored(
                    "\nEnter demo number or directory name to replay (or 'q' to quit): ",
                    "cyan",
                    attrs=["bold"],
                )
                try:
                    user_input = prompt_with_camera(prompt_text, camera_feed=camera_feed)
                except KeyboardInterrupt:
                    print("\nReceived interrupt, exiting.")
                    break

            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting.")
                break
            if not user_input:
                continue

            try:
                traj_path, grasp_path = resolve_rollout(output_dir, config.method, user_input)
            except Exception as e:
                print(colored(f"Error: {e}", "red"))
                continue

            try:
                source_hz = float(config["trajectory"]["source_hz"])
                grasp_trajectory = load_joint_trajectory(
                    grasp_path,
                    require_feasible=False,
                )
                min_grasp_tcp_z = float(config.get("grasp", {}).get("minimum_tcp_z", 0.02))
                min_control_tcp_z = float(config.get("control", {}).get("minimum_tcp_z", min_grasp_tcp_z))
                grasp_trajectory = clamp_trajectory_min_tcp_z(
                    grasp_trajectory,
                    min_z=min_grasp_tcp_z,
                )
                with np.load(grasp_path, allow_pickle=False) as archive:
                    grasp_source_hz = float(archive.get("source_hz", source_hz))
                grasp_speeds = (
                    np.linalg.norm(np.diff(grasp_trajectory, axis=0), axis=1)
                    * grasp_source_hz
                )
                grasp_control = make_grasp_control(config["control"], grasp_cfg)

                grasp_rate = playback_rate(
                    grasp_trajectory,
                    grasp_source_hz,
                    grasp_control,
                )
                print(f"Grasp trajectory: {grasp_path}")
                print(
                    f"Grasp shape: {grasp_trajectory.shape}; duration: "
                    f"{(len(grasp_trajectory)-1)/grasp_source_hz:.3f}s"
                )
                print(
                    "Maximum grasp joint-space speed: "
                    f"{grasp_speeds.max():.3f} rad/s; playback rate: "
                    f"{grasp_rate:.4f}x; execution duration: "
                    f"{(len(grasp_trajectory)-1)/(grasp_source_hz*grasp_rate):.3f}s"
                )
                print(
                    f"Grasp tracking abort limit: {grasp_control['tracking_abort_error']:.2f} rad "
                    f"(rollout abort limit: {config['control']['tracking_abort_error']} rad)"
                )

                trajectory, traj_type = load_trajectory(
                    traj_path,
                    require_feasible=config["trajectory"]["require_feasible"],
                )
            except Exception as e:
                print(colored(f"Failed to load trajectory {traj_path.name}: {e}", "red"))
                continue

            with np.load(traj_path, allow_pickle=False) as archive:
                demo_source_hz = float(archive.get("source_hz", source_hz))

            rate = playback_rate(trajectory, demo_source_hz, config["control"], traj_type)
            describe_trajectory(traj_path, trajectory, traj_type, demo_source_hz, rate)

            if not config.execute:
                print(colored(f"[DRY RUN] Validated {traj_path.name}.", "yellow"))
                continue

            # Flush any stale state packets accumulated in socket buffer while idle
            try:
                robot_conn.setblocking(False)
                while robot_conn.recv(65536):
                    pass
            except Exception:
                pass
            finally:
                robot_conn.setblocking(True)

            try:
                print("Opening gripper at start of execution...", flush=True)
                set_gripper(robot, gripper_conn, config["grasp"], "open", camera_feed)

                move_to_start_speed = float(config["control"].get("move_to_start_speed", 0.30))
                if config["control"]["move_to_start"]:
                    print(f"Moving to start position (homing at {move_to_start_speed:.2f} rad/s)...", flush=True)
                    robot.go2position(
                        robot_conn,
                        grasp_trajectory[0],
                        callback=camera_feed.update if camera_feed else None,
                        speed_limit=move_to_start_speed,
                    )

                confirm = prompt_with_camera(
                    colored("\nRobot at start position. Press [Enter] to start execution (or 'q' to cancel): ", "yellow", attrs=["bold"]),
                    camera_feed=camera_feed,
                )
                if confirm.lower() in ("q", "quit", "c", "cancel"):
                    print(colored("Execution cancelled by user.", "yellow"))
                    continue

                print(colored("grasping", "green"), flush=True)
                if should_record and camera_feed is not None and camera_feed.available:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = f"{task_name}_{traj_path.parent.name}_{timestamp}.mp4"
                    record_path = record_dir / video_filename
                    camera_feed.start_recording(record_path, fps=record_fps)

                replay(
                    robot,
                    robot_conn,
                    grasp_trajectory,
                    grasp_control,
                    grasp_source_hz,
                    grasp_rate,
                    move_to_start=False,
                    min_tcp_z=min_grasp_tcp_z,
                    camera_feed=camera_feed,
                )
                set_gripper(robot, gripper_conn, config["grasp"], "close", camera_feed)
                print(colored(f"start trajectory executing ({traj_type.upper()})", "green"), flush=True)
                if traj_type == "joint":
                    if config["control"]["move_to_start"]:
                        robot.go2position(
                            robot_conn,
                            trajectory[0],
                            callback=camera_feed.update if camera_feed else None,
                            speed_limit=move_to_start_speed,
                        )
                    replay(
                        robot,
                        robot_conn,
                        trajectory,
                        config["control"],
                        demo_source_hz,
                        rate,
                        move_to_start=False,
                        camera_feed=camera_feed,
                    )
                else:
                    replay_tcp(
                        robot,
                        robot_conn,
                        trajectory,
                        config["control"],
                        demo_source_hz,
                        rate,
                        move_to_start=config["control"]["move_to_start"],
                        min_tcp_z=min_control_tcp_z,
                        camera_feed=camera_feed,
                    )
                if release_at_end:
                    print(colored("releasing", "green"), flush=True)
                    set_gripper(robot, gripper_conn, config["grasp"], "open", camera_feed)
                else:
                    print(colored("holding (gripper stays closed)", "cyan"), flush=True)
                if camera_feed is not None and camera_feed.recording:
                    if record_post_seconds > 0:
                        print(
                            f"Holding recording for {record_post_seconds:.1f}s after trajectory completes...",
                            flush=True,
                        )
                        sleep_with_camera(record_post_seconds, camera_feed=camera_feed)
                    camera_feed.stop_recording()
                print(colored(f"Demo {traj_path.name} execution complete; zero velocity commanded.", "green"), flush=True)

                if not release_at_end:
                    prompt_with_camera(
                        colored("\n[GRIPPER] Video saved. Press [Enter] to open gripper: ", "yellow", attrs=["bold"]),
                        camera_feed=camera_feed,
                    )
                    print(colored("Opening gripper...", "green"), flush=True)
                    set_gripper(robot, gripper_conn, config["grasp"], "open", camera_feed)
                    print(colored("Gripper opened.", "green"), flush=True)
            except Exception as e:
                print(colored(f"Error during execution: {e}", "red"), flush=True)
                try:
                    robot.send2robot(robot_conn, np.zeros(7))
                except Exception:
                    pass
                if camera_feed is not None and camera_feed.recording:
                    camera_feed.stop_recording()
    finally:
        if robot is not None and robot_conn is not None:
            try:
                robot.send2robot(robot_conn, np.zeros(7))
            except Exception:
                pass
        for conn in (robot_conn, gripper_conn):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        if camera_feed is not None:
            camera_feed.close()


if __name__ == "__main__":
    main()
