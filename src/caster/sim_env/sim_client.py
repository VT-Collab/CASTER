"""A small client for the persistent Isaac simulation server."""

from pathlib import Path

import zmq


class SimClient:
    def __init__(self, address="tcp://localhost:15555", timeout=3600):
        if timeout <= 0:
            raise ValueError("Simulation timeout must be positive")
        self.address = address
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self.socket.connect(address)
        self.frame = 0

    def _request(self, command, **arguments):
        if self.socket.closed:
            raise RuntimeError("Simulation client is closed; create a new client")
        try:
            self.socket.send_json({"command": command, **arguments})
            response = self.socket.recv_json()
        except zmq.Again as error:
            self.close()
            raise TimeoutError(
                f"No reply to {command!r} from {self.address} within {self.timeout}s. "
                "Check the server output and address before retrying."
            ) from error
        if response.get("status") == "error":
            raise RuntimeError(f"Simulation {command}: {response.get('error')}")
        return response

    def health(self):
        return self._request("health")

    def reset(self, transforms_path, video_path, render=True):
        response = self._request(
            "reset", transforms_path=str(Path(transforms_path).resolve()),
            video_path=str(Path(video_path).resolve()), render=render,
        )
        self.frame = 0
        return response

    def execute_grasp(self, object_name, **options):
        return self._request("execute_grasp", object_name=object_name,
                             mode="m2t2", **options).get("info", {})

    def step(self, action, record=True, return_positions=False,
             return_robot_links=None, contact_query=None):
        response = self._request("step", action=action, frame_idx=self.frame,
                                 record=record, return_positions=return_positions,
                                 return_robot_links=return_robot_links or [],
                                 contact_query=contact_query)
        self.frame += 1
        return response.get("info", {})

    def execute_grasp_get_all(self, object_name, **options):
        return self.execute_grasp(object_name, **options)

    def restore_or_execute_grasp_get_all(self, object_name, **options):
        response = self._request("restore_post_grasp", object_name=object_name)
        if response.get("status") == "not_cached":
            return self.execute_grasp(object_name, **options)
        return response.get("info", {})

    def restore_or_execute_grasp(self, object_name, **options):
        return self.restore_or_execute_grasp_get_all(object_name, **options).get("object_poses", {})

    def disconnect(self):
        self.close()

    def execute_release(self):
        self._request("execute_release")

    def save_video(self):
        self._request("save_video")

    def close(self):
        self.socket.close(linger=0)
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
