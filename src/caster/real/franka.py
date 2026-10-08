import select
import socket
import time

import numpy as np


class Franka:
    def __init__(self, bind_address="172.16.0.3"):
        self.home = np.array(
            [-0.232867, -0.524729, 0.137301, -2.3478, 0.0455863, 1.85082, 0.64003]
        )
        self.bind_address = bind_address

    def connect(self, port, callback=None):
        server = self._listen(port)
        try:
            while True:
                if callback is not None:
                    callback()
                readable, _, _ = select.select([server], [], [], 0.03)
                if readable:
                    conn, _ = server.accept()
                    return conn
        finally:
            server.close()

    def connect_robot_and_gripper(self, robot_port, gripper_port, callback=None):
        """Listen on both ports before accepting either controller."""
        robot_server = self._listen(robot_port)
        gripper_server = self._listen(gripper_port)
        robot_conn = None
        gripper_conn = None
        try:
            while robot_conn is None or gripper_conn is None:
                if callback is not None:
                    callback()
                listening = []
                if robot_conn is None:
                    listening.append(robot_server)
                if gripper_conn is None:
                    listening.append(gripper_server)
                readable, _, _ = select.select(listening, [], [], 0.03)
                for s in readable:
                    if s is robot_server and robot_conn is None:
                        robot_conn, _ = robot_server.accept()
                    elif s is gripper_server and gripper_conn is None:
                        gripper_conn, _ = gripper_server.accept()
            return robot_conn, gripper_conn
        finally:
            robot_server.close()
            gripper_server.close()

    def _listen(self, port):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.bind_address, port))
        server.listen(1)
        return server

    @staticmethod
    def send2gripper(conn, command):
        conn.sendall(f"s,{command},".encode())

    @staticmethod
    def send2robot(conn, qdot, limit=1.0):
        qdot = np.asarray(qdot, dtype=np.float64).copy()
        if qdot.shape != (7,) or not np.all(np.isfinite(qdot)):
            raise ValueError("qdot must contain seven finite values")
        speed = np.linalg.norm(qdot)
        if speed > limit:
            qdot *= limit / speed
        payload = ",".join(f"{value:.5f}" for value in qdot)
        conn.sendall(f"s,{payload},".encode())

    def listen2robot(self, conn):
        state_length = 55
        fields = conn.recv(20480).decode(errors="ignore").split(",")
        try:
            start = fields.index("s") + 1
            values = np.asarray(
                [float(value) for value in fields[start : start + state_length]]
            )
        except (ValueError, IndexError):
            return None
        if len(values) != state_length:
            return None

        position, rotation = self.joint2pose(values[:7])
        beta = -np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0))
        angle = np.array([
            np.arctan2(rotation[2, 1], rotation[2, 2]),
            beta,
            np.arctan2(rotation[1, 0], rotation[0, 0]),
        ])
        return {
            "q": values[:7],
            "O_F": values[7:13],
            "J": values[13:].reshape((7, 6)).T,
            "x": np.concatenate((position, angle)),
            "angle": angle,
        }

    def readState(self, conn):
        while True:
            state = self.listen2robot(conn)
            if state is not None:
                return state

    @staticmethod
    def xdot2qdot(xdot, states):
        return np.linalg.pinv(states["J"]) @ np.asarray(xdot)

    @staticmethod
    def joint2pose(q):
        def rot_x(theta):
            return np.array([[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0],
                             [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])

        def rot_z(theta):
            return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                             [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        def trans_x(theta, x, y, z):
            transform = rot_x(theta)
            transform[:3, 3] = [x, y, z]
            return transform

        def trans_z(theta, x, y, z):
            transform = rot_z(theta)
            transform[:3, 3] = [x, y, z]
            return transform

        q = np.asarray(q, dtype=np.float64)
        transform = np.linalg.multi_dot([
            trans_z(q[0], 0, 0, 0.333),
            rot_x(-np.pi / 2) @ rot_z(q[1]),
            trans_x(np.pi / 2, 0, -0.316, 0) @ rot_z(q[2]),
            trans_x(np.pi / 2, 0.0825, 0, 0) @ rot_z(q[3]),
            trans_x(-np.pi / 2, -0.0825, 0.384, 0) @ rot_z(q[4]),
            rot_x(np.pi / 2) @ rot_z(q[5]),
            trans_x(np.pi / 2, 0.088, 0, 0) @ rot_z(q[6]),
            trans_z(-np.pi / 4, 0, 0, 0.107 + 0.20),
        ])
        return transform[:3, 3], transform[:3, :3]

    def go2position(self, conn, goal=None, callback=None, speed_limit=0.3):
        goal = self.home if goal is None else np.asarray(goal, dtype=np.float64)
        start = time.monotonic()
        state = self.readState(conn)
        if callback is not None:
            callback()
        while np.linalg.norm(state["q"] - goal) > 0.05:
            if time.monotonic() - start > 20.0:
                raise TimeoutError("Timed out moving to the requested joint pose")
            self.send2robot(conn, np.clip(goal - state["q"], -speed_limit, speed_limit))
            state = self.readState(conn)
            if callback is not None:
                callback()
        self.send2robot(conn, np.zeros(7))
