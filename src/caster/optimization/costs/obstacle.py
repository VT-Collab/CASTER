import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..grasp import resolve_transforms

logger = logging.getLogger(__name__)

def wxyz_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert one or more ``[w, x, y, z]`` quaternions to rotation matrices."""
    quaternions = np.asarray(quaternions)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 1e-6, norms, 1.0)
    q = quaternions / safe_norms
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    rotation = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    rotation[..., 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
    rotation[..., 0, 1] = 2.0 * (x * y - w * z)
    rotation[..., 0, 2] = 2.0 * (x * z + w * y)

    rotation[..., 1, 0] = 2.0 * (x * y + w * z)
    rotation[..., 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
    rotation[..., 1, 2] = 2.0 * (y * z - w * x)

    rotation[..., 2, 0] = 2.0 * (x * z - w * y)
    rotation[..., 2, 1] = 2.0 * (y * z + w * x)
    rotation[..., 2, 2] = 1.0 - 2.0 * (x**2 + y**2)
    return rotation

class ObstacleCost:
    """Evaluate robot-proxy clearance against cached obstacle surface fields."""

    def __init__(
        self,
        demo_path: str | Path | None,
        tcp_radius: float,
        safety_margin: float,
        temperature: float,
        surface_samples: int = 10_000,
        sampling_buffer: float = 0.003,
        query_workers: int = 8,
        distance_backend: str = "grid",
        grid_resolution: float = 0.003,
        grid_padding: float | None = None,
        grid_temperature_tail: float = 8.0,
        max_grid_cells: int = 30_000_000,
    ):
        self.demo_path = Path(demo_path) if demo_path else None
        self.safety_margin = safety_margin
        self.temperature = temperature
        self.surface_samples = surface_samples
        self.sampling_buffer = sampling_buffer
        if distance_backend not in {"grid", "kdtree"}:
            raise ValueError(
                "distance_backend must be either 'grid' or 'kdtree'."
            )
        if grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be positive.")
        if grid_padding is not None and grid_padding <= 0.0:
            raise ValueError("grid_padding must be positive when provided.")
        if grid_temperature_tail < 0.0:
            raise ValueError("grid_temperature_tail cannot be negative.")
        if max_grid_cells < 1:
            raise ValueError("max_grid_cells must be positive.")

        self.distance_backend = distance_backend
        self.grid_resolution = float(grid_resolution)
        self.grid_padding = (
            float(grid_padding)
            if grid_padding is not None
            else None
        )
        self.grid_temperature_tail = float(grid_temperature_tail)
        self.max_grid_cells = int(max_grid_cells)
        if query_workers == -1:
            self.query_workers = -1
        elif query_workers >= 1:
            self.query_workers = min(
                query_workers,
                os.cpu_count() or 1,
            )
        else:
            raise ValueError(
                "query_workers must be -1 or a positive integer."
            )
        # Four sphere proxies: palm, distal fingers, and lower forearm.  The
        # central offsets are expressed in the active UMI TCP frame.  Relative
        # to the stock TCP, the UMI TCP is 0.0916 m farther from the hand, so
        # the stock palm/forearm centers (-0.08/-0.18 m) shift by -0.0916 m.
        # The former TCP-centered sphere was redundant with the overlapping
        # palm/finger proxies and introduced a large imaginary volume ahead of
        # the hand.
        self.local_proxies = np.array(
            [
                [0.0, 0.0, -0.1716],
                [0.0, 0.04, -0.04],
                [0.0, -0.04, -0.04],
                [0.0, 0.0, -0.2716],
            ],
            dtype=np.float64,
        )
        self.proxy_radii = np.array(
            [
                tcp_radius * 0.9,
                tcp_radius * 0.5,
                tcp_radius * 0.5,
                tcp_radius * 0.8,
            ],
            dtype=np.float64,
        )

        self._post_grasp_poses: dict[str, Mapping[str, Any]] = {}
        self._cached_target_object: str | None = None
        self._cached_meshes: list[dict[str, Any]] | None = None
        self._cached_distance_fields: list[dict[str, Any]] | None = None

    @property
    def has_post_grasp_poses(self) -> bool:
        return bool(self._post_grasp_poses)

    def set_demo_path(self, demo_path: str | Path | None) -> None:
        new_path = Path(demo_path) if demo_path else None
        if new_path != self.demo_path:
            self.demo_path = new_path
            self.clear_all_caches()

    def set_post_grasp_poses(
        self,
        poses: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self._post_grasp_poses = dict(poses)
        self.clear_distance_field_cache()

    def clear_distance_field_cache(self) -> None:
        """Invalidate fields whose world transforms may have changed."""
        self._cached_distance_fields = None

    def clear_all_caches(self) -> None:
        self._cached_target_object = None
        self._cached_meshes = None
        self._cached_distance_fields = None

    def evaluate_batch(
        self,
        trajectories: np.ndarray,
        target_object: str,
        extra_sphere_positions: np.ndarray | None = None,
        extra_sphere_radii: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate TCP proxies and optional simulated-link spheres in one query."""
        fields = self._get_distance_fields(target_object)
        batch_size, num_steps, _ = trajectories.shape
        if not fields:
            return np.zeros(batch_size)

        positions = trajectories[:, :, :3]
        rotations = wxyz_to_rotation_matrix(trajectories[:, :, 3:7])
        proxy_offsets = np.matmul(
            rotations,
            self.local_proxies.T,
        ).transpose(0, 1, 3, 2)
        proxy_positions = positions[:, :, None, :] + proxy_offsets
        sphere_positions = proxy_positions
        sphere_radii = self.proxy_radii

        if extra_sphere_positions is not None:
            extra_sphere_positions = np.asarray(
                extra_sphere_positions,
                dtype=np.float64,
            )
            extra_sphere_radii = np.asarray(
                extra_sphere_radii,
                dtype=np.float64,
            )
            if (
                extra_sphere_positions.ndim != 4
                or extra_sphere_positions.shape[:2]
                != (batch_size, num_steps)
                or extra_sphere_positions.shape[-1] != 3
            ):
                raise ValueError(
                    "extra_sphere_positions must have shape (B, N, S, 3)."
                )
            if extra_sphere_radii.shape != (
                extra_sphere_positions.shape[2],
            ):
                raise ValueError(
                    "extra_sphere_radii must contain one radius per sphere."
                )
            if np.any(extra_sphere_radii <= 0.0):
                raise ValueError("All extra sphere radii must be positive.")
            sphere_positions = np.concatenate(
                (sphere_positions, extra_sphere_positions),
                axis=2,
            )
            sphere_radii = np.concatenate(
                (sphere_radii, extra_sphere_radii),
            )

        num_spheres = sphere_positions.shape[2]
        all_points = sphere_positions.reshape(-1, 3)

        total_costs = np.zeros(batch_size)
        for field in fields:
            approximate_distances = self._query_distances(
                field,
                all_points,
            )
            surface_distances = np.maximum(
                approximate_distances - self.sampling_buffer,
                0.0,
            ).reshape(batch_size, num_steps, num_spheres)
            clearances = (
                surface_distances
                - sphere_radii[None, None, :]
            )
            per_step_costs = self._per_step_cost(clearances)
            obstacle_costs = self._aggregate_batch_over_time(per_step_costs)

            total_costs += obstacle_costs

        return total_costs

    def _per_step_cost(self, clearances: np.ndarray) -> np.ndarray:
        violations = self.temperature * np.logaddexp(
            0.0,
            (self.safety_margin - clearances) / self.temperature,
        )
        return np.max(
            (violations / self.safety_margin) ** 2,
            axis=-1,
        )

    @staticmethod
    def _aggregate_batch_over_time(
        per_step_costs: np.ndarray,
    ) -> np.ndarray:
        num_steps = per_step_costs.shape[1]
        worst_count = max(1, int(np.ceil(0.10 * num_steps)))
        worst_costs = np.partition(
            per_step_costs,
            num_steps - worst_count,
            axis=1,
        )[:, -worst_count:]
        return (
            0.10 * np.mean(per_step_costs, axis=1)
            + 0.90 * np.mean(worst_costs, axis=1)
        )

    def _load_meshes(self, target_object: str) -> list[dict[str, Any]]:
        if (
            self._cached_meshes is not None
            and self._cached_target_object == target_object
        ):
            return self._cached_meshes

        transforms_path = resolve_transforms(self.demo_path)

        import trimesh

        with transforms_path.open("r") as stream:
            transforms_data = json.load(stream)

        meshes = []
        for entry in transforms_data["objects"]:
            object_name = entry["name"]
            if object_name == target_object:
                continue

            mesh_path = transforms_path.parent / f"{object_name}.obj"
            if not mesh_path.exists():
                mesh_path = (
                    transforms_path.parent
                    / "meshes"
                    / f"{object_name}.obj"
                )

            mesh = trimesh.load(str(mesh_path))
            raw_extents = mesh.extents
            scale = entry.get("scale", 1.0)
            mesh.apply_scale(scale)
            # Match sim_env.sim_utils.convert_mesh(). MeshConverterCfg
            # stores quaternions as XYZW, so its configured quaternion
            # applies a +90-degree rotation about local X.
            mesh.apply_transform(
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )
            )
            print(
                f"[Cost Setup] Loaded local obstacle '{object_name}': "
                f"raw extents={raw_extents.tolist()} (m) | "
                f"scale={scale} | "
                f"scaled extents={mesh.extents.tolist()} (m)",
                flush=True,
            )
            meshes.append(
                {"name": object_name, "local_mesh": mesh}
            )

        self._cached_meshes = meshes
        self._cached_target_object = target_object
        return meshes

    def _mesh_in_world(self, obstacle: Mapping[str, Any]):
        from scipy.spatial.transform import Rotation

        pose = self._post_grasp_poses[obstacle["name"]]
        position = pose["position"]
        quaternion = pose["orientation"]
        if isinstance(position[0], list):
            position = position[0]
        if isinstance(quaternion[0], list):
            quaternion = quaternion[0]

        w, x, y, z = quaternion
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        transform[:3, 3] = position

        mesh = obstacle["local_mesh"].copy()
        mesh.apply_transform(transform)
        return mesh

    def _get_distance_fields(self, target_object: str) -> list[dict[str, Any]]:
        if self._cached_distance_fields is not None:
            return self._cached_distance_fields

        import trimesh
        from scipy.spatial import cKDTree

        fields = []
        random_state = np.random.get_state()
        np.random.seed(0)
        try:
            for obstacle in self._load_meshes(target_object):
                world_mesh = self._mesh_in_world(obstacle)
                if self.distance_backend == "grid":
                    try:
                        field = self._build_grid_field(
                            obstacle["name"],
                            world_mesh,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to build distance grid for %s; "
                            "falling back to cKDTree: %s",
                            obstacle["name"],
                            exc,
                        )
                        field = self._build_kdtree_field(
                            obstacle["name"],
                            world_mesh,
                            trimesh,
                            cKDTree,
                        )
                else:
                    field = self._build_kdtree_field(
                        obstacle["name"],
                        world_mesh,
                        trimesh,
                        cKDTree,
                    )
                fields.append(field)
        finally:
            np.random.set_state(random_state)

        self._cached_distance_fields = fields
        return fields

    def _build_kdtree_field(
        self,
        name: str,
        world_mesh,
        trimesh_module,
        tree_class,
    ) -> dict[str, Any]:
        sampled_points, _ = trimesh_module.sample.sample_surface(
            world_mesh,
            self.surface_samples,
        )
        surface_points = np.vstack(
            [world_mesh.vertices, sampled_points]
        )
        print(
            f"[Obstacle Setup] {name}: backend=kdtree, "
            f"vertices={len(world_mesh.vertices)}, "
            f"samples={len(sampled_points)}, "
            f"tree_points={len(surface_points)}",
            flush=True,
        )
        return {
            "name": name,
            "backend": "kdtree",
            "tree": tree_class(surface_points),
            "world_mesh": world_mesh,
        }

    def _build_grid_field(
        self,
        name: str,
        world_mesh,
    ) -> dict[str, Any]:
        """Build a cached conservative unsigned-distance grid."""
        from scipy.ndimage import distance_transform_edt

        resolution = self.grid_resolution
        max_proxy_radius = float(np.max(self.proxy_radii))
        padding = self.grid_padding
        if padding is None:
            padding = (
                max_proxy_radius
                + self.safety_margin
                + self.sampling_buffer
                + self.grid_temperature_tail * self.temperature
                + 2.0 * resolution
            )

        mesh_bounds = np.asarray(world_mesh.bounds, dtype=np.float64)
        grid_min = (
            np.floor((mesh_bounds[0] - padding) / resolution)
            * resolution
        )
        grid_max = (
            np.ceil((mesh_bounds[1] + padding) / resolution)
            * resolution
        )
        shape = (
            np.ceil((grid_max - grid_min) / resolution).astype(np.int64)
            + 1
        )
        cell_count = int(np.prod(shape, dtype=np.int64))
        if cell_count > self.max_grid_cells:
            raise MemoryError(
                f"grid requires {cell_count:,} cells, exceeding "
                f"max_grid_cells={self.max_grid_cells:,}"
            )

        # trimesh voxelization marks cells intersecting the triangle surface.
        surface_voxels = world_mesh.voxelized(pitch=resolution)
        surface_points = np.asarray(
            surface_voxels.points,
            dtype=np.float64,
        )
        if len(surface_points) == 0:
            raise ValueError("mesh voxelization produced no surface cells")

        surface_indices = np.rint(
            (surface_points - grid_min) / resolution
        ).astype(np.int64)
        surface_indices = np.clip(
            surface_indices,
            0,
            shape - 1,
        )

        occupied = np.zeros(tuple(shape), dtype=bool)
        occupied[
            surface_indices[:, 0],
            surface_indices[:, 1],
            surface_indices[:, 2],
        ] = True

        distances = distance_transform_edt(
            ~occupied,
            sampling=resolution,
        ).astype(np.float32)

        print(
            f"[Obstacle Setup] {name}: backend=grid, "
            f"resolution={resolution:.4f}m, "
            f"surface_voxels={len(surface_points):,}, "
            f"shape={tuple(int(v) for v in shape)}, "
            f"cells={cell_count:,}, "
            f"memory={distances.nbytes / (1024 ** 2):.1f}MiB",
            flush=True,
        )
        return {
            "name": name,
            "backend": "grid",
            "grid": distances,
            "origin": grid_min,
            "resolution": resolution,
            "mesh_bounds": mesh_bounds,
            "world_mesh": world_mesh,
        }

    def _query_distances(
        self,
        field: Mapping[str, Any],
        points: np.ndarray,
    ) -> np.ndarray:
        if field["backend"] == "grid":
            return self._query_grid(field, points)

        distances, _ = field["tree"].query(
            points,
            k=1,
            workers=self.query_workers,
        )
        return distances

    @staticmethod
    def _distance_to_bounds(
        points: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """Return the exact Euclidean distance to an axis-aligned box."""
        below = np.maximum(bounds[0] - points, 0.0)
        above = np.maximum(points - bounds[1], 0.0)
        return np.linalg.norm(below + above, axis=1)

    def _query_grid(
        self,
        field: Mapping[str, Any],
        points: np.ndarray,
    ) -> np.ndarray:
        """Vectorized trilinear lookup with a conservative error allowance."""
        points = np.asarray(points, dtype=np.float64)
        grid = field["grid"]
        resolution = float(field["resolution"])
        coordinates = (points - field["origin"]) / resolution
        lower = np.floor(coordinates).astype(np.int64)
        fractions = coordinates - lower

        shape = np.asarray(grid.shape, dtype=np.int64)
        inside = np.all(
            (lower >= 0) & (lower < shape - 1),
            axis=1,
        )

        # Outside the padded grid, distance to the mesh AABB is a conservative
        # lower bound on distance to the actual mesh surface.
        distances = self._distance_to_bounds(
            points,
            field["mesh_bounds"],
        )

        if np.any(inside):
            indices = lower[inside]
            weights = fractions[inside]
            x, y, z = indices.T
            wx, wy, wz = weights.T

            c000 = grid[x, y, z]
            c100 = grid[x + 1, y, z]
            c010 = grid[x, y + 1, z]
            c110 = grid[x + 1, y + 1, z]
            c001 = grid[x, y, z + 1]
            c101 = grid[x + 1, y, z + 1]
            c011 = grid[x, y + 1, z + 1]
            c111 = grid[x + 1, y + 1, z + 1]

            c00 = c000 * (1.0 - wx) + c100 * wx
            c10 = c010 * (1.0 - wx) + c110 * wx
            c01 = c001 * (1.0 - wx) + c101 * wx
            c11 = c011 * (1.0 - wx) + c111 * wx
            c0 = c00 * (1.0 - wy) + c10 * wy
            c1 = c01 * (1.0 - wy) + c11 * wy
            interpolated = c0 * (1.0 - wz) + c1 * wz

            # Surface voxelization and interpolation each introduce at most a
            # cell-scale error. Subtracting one cell diagonal keeps distances
            # conservative: clearance is never deliberately overestimated.
            allowance = np.sqrt(3.0) * resolution
            distances[inside] = np.maximum(
                interpolated - allowance,
                0.0,
            )

        return distances
