#!/usr/bin/env python3
"""Offline USD-to-browser conversion. Run with uv --with usd-core --with numpy.

Uses recorded link-origin poses, not USD's possibly stale dynamic transforms.
Only visual geometry is exported, including USD instance proxies.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, UsdShade


def read_material(prim):
    result = {'color': [0.72, 0.74, 0.76], 'roughness': 0.72, 'metalness': 0.08}
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return result
    for child in material.GetPrim().GetChildren():
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        for target, keys in {
            'color': ('diffuseColor', 'diffuse_color_constant'),
            'roughness': ('roughness', 'reflection_roughness_constant'),
            'metalness': ('metallic', 'metallic_constant'),
            'emissive': ('emissiveColor', 'emissive_color'),
        }.items():
            for key in keys:
                value = shader.GetInput(key).Get()
                if value is not None:
                    result[target] = list(value) if target in ('color', 'emissive') else float(value)
                    break
    return result


def read_vertex_colors(mesh, vertex_ids, face_counts, vertex_count):
    """Collapse USD displayColor interpolation to one RGB value per vertex."""
    primvar = mesh.GetDisplayColorPrimvar()
    values = primvar.ComputeFlattened()
    if values is None or not len(values):
        return None
    values = np.asarray(values, dtype=float)
    interpolation = primvar.GetInterpolation()
    if interpolation in ('vertex', 'varying') and len(values) == vertex_count:
        colors = values
    elif interpolation == 'faceVarying' and len(values) == len(vertex_ids):
        weights = np.bincount(vertex_ids, minlength=vertex_count)
        colors = np.column_stack([
            np.bincount(vertex_ids, weights=values[:, axis], minlength=vertex_count)
            for axis in range(3)
        ]) / np.maximum(weights[:, None], 1)
    elif interpolation == 'uniform' and len(values) == len(face_counts):
        corner_colors = np.repeat(values, face_counts, axis=0)
        weights = np.bincount(vertex_ids, minlength=vertex_count)
        colors = np.column_stack([
            np.bincount(vertex_ids, weights=corner_colors[:, axis], minlength=vertex_count)
            for axis in range(3)
        ]) / np.maximum(weights[:, None], 1)
    elif interpolation == 'constant':
        colors = np.broadcast_to(values[0], (vertex_count, 3)).copy()
    else:
        return None
    # A single uniform value is smaller and more accurately represented by the
    # existing material color. Export only genuine surface-color variation.
    if np.ptp(colors, axis=0).max() < 1 / 255:
        return None
    return np.clip(colors, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('browser_rollout_examples'))
    parser.add_argument('--output', type=Path, default=Path('static/resources/simulation'))
    parser.add_argument('--site-manifest', type=Path,
                        default=Path('static/resources/pointcloud/manifest.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_entries = json.loads((args.source / 'manifest.json').read_text())['rollouts']
    required = ('metadata', 'poses', 'geometry')
    missing = [(entry, [key for key in required if not (args.source / entry[key]).is_file()])
               for entry in source_entries]
    if any(keys for _, keys in missing):
        details = '\n'.join(
            f"  {entry.get('scene')}/{entry.get('demo')}: " +
            ', '.join(f"missing {key}={entry[key]}" for key in keys)
            for entry, keys in missing if keys)
        raise FileNotFoundError(
            'Rollout manifest contains stale or incomplete entries:\n' + details +
            '\nUpdate browser_rollout_examples/manifest.json before rebuilding.')

    site_ids = {}
    if args.site_manifest.is_file():
        for example in json.loads(args.site_manifest.read_text())['examples']:
            scene_number = int(example['sceneNumber'])
            if scene_number in site_ids:
                raise ValueError(f'Duplicate scene {scene_number} in {args.site_manifest}')
            site_ids[scene_number] = example['id']

    source_scenes = [int(entry['scene'].split('_')[1]) for entry in source_entries]
    duplicates = sorted({scene for scene in source_scenes if source_scenes.count(scene) > 1})
    if duplicates:
        raise ValueError(f'Multiple valid simulation rollouts for scene(s): {duplicates}')
    index = {}
    shared_payload = bytearray()
    shared_meshes = []
    export_shared_robot = True
    for entry in source_entries:
        meta = json.loads((args.source / entry['metadata']).read_text())
        with np.load(args.source / entry['poses'], allow_pickle=False) as data:
            names = data['body_names'].tolist()
            positions = data['positions'].copy()
            rotations = data['quaternions_wxyz'].copy()
            times = data['timestamps'].tolist()
            phases = data['phase_ids'].tolist()
            task_frames = data['trajectory_frame_indices'].tolist()
        assert positions.shape == (len(times), len(names), 3)
        assert rotations.shape == (len(times), len(names), 4)
        assert np.isfinite(positions).all() and np.isfinite(rotations).all()
        # Ignore exporter diagnostics incorrectly nested under bodies.
        mapping = {mesh['prim']: (i, mesh) for i, name in enumerate(names)
                   for mesh in meta['bodies'][name]['meshes']}
        stage = Usd.Stage.Open(str(args.source / entry['geometry']))
        cache = UsdGeom.XformCache()
        payload = bytearray()

        def pack(array, dtype, target=payload):
            while len(target) % 4:
                target.append(0)
            a = np.asarray(array, dtype=dtype).reshape(-1)
            result = {'offset': len(target), 'count': len(a)}
            target.extend(a.tobytes())
            return result

        meshes = []
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            imageable = UsdGeom.Imageable(prim)
            if not imageable or imageable.ComputeVisibility() == 'invisible' or imageable.ComputePurpose() not in ('default', 'render'):
                continue
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
                continue
            path = str(prim.GetPath())
            if '/GroundPlane/' in path:
                continue  # The browser supplies a simple floor without MDL shaders.
            body, descriptor = mapping.get(path, (-1, None))
            body_name = names[body] if body >= 0 else None
            is_robot = bool(body_name and body_name.startswith('robot/'))
            if is_robot and not export_shared_robot:
                continue
            transform = np.array(descriptor['mesh_to_body'] if descriptor else
                                 np.array(cache.GetLocalToWorldTransform(prim)).T, dtype=float)
            materials = [read_material(prim)]
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
                ids = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                vertex_colors = read_vertex_colors(mesh, ids, counts, len(vertices))
                face_materials = np.zeros(len(counts), dtype=np.int32)
                # USD binds the Panda materials to face subsets, not the mesh.
                for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
                    material = read_material(subset.GetPrim())
                    if material not in materials:
                        materials.append(material)
                    face_materials[np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)] = materials.index(material)
                triangle_materials = np.repeat(face_materials, np.maximum(counts - 2, 0))
                faces, start = [], 0
                for count in counts:
                    faces.extend((ids[start], ids[start+i], ids[start+i+1]) for i in range(1, int(count)-1))
                    start += int(count)
                faces = np.asarray(faces, dtype=np.uint32)
                if mesh.GetOrientationAttr().Get() == 'leftHanded':
                    faces = faces[:, ::-1]
            else:
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
                vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * size
                faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]],dtype=np.uint32)
                triangle_materials = np.zeros(len(faces), dtype=np.int32)
                vertex_colors = None
            if not len(vertices) or not len(faces):
                continue
            # Vertex clustering for the dense reconstructed bowls only. Keep a
            # fine grid because these scans have thin inner/outer walls and rims;
            # coarse cells can merge those surfaces and leave visible pinholes.
            if len(vertices) > 50000:
                step = max(np.ptp(vertices, axis=0)) / 112
                _, inverse = np.unique(np.floor((vertices-vertices.min(axis=0))/step).astype(np.int32),axis=0,return_inverse=True)
                weights = np.bincount(inverse)
                vertices = np.column_stack([np.bincount(inverse, weights=vertices[:,a])/weights for a in range(3)])
                if vertex_colors is not None:
                    vertex_colors = np.column_stack([
                        np.bincount(inverse, weights=vertex_colors[:,a])/weights for a in range(3)
                    ])
                faces = inverse[faces]
                valid = (faces[:,0]!=faces[:,1]) & (faces[:,1]!=faces[:,2]) & (faces[:,0]!=faces[:,2])
                faces = faces[valid]
                triangle_materials = triangle_materials[valid]
                _, unique = np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
                faces = faces[np.sort(unique)]
                triangle_materials = triangle_materials[np.sort(unique)]
            vertices = vertices @ transform[:3,:3].T + transform[:3,3]
            color = materials[0]['color']
            display = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            if display is not None and len(display):
                color = np.asarray(display).mean(axis=0).tolist()
            if body >= 0 and names[body] == 'object/yellow_bowl': color = [0.95,0.72,0.08]
            if body >= 0 and names[body] == 'object/white_bowl': color = [0.86,0.87,0.84]
            materials[0]['color'] = color
            # Group by material without duplicating vertex buffers. Subsets using
            # identical materials are merged to minimize rendering draw calls.
            order = np.argsort(triangle_materials, kind='stable')
            faces = faces[order]
            groups, used_materials, start = [], [], 0
            for material_id in np.unique(triangle_materials):
                count = int(np.count_nonzero(triangle_materials == material_id)) * 3
                groups.append({'start': start, 'count': count, 'material': len(used_materials)})
                used_materials.append(materials[material_id])
                start += count
            target_payload = shared_payload if is_robot else payload
            exported = {'name':path, 'body':body_name if is_robot else body, 'color':color,
                        'materials':used_materials, 'groups':groups,
                        'position':pack(vertices,'<f4',target_payload),
                        'index':pack(faces,'<u4',target_payload)}
            if vertex_colors is not None:
                exported['vertexColor'] = pack(np.rint(vertex_colors * 255),'<u1',target_payload)
            (shared_meshes if is_robot else meshes).append(exported)
        visible_bodies = {m['body'] for m in meshes}
        visible_bodies.update(names.index(m['body']) for m in shared_meshes if m['body'] in names)
        assert set(range(len(names))) <= visible_bodies, 'Missing visible body geometry'
        pose_block = pack(np.concatenate([positions,rotations],axis=2),'<f4')
        scene_number = int(meta['scene'].split('_')[1])
        source_identifier = f"scene-{scene_number}-demo-{int(meta['demo'].split('_')[1])}"
        # Simulation demonstrations may intentionally differ from the synthetic
        # video demo. Match the website task by scene, not by rollout demo number.
        identifier = site_ids.get(scene_number, source_identifier)
        descriptor = {'bodies':names,'meshes':meshes,'poses':pose_block,'timestamps':times,
                      'phases':phases,'phaseNames':meta['phases'],'taskFrames':task_frames,
                      'robotBase':meta['robot_base_position'],'binary':f'{identifier}.bin',
                      'sourceRollout':source_identifier}
        (args.output / f'{identifier}.bin').write_bytes(payload)
        (args.output / f'{identifier}.json').write_text(json.dumps(descriptor,separators=(',',':'))+'\n')
        index[identifier] = f'{identifier}.json'
        source_note = '' if source_identifier == identifier else f' (from {source_identifier})'
        print(f'{identifier}{source_note}: {len(meshes)} meshes, {len(times)} frames, {len(payload)/1048576:.2f} MiB')
        if export_shared_robot:
            (args.output / 'robot.bin').write_bytes(shared_payload)
            shared = {'binary':'robot.bin','meshes':shared_meshes}
            (args.output / 'robot.json').write_text(json.dumps(shared,separators=(',',':'))+'\n')
            print(f'robot: {len(shared_meshes)} meshes, {len(shared_payload)/1048576:.2f} MiB (shared)')
            export_shared_robot = False
    manifest = {'format':'caster.simulation.v2','shared':'robot.json','rollouts':index}
    (args.output / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    keep = {'manifest.json', 'robot.json', 'robot.bin'}
    keep.update(index.values())
    keep.update(Path(path).with_suffix('.bin').name for path in index.values())
    for path in args.output.glob('scene-*-demo-*.*'):
        if path.suffix in ('.json', '.bin') and path.name not in keep:
            path.unlink()
            print(f'removed stale asset: {path.name}')


if __name__ == '__main__':
    main()
