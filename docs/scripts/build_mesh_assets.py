#!/usr/bin/env python3
"""Build compact colored triangle meshes with vertex clustering (stdlib only).

Saved poses are robot-frame poses; the viewer applies the reference's local
+90 degree X correction. Original research assets are never modified.
"""
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'replay_depth_align_examples'
OUTPUT = ROOT / 'static/resources/pointcloud/meshes'


def convert(path, output):
    vertices, faces = [], []
    with path.open() as source:
        for line in source:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                values = list(map(float, parts[1:]))
                vertices.append(values[:3] + (values[3:6] if len(values) >= 6 else [0.65] * 3))
            elif parts[0] == 'f':
                ids = [int(item.split('/')[0]) for item in parts[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                faces.extend((ids[0], ids[i], ids[i + 1]) for i in range(1, len(ids) - 1))
    minimum = [min(v[a] for v in vertices) for a in range(3)]
    cell = max(max(v[a] for v in vertices) - minimum[a] for a in range(3)) / 64
    clusters, sums, counts, mapping = {}, [], [], []
    for vertex in vertices:
        key = tuple(int((vertex[a] - minimum[a]) / cell) for a in range(3))
        if key not in clusters:
            clusters[key] = len(sums)
            sums.append([0.0] * 6)
            counts.append(0)
        index = clusters[key]
        mapping.append(index)
        counts[index] += 1
        for a in range(6):
            sums[index][a] += vertex[a]
    reduced = [[v / counts[i] for v in total] for i, total in enumerate(sums)]
    triangles, seen = [], set()
    for face in faces:
        mapped = tuple(mapping[i] for i in face)
        key = tuple(sorted(mapped))
        if len(set(mapped)) == 3 and key not in seen:
            triangles.append(mapped)
            seen.add(key)
    with output.open('wb') as target:
        target.write(struct.pack('<II', len(reduced), len(triangles)))
        for vertex in reduced:
            target.write(struct.pack('<6f', *vertex))
        for face in triangles:
            target.write(struct.pack('<3I', *face))
    return len(reduced), len(triangles)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    result = {}
    for example in manifest['examples']:
        directory = SOURCE / example['mesh_directory']
        poses = json.loads((directory / 'transforms.json').read_text())['objects']
        entries = []
        for pose in poses:
            source = directory / Path(pose['glb']).with_suffix('.obj')
            filename = f"scene-{example['scene']}-{source.stem}.bin"
            vertices, triangles = convert(source, OUTPUT / filename)
            entries.append({**pose, 'data': f'meshes/{filename}',
                            'vertices': vertices, 'triangles': triangles})
            print(f'{filename}: {vertices:,} vertices, {triangles:,} triangles', flush=True)
        result[f"scene-{example['scene']}-demo-{example['demo']}"] = entries
    (OUTPUT / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
