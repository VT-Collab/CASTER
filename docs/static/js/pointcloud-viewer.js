import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { taskPrompts } from './task-prompts.js';
import { SimulationViewer } from './simulation-viewer.js';

const ASSET_ROOT = './static/resources/pointcloud/';
const REAL_VIDEO_ROOT = './static/resources/real_videos/web/';
const REAL_VIDEO_PLAYBACK_RATE = 2;
const gallery = document.querySelector('#pointcloud-gallery');

if (gallery) {
  initialiseExplorer().catch((error) => {
    console.error(error);
    gallery.innerHTML = `
      <div class="gallery-error" role="alert">
        The interactive point-cloud data could not be loaded. Please refresh the page and try again.
      </div>`;
  });
}

async function initialiseExplorer() {
  const response = await fetch(`${ASSET_ROOT}manifest.json`);
  if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);

  const manifest = await response.json();
  const meshManifest = await fetch(`${ASSET_ROOT}meshes/manifest.json`)
    .then((result) => result.ok ? result.json() : {})
    .catch(() => ({}));
  const realVideoManifest = await fetch(`${REAL_VIDEO_ROOT}manifest.json`)
    .then((result) => result.ok ? result.json() : {videos:{}})
    .catch(() => ({videos:{}}));
  if (!manifest.examples?.length) throw new Error('The point-cloud manifest is empty.');

  gallery.innerHTML = buildInterface(manifest.examples);

  const canvas = gallery.querySelector('.shared-canvas');
  const canvasWrap = gallery.querySelector('.shared-canvas-wrap');
  const taskTitle = gallery.querySelector('.shared-task-title');
  const taskPrompt = gallery.querySelector('.shared-task-prompt');
  const taskPromptText = taskPrompt.querySelector('.generation-prompt-text');
  const taskButtons = [...gallery.querySelectorAll('.task-menu-button')];
  const playButton = gallery.querySelector('.shared-video-wrap .canvas-play-button');
  const timeline = gallery.querySelector('.shared-timeline input');
  const frameOutput = gallery.querySelector('.shared-timeline output');
  const loading = gallery.querySelector('.shared-loading');
  const video = gallery.querySelector('.shared-video-preview video');
  const realVideo = gallery.querySelector('.real-video-wrap video');
  const realPlayButton = gallery.querySelector('.real-video-wrap .canvas-play-button');
  const realPlaceholder = gallery.querySelector('.real-rollout .rollout-empty');
  const layerInputs = [...gallery.querySelectorAll('.shared-layers input')];

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xe8ece8);
  scene.fog = new THREE.FogExp2(0xe8ece8, 0.028);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x92968b, 2));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2);
  keyLight.position.set(2, -3, 5);
  scene.add(keyLight);

  const camera = new THREE.PerspectiveCamera(44, 1, 0.01, 100);
  camera.up.set(0, 0, 1);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  let controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.screenSpacePanning = true;

  const content = new THREE.Group();
  scene.add(content);

  const bufferCache = new Map();
  let currentExample = null;
  let currentIndex = 0;
  let loadSequence = 0;
  let sceneCloud = null;
  let videoCloud = null;
  let videoPositions = null;
  let videoColors = null;
  let trackClouds = [];
  let meshes = [];
  let grid = null;
  let axes = null;
  let playing = false;
  let lastDisplayedFrame = -1;
  const simulation = new SimulationViewer(gallery.querySelector('.simulation-rollout'), () => {
    video.pause();
    realVideo.pause();
    setPlaying(false);
  });

  const resizeObserver = new ResizeObserver(resizeRenderer);
  resizeObserver.observe(canvasWrap);

  taskButtons.forEach((button, index) => {
    button.addEventListener('click', () => loadExample(index));
  });

  playButton.addEventListener('click', async () => {
    if (!currentExample || playButton.disabled) return;
    if (playing) {
      video.pause();
      setPlaying(false);
      return;
    }
    if (Number(timeline.value) >= Number(timeline.max)) {
      setFrame(0, true);
    }
    try {
      simulation.pause();
      realVideo.pause();
      await video.play();
      setPlaying(true);
    } catch (error) {
      console.warn('Video playback was blocked:', error);
    }
  });

  video.addEventListener('play', () => setPlaying(true));
  video.addEventListener('pause', () => setPlaying(false));
  realPlayButton.addEventListener('click', async () => {
    if (!realVideo.src) return;
    if (!realVideo.paused) {
      realVideo.pause();
      return;
    }
    video.pause();
    simulation.pause();
    setPlaying(false);
    try {
      await realVideo.play();
    } catch (error) {
      console.warn('Real-world video playback was blocked:', error);
    }
  });
  const setRealPlaying = (active) => {
    realPlayButton.classList.toggle('is-playing',active);
    const label = active ? 'Pause real-world rollout' : 'Play real-world rollout';
    realPlayButton.setAttribute('aria-label',label);
    realPlayButton.title = label;
  };
  realVideo.addEventListener('play',() => setRealPlaying(true));
  realVideo.addEventListener('pause',() => setRealPlaying(false));
  realVideo.addEventListener('ended',() => setRealPlaying(false));
  realVideo.addEventListener('loadedmetadata',() => {
    realVideo.defaultPlaybackRate = REAL_VIDEO_PLAYBACK_RATE;
    realVideo.playbackRate = REAL_VIDEO_PLAYBACK_RATE;
  });

  timeline.addEventListener('input', () => {
    video.pause();
    realVideo.pause();
    setRealPlaying(false);
    setPlaying(false);
    setFrame(Number(timeline.value), true);
  });

  layerInputs.forEach((input) => {
    input.addEventListener('change', () => {
      const target = input.dataset.layer;
      if (target === 'scene' && sceneCloud) sceneCloud.visible = input.checked;
      if (target === 'video' && videoCloud) videoCloud.visible = input.checked;
      if (target === 'tracks') trackClouds.forEach((cloud) => { cloud.visible = input.checked; });
      if (target === 'meshes') meshes.forEach((mesh) => { mesh.visible = input.checked; });
    });
  });

  await loadExample(0);
  renderer.setAnimationLoop(render);

  function buildPoints(positions, colors, size, opacity = 1) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (colors) geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
    const material = new THREE.PointsMaterial({
      size,
      sizeAttenuation: true,
      vertexColors: Boolean(colors),
      transparent: opacity < 1,
      opacity,
      depthWrite: opacity >= 1,
    });
    return new THREE.Points(geometry, material);
  }

  async function loadExample(index) {
    const sequence = ++loadSequence;
    const example = manifest.examples[index];
    currentIndex = index;
    currentExample = example;
    video.pause();
    realVideo.pause();
    setRealPlaying(false);
    setPlaying(false);
    setLoading(true);

    taskButtons.forEach((button, buttonIndex) => {
      const active = buttonIndex === index;
      button.classList.toggle('is-active', active);
      if (active) button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    });
    taskTitle.textContent = example.label;
    taskPromptText.textContent = taskPrompts[example.id] || '';
    taskPrompt.hidden = !taskPromptText.textContent;
    simulation.load(example.id, example.camera);
    const realSource = realVideoManifest.videos?.[example.id];
    realVideo.hidden = !realSource;
    realPlayButton.hidden = !realSource;
    realPlaceholder.hidden = Boolean(realSource);
    if (realSource) {
      realVideo.src = `${REAL_VIDEO_ROOT}${realSource}`;
      realVideo.defaultPlaybackRate = REAL_VIDEO_PLAYBACK_RATE;
      realVideo.playbackRate = REAL_VIDEO_PLAYBACK_RATE;
      realVideo.load();
    } else {
      realVideo.removeAttribute('src');
      realVideo.load();
    }

    try {
      const buffer = await fetchBuffer(example.data);
      if (sequence !== loadSequence) return;

      clearContent();
      const minimum = example.bounds.min;
      const extent = example.bounds.extent;

      const scenePositions = decodePositions(buffer, example.scene.position, example.scene.pointCount, minimum, extent);
      const sceneColors = byteView(buffer, example.scene.color, example.scene.pointCount * 3);
      sceneCloud = buildPoints(scenePositions, sceneColors, 0.0135, 0.82);
      sceneCloud.name = 'scene';
      content.add(sceneCloud);

      const frameCount = example.videoCloud.frameCount;
      const videoPointCount = example.videoCloud.pointCount;
      videoPositions = decodePositions(buffer, example.videoCloud.position, frameCount * videoPointCount, minimum, extent);
      videoColors = byteView(buffer, example.videoCloud.color, frameCount * videoPointCount * 3);
      videoCloud = buildPoints(
        new Float32Array(videoPointCount * 3),
        new Uint8Array(videoPointCount * 3),
        0.018,
      );
      videoCloud.name = 'video';
      content.add(videoCloud);

      trackClouds = buildTrackTrails(example, buffer);
      trackClouds.forEach((object) => content.add(object));

      buildGroundReference(example.bounds);
      frameRange(example);
      applyLayerVisibility();
      fitCamera(example);
      setFrame(0, false);

      video.src = `${ASSET_ROOT}${example.video}`;
      video.load();
      setLoading(false);
      loadMeshes(example, sequence);
    } catch (error) {
      if (sequence !== loadSequence) return;
      console.error(error);
      loading.innerHTML = '<span>Unable to load this task.</span>';
      loading.classList.add('is-visible', 'is-error');
      playButton.disabled = true;
      timeline.disabled = true;
    }
  }

  async function loadMeshes(example, sequence) {
    const status = gallery.querySelector('.mesh-status');
    const entries = meshManifest[example.id] || [];
    status.hidden = true;
    const results = await Promise.allSettled(entries.map(async (entry) => {
      const buffer = await fetchBuffer(entry.data);
      if (sequence !== loadSequence) return;
      const header = new DataView(buffer);
      const vertexCount = header.getUint32(0, true);
      const triangleCount = header.getUint32(4, true);
      const packed = new Float32Array(buffer, 8, vertexCount * 6);
      const positions = new Float32Array(vertexCount * 3);
      const colors = new Float32Array(vertexCount * 3);
      const color = new THREE.Color();
      for (let i = 0; i < vertexCount; i++) {
        positions.set(packed.subarray(i * 6, i * 6 + 3), i * 3);
        color.setRGB(packed[i * 6 + 3], packed[i * 6 + 4], packed[i * 6 + 5], THREE.SRGBColorSpace);
        color.toArray(colors, i * 3);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, 8 + vertexCount * 24, triangleCount * 3), 1));
      geometry.computeVertexNormals();
      const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.85, side: THREE.DoubleSide,
        transparent: true, opacity: 0.8, depthWrite: false,
      }));
      mesh.name = entry.name;
      mesh.position.fromArray(entry.translation || [0, 0, 0]);
      const [w, x, y, z] = entry.rotation_wxyz || entry.rotation || [1, 0, 0, 0];
      mesh.quaternion.set(x, y, z, w).normalize().multiply(
        new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2),
      );
      mesh.scale.setScalar(Array.isArray(entry.scale) ? entry.scale[0] : (entry.scale ?? 1));
      mesh.visible = gallery.querySelector('[data-layer="meshes"]').checked;
      meshes.push(mesh);
      content.add(mesh);
    }));
    if (sequence !== loadSequence) return;
    const failed = results.filter((result) => result.status === 'rejected');
    status.hidden = !failed.length;
    status.textContent = failed.length ? 'Some meshes could not be loaded. Select the task to retry.' : '';
  }

  function fetchBuffer(filename) {
    if (!bufferCache.has(filename)) {
      const request = fetch(`${ASSET_ROOT}${filename}`).then((response) => {
        if (!response.ok) throw new Error(`Point data request failed (${response.status})`);
        return response.arrayBuffer();
      }).catch((error) => { bufferCache.delete(filename); throw error; });
      bufferCache.set(filename, request);
    }
    return bufferCache.get(filename);
  }

  function setFrame(frame, syncVideo) {
    if (!currentExample || !videoCloud) return;
    const frameCount = currentExample.videoCloud.frameCount;
    const safeFrame = THREE.MathUtils.clamp(Math.round(frame), 0, frameCount - 1);
    const videoPointCount = currentExample.videoCloud.pointCount;
    const videoStart = safeFrame * videoPointCount * 3;
    videoCloud.geometry.attributes.position.array.set(
      videoPositions.subarray(videoStart, videoStart + videoPointCount * 3),
    );
    videoCloud.geometry.attributes.color.array.set(
      videoColors.subarray(videoStart, videoStart + videoPointCount * 3),
    );
    videoCloud.geometry.attributes.position.needsUpdate = true;
    videoCloud.geometry.attributes.color.needsUpdate = true;

    trackClouds.forEach((object) => {
      const count = object.userData.pointCount;
      // Frame-major buffers stay on the GPU; scrubbing only changes draw ranges.
      if (object.isLineSegments) object.geometry.setDrawRange(0, safeFrame * count * 2);
      else object.geometry.setDrawRange(safeFrame * count, count);
    });

    timeline.value = String(safeFrame);
    frameOutput.value = `${safeFrame + 1} / ${frameCount}`;
    frameOutput.textContent = frameOutput.value;
    lastDisplayedFrame = safeFrame;

    if (syncVideo) {
      const sourceFrame = currentExample.frameIndices[safeFrame];
      video.currentTime = sourceFrame / currentExample.sourceFps;
      simulation.seekTaskFrame(sourceFrame);
    }
  }

  function render() {
    if (playing && currentExample && Number.isFinite(video.currentTime)) {
      const sourceFrame = video.currentTime * currentExample.sourceFps;
      simulation.seekTaskFrame(sourceFrame);
      let closest = 0;
      let smallestDistance = Infinity;
      currentExample.frameIndices.forEach((frame, index) => {
        const distance = Math.abs(frame - sourceFrame);
        if (distance < smallestDistance) {
          smallestDistance = distance;
          closest = index;
        }
      });
      if (closest !== lastDisplayedFrame) setFrame(closest, false);
    }
    controls.update();
    renderer.render(scene, camera);
  }

  function frameRange(example) {
    timeline.min = '0';
    timeline.max = String(example.videoCloud.frameCount - 1);
    timeline.step = '1';
    timeline.value = '0';
  }

  function setPlaying(value) {
    playing = value;
    playButton.classList.toggle('is-playing', value);
    const action = value ? 'Pause animation' : 'Play animation';
    playButton.setAttribute('aria-label', action);
    playButton.title = action;
  }

  function setLoading(value) {
    loading.innerHTML = '<span class="viewer-spinner" aria-hidden="true"></span><span>Loading point cloud&hellip;</span>';
    loading.classList.toggle('is-visible', value);
    loading.classList.remove('is-error');
    playButton.disabled = value;
    timeline.disabled = value;
  }

  function applyLayerVisibility() {
    layerInputs.forEach((input) => {
      if (input.dataset.layer === 'scene' && sceneCloud) sceneCloud.visible = input.checked;
      if (input.dataset.layer === 'video' && videoCloud) videoCloud.visible = input.checked;
      if (input.dataset.layer === 'tracks') trackClouds.forEach((cloud) => { cloud.visible = input.checked; });
    });
  }

  function buildGroundReference(bounds) {
    const extent = bounds.extent;
    const centerX = bounds.min[0] + extent[0] / 2;
    const centerY = bounds.min[1] + extent[1] / 2;
    const floorZ = bounds.min[2];
    const size = Math.max(extent[0], extent[1]) * 1.15;
    grid = new THREE.GridHelper(size, 12, 0xc0c8c0, 0xd6ddd6);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(centerX, centerY, floorZ);
    grid.material.transparent = true;
    grid.material.opacity = 0.48;
    content.add(grid);

    axes = new THREE.AxesHelper(size * 0.12);
    // All exported geometry is in robot coordinates: the base frame is zero.
    axes.position.set(0, 0, 0);
    axes.material.transparent = true;
    axes.material.opacity = 0.75;
    content.add(axes);
  }

  function fitCamera(example) {
    const bounds = example.bounds;
    const center = new THREE.Vector3(
      bounds.min[0] + bounds.extent[0] / 2,
      bounds.min[1] + bounds.extent[1] / 2,
      bounds.min[2] + bounds.extent[2] / 2,
    );
    const radius = Math.max(...bounds.extent);
    controls.dispose();
    let target = center;
    if (example.camera) {
      const matrix = example.camera.cameraToRobot;
      camera.position.set(matrix[0][3], matrix[1][3], matrix[2][3]);
      // OpenCV: +Z forward, +Y down. Three.js: -Z forward, +Y up.
      const forward = new THREE.Vector3(matrix[0][2], matrix[1][2], matrix[2][2]).normalize();
      camera.up.set(-matrix[0][1], -matrix[1][1], -matrix[2][1]).normalize();
      const depth = Math.max(center.clone().sub(camera.position).dot(forward), radius * 0.25);
      target = camera.position.clone().addScaledVector(forward, depth);
    } else {
      camera.up.set(0, 0, 1);
      camera.position.set(center.x + radius * 0.85, center.y - radius * 1.05, center.z + radius * 0.72);
    }
    camera.lookAt(target);
    // Recreate controls after changing the up axis, also clearing old damping.
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.screenSpacePanning = true;
    controls.target.copy(target);
    camera.near = Math.max(radius / 1000, 0.002);
    camera.far = radius * 20;
    updateCameraProjection();
    controls.minDistance = radius * 0.18;
    controls.maxDistance = radius * 5;
    controls.update();
  }

  function clearContent() {
    for (const child of [...content.children]) {
      content.remove(child);
      child.geometry?.dispose();
      if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose());
      else child.material?.dispose();
    }
    sceneCloud = null;
    videoCloud = null;
    trackClouds = [];
    meshes = [];
    grid = null;
    axes = null;
  }

  function resizeRenderer() {
    const width = canvasWrap.clientWidth;
    const height = canvasWrap.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    updateCameraProjection();
  }

  function updateCameraProjection() {
    const intrinsics = currentExample?.camera?.intrinsics;
    if (!intrinsics) {
      camera.fov = 44;
      camera.updateProjectionMatrix();
      return;
    }
    const { width, height, fx, fy, cx, cy } = intrinsics;
    // Fit the calibrated image within any viewport shape without cropping it.
    const viewHeight = Math.max(height, width / camera.aspect);
    const viewWidth = viewHeight * camera.aspect;
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(viewHeight / (2 * fy)));
    camera.updateProjectionMatrix();
    camera.projectionMatrix.elements[0] = 2 * fx / viewWidth;
    camera.projectionMatrix.elements[8] = (width - 2 * cx) / viewWidth;
    camera.projectionMatrix.elements[9] = (2 * cy - height) / viewHeight;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
}

function byteView(buffer, descriptor, count) {
  return new Uint8Array(buffer, descriptor.offset, count);
}

function buildTrackTrails(example, buffer) {
  const frameCount = example.videoCloud.frameCount;
  const pointCount = example.tracks.reduce((sum, track) => sum + track.pointCount, 0);
  if (!pointCount || !frameCount) return [];

  // Reuse a spatial palette for every object; point identities retain their color.
  const palette = ['#8c36ef', '#395bf0', '#29bce8', '#82e5d1', '#bbec73', '#efe66c']
    .map((hex) => new THREE.Color(hex));
  const positions = new Float32Array(frameCount * pointCount * 3);
  const colors = new Float32Array(positions.length);
  let objectOffset = 0;
  for (const track of example.tracks) {
    const decoded = decodePositions(buffer, track.position, frameCount * track.pointCount,
      example.bounds.min, example.bounds.extent);
    // Color along the object's longest initial spatial axis. Compute only once,
    // so motion and rotation cannot change a tracked point's assigned color.
    const lower = [Infinity, Infinity, Infinity];
    const upper = [-Infinity, -Infinity, -Infinity];
    for (let point = 0; point < track.pointCount; point++) {
      for (let axis = 0; axis < 3; axis++) {
        const value = decoded[point * 3 + axis];
        lower[axis] = Math.min(lower[axis], value);
        upper[axis] = Math.max(upper[axis], value);
      }
    }
    const spans = upper.map((value, axis) => value - lower[axis]);
    const axis = spans.indexOf(Math.max(...spans));
    const pointColors = new Float32Array(track.pointCount * 3);
    const color = new THREE.Color();
    for (let point = 0; point < track.pointCount; point++) {
      const fraction = spans[axis] > 1e-8
        ? (decoded[point * 3 + axis] - lower[axis]) / spans[axis] : 0.5;
      const progress = THREE.MathUtils.clamp(fraction, 0, 1) * (palette.length - 1);
      const stop = Math.min(Math.floor(progress), palette.length - 2);
      color.copy(palette[stop]).lerp(palette[stop + 1], progress - stop);
      color.toArray(pointColors, point * 3);
    }
    for (let frame = 0; frame < frameCount; frame++) {
      const start = frame * track.pointCount * 3;
      positions.set(decoded.subarray(start, start + track.pointCount * 3),
        (frame * pointCount + objectOffset) * 3);
      colors.set(pointColors, (frame * pointCount + objectOffset) * 3);
    }
    objectOffset += track.pointCount;
  }

  // Indexed segments reuse each vertex across adjacent timesteps. Two draw calls
  // cover every object: thin accumulated trails and small current-frame tips.
  const indices = new Uint32Array(Math.max(0, frameCount - 1) * pointCount * 2);
  let cursor = 0;
  for (let frame = 1; frame < frameCount; frame++) {
    for (let point = 0; point < pointCount; point++) {
      indices[cursor++] = (frame - 1) * pointCount + point;
      indices[cursor++] = frame * pointCount + point;
    }
  }
  const positionAttribute = new THREE.BufferAttribute(positions, 3);
  const colorAttribute = new THREE.BufferAttribute(colors, 3);
  const trailGeometry = new THREE.BufferGeometry();
  trailGeometry.setAttribute('position', positionAttribute);
  trailGeometry.setAttribute('color', colorAttribute);
  trailGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
  trailGeometry.computeBoundingSphere();
  const trails = new THREE.LineSegments(trailGeometry, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.65, depthWrite: false,
  }));
  const tipGeometry = new THREE.BufferGeometry();
  tipGeometry.setAttribute('position', positionAttribute);
  tipGeometry.setAttribute('color', colorAttribute);
  tipGeometry.boundingSphere = trailGeometry.boundingSphere.clone();
  const tips = new THREE.Points(tipGeometry, new THREE.PointsMaterial({
    vertexColors: true, size: 0.012, sizeAttenuation: true,
    transparent: true, opacity: 0.9, depthWrite: false,
  }));
  trails.userData.pointCount = tips.userData.pointCount = pointCount;
  return [trails, tips];
}

function decodePositions(buffer, descriptor, pointCount, minimum, extent) {
  const quantized = new Uint16Array(buffer, descriptor.offset, pointCount * 3);
  const decoded = new Float32Array(pointCount * 3);
  for (let index = 0; index < decoded.length; index += 3) {
    decoded[index] = minimum[0] + (quantized[index] / 65535) * extent[0];
    decoded[index + 1] = minimum[1] + (quantized[index + 1] / 65535) * extent[1];
    decoded[index + 2] = minimum[2] + (quantized[index + 2] / 65535) * extent[2];
  }
  return decoded;
}

function buildInterface(examples) {
  const tasks = examples.map((example, index) => `
    <button class="task-menu-button${index === 0 ? ' is-active' : ''}" type="button"
      data-index="${index}"${index === 0 ? ' aria-current="true"' : ''}>
      ${escapeHtml(example.label)}
    </button>`).join('');

  return `
    <div class="pointcloud-browser">
      <nav class="task-menu" aria-label="Choose a task">
        <p class="task-menu-label">Tasks</p>
        <div class="task-menu-items">${tasks}</div>
      </nav>

      <article class="shared-viewer">
        <header class="shared-viewer-header">
          <h3 class="shared-task-title">${escapeHtml(examples[0].label)}</h3>
        </header>
        <p class="shared-task-prompt"><strong class="generation-prompt-label">Generation prompt:</strong> <span class="generation-prompt-text">${escapeHtml(taskPrompts[examples[0].id] || '')}</span></p>

        <div class="shared-viewer-content">
          <figure class="shared-video-preview">
            <div class="shared-video-wrap">
              <video muted playsinline loop preload="metadata"></video>
              <button class="canvas-play-button" type="button" aria-label="Play animation" title="Play animation" disabled>
                <span class="play-icon" aria-hidden="true"></span>
              </button>
            </div>
            <figcaption>Generated video</figcaption>
          </figure>
          <div class="shared-canvas-wrap">
            <canvas class="shared-canvas" aria-label="Interactive 3D point-cloud visualization"></canvas>
            <div class="shared-loading is-visible" role="status">
              <span class="viewer-spinner" aria-hidden="true"></span>
              <span>Loading point cloud&hellip;</span>
            </div>
            <span class="shared-viewer-hint">Drag to orbit &middot; Scroll to zoom</span>
          </div>
        </div>

        <div class="shared-timeline">
          <label><span>Trajectory frame</span><output>1 / 1</output></label>
          <input type="range" min="0" max="0" value="0" step="1" aria-label="Trajectory frame" disabled>
        </div>
        <fieldset class="shared-layers" aria-label="Visualization layers">
          <label><input type="checkbox" data-layer="scene" checked><span>Initial Observation</span></label>
          <label><input type="checkbox" data-layer="video" checked><span>Generated motion</span></label>
          <label><input type="checkbox" data-layer="tracks" checked><span>Object tracks</span></label>
          <label><input type="checkbox" data-layer="meshes" checked><span>Object meshes</span></label>
        </fieldset>
        <p class="mesh-status" role="status" hidden></p>
        <div class="rollout-videos">
          <figure class="simulation-rollout"></figure>
          <figure class="real-rollout">
            <div class="real-video-wrap">
              <video muted playsinline loop preload="metadata" aria-label="Real-world robot rollout"></video>
              <span class="real-video-speed" aria-label="Playback speed: two times">2×</span>
              <button class="canvas-play-button" type="button" aria-label="Play real-world rollout" title="Play real-world rollout">
                <span class="play-icon" aria-hidden="true"></span>
              </button>
              <div class="rollout-empty" hidden><span aria-hidden="true">▷</span><span>Coming soon</span></div>
            </div>
            <figcaption>Real-world rollout</figcaption>
          </figure>
        </div>
      </article>
    </div>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;',
  })[character]);
}
