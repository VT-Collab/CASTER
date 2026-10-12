import * as THREE from 'three';

// Keep native depth on the CPU and reconstruct every pixel on the GPU. Only
// the current and next chunks are cached; switching tasks aborts pending loads.
export class RawPointCloud {
  constructor(example, video, assetRoot, onError) {
    this.example = example;
    this.assetRoot = assetRoot;
    this.onError = onError;
    this.chunks = new Map();
    this.abort = new AbortController();
    this.request = 0;
    const { width, height, pointCount } = example.videoCloud;
    const pixels = new Float32Array(pointCount * 3);
    for (let index = 0; index < pointCount; index++) {
      pixels[index * 3] = index % width;
      pixels[index * 3 + 1] = Math.floor(index / width);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pixels, 3));
    this.depth = new THREE.DataTexture(new Float32Array(pointCount), width, height, THREE.RedFormat, THREE.FloatType);
    this.depth.needsUpdate = true;
    this.video = new THREE.VideoTexture(video);
    this.video.colorSpace = THREE.SRGBColorSpace;
    this.video.flipY = true;
    const calibration = example.camera.intrinsics;
    const scaleX = width / calibration.width;
    const scaleY = height / calibration.height;
    const material = new THREE.ShaderMaterial({
      uniforms: {
        depthMap: { value: this.depth },
        videoMap: { value: this.video },
        imageSize: { value: new THREE.Vector2(width, height) },
        intrinsics: { value: new THREE.Vector4(calibration.fx * scaleX, calibration.fy * scaleY,
          calibration.cx * scaleX, calibration.cy * scaleY) },
        cameraToRobot: { value: new THREE.Matrix4().set(...example.camera.cameraToRobot.flat()) },
        pointSize: { value: 0.004 },
        viewportScale: { value: 1 },
      },
      vertexShader: `
        uniform sampler2D depthMap;
        uniform vec2 imageSize;
        uniform vec4 intrinsics;
        uniform mat4 cameraToRobot;
        uniform float pointSize;
        uniform float viewportScale;
        varying vec2 videoUv;
        varying float validDepth;
        void main() {
          vec2 depthUv = (position.xy + 0.5) / imageSize;
          videoUv = vec2(depthUv.x, 1.0 - depthUv.y);
          float z = texture2D(depthMap, depthUv).r;
          validDepth = (z > 0.0 && z < 1.0e10) ? 1.0 : 0.0;
          if (validDepth < 0.5) z = 0.0;
          vec3 cameraPoint = vec3((position.x - intrinsics.z) * z / intrinsics.x,
            (position.y - intrinsics.w) * z / intrinsics.y, z);
          vec4 viewPoint = modelViewMatrix * cameraToRobot * vec4(cameraPoint, 1.0);
          gl_Position = projectionMatrix * viewPoint;
          gl_PointSize = max(1.0, pointSize * viewportScale / max(0.001, -viewPoint.z));
        }
      `,
      fragmentShader: `
        uniform sampler2D videoMap;
        varying vec2 videoUv;
        varying float validDepth;
        void main() {
          if (validDepth < 0.5) discard;
          vec3 rgb = texture2D(videoMap, videoUv).rgb;
          vec3 linear = mix(rgb / 12.92, pow((rgb + 0.055) / 1.055, vec3(2.4)),
            step(vec3(0.04045), rgb));
          gl_FragColor = vec4(linear, 1.0);
          #include <colorspace_fragment>
        }
      `,
    });
    this.object = new THREE.Points(geometry, material);
    this.object.name = 'video';
    // Shader positions are reconstructed in robot coordinates, whereas the
    // geometry attribute contains image pixels rather than spatial bounds.
    this.object.frustumCulled = false;
  }

  resize(renderer) {
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.object.material.uniforms.viewportScale.value = size.y / 2;
  }

  async loadChunk(index) {
    if (!this.chunks.has(index)) {
      const descriptor = this.example.videoCloud.depthChunks[index];
      const pending = fetch(`${this.assetRoot}${descriptor.data}`, { signal: this.abort.signal })
        .then(async (response) => {
          if (!response.ok) throw new Error(`Depth request failed (${response.status})`);
          const buffer = await response.arrayBuffer();
          if (buffer.byteLength !== descriptor.bytes) throw new Error('Incomplete raw depth chunk.');
          return new Float32Array(buffer);
        });
      this.chunks.set(index, pending);
      pending.catch(() => { if (this.chunks.get(index) === pending) this.chunks.delete(index); });
    }
    return this.chunks.get(index);
  }

  async setFrame(frame) {
    const request = ++this.request;
    const { depthChunks, pointCount } = this.example.videoCloud;
    const chunkIndex = depthChunks.findIndex((chunk) =>
      frame >= chunk.startFrame && frame < chunk.startFrame + chunk.frameCount);
    if (chunkIndex < 0) throw new Error(`Raw depth frame ${frame} is missing.`);
    const values = await this.loadChunk(chunkIndex);
    if (request !== this.request || this.abort.signal.aborted) return;
    const offset = (frame - depthChunks[chunkIndex].startFrame) * pointCount;
    this.depth.image.data = values.subarray(offset, offset + pointCount);
    this.depth.needsUpdate = true;
    for (const index of this.chunks.keys()) {
      if (index !== chunkIndex && index !== chunkIndex + 1) this.chunks.delete(index);
    }
    if (chunkIndex + 1 < depthChunks.length) {
      this.loadChunk(chunkIndex + 1).catch((error) => {
        if (error.name !== 'AbortError') this.onError(error);
      });
    }
  }

  dispose() {
    this.abort.abort();
    this.request++;
    this.chunks.clear();
    this.depth.dispose();
    this.video.dispose();
  }
}
