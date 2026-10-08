import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DEFAULT_ROOT = './static/resources/simulation/';

export class SimulationViewer {
  constructor(element, onPlay, options = {}) {
    this.element = element;
    this.onPlay = onPlay;
    this.options = options;
    this.assetRoot = options.assetRoot || DEFAULT_ROOT;
    this.sequence = 0;
    this.time = 0;
    this.playing = false;
    this.ready = false;
    this.scratch = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Quaternion(), new THREE.Quaternion()];
    element.innerHTML = `
      <div class="simulation-placeholder rollout-empty"><span aria-hidden="true">▷</span><span>Coming soon</span></div>
      <div class="simulation-player" hidden>
        <div class="simulation-stage">
          <canvas aria-label="Interactive simulation rollout; drag to orbit, scroll to zoom"></canvas>
          <button class="canvas-play-button" type="button" aria-label="Play simulation" title="Play simulation"><span class="play-icon" aria-hidden="true"></span></button>
          <span class="simulation-hint">Drag to orbit · Scroll to zoom</span>
        </div>
        <div class="simulation-transport">
          <div><span class="simulation-phase"></span><output></output></div>
          <input type="range" min="0" step="0.001" value="0" aria-label="Simulation rollout time">
        </div>
      </div>
      <figcaption>Optimized rollout in simulation</figcaption>`;
    this.player = element.querySelector('.simulation-player');
    this.placeholder = element.querySelector('.simulation-placeholder');
    this.stage = element.querySelector('.simulation-stage');
    this.button = element.querySelector('button');
    this.slider = element.querySelector('input');
    this.output = element.querySelector('output');
    this.phase = element.querySelector('.simulation-phase');
    this.button.onclick = () => {
      if (!this.ready) return;
      if (this.playing) this.pause();
      else {
        this.onPlay();
        if (this.time >= this.duration) this.seek(0);
        this.playing = true;
        this.updateButton();
        this.stamp = null;
        this.raf = requestAnimationFrame((stamp) => this.tick(stamp));
      }
    };
    this.slider.oninput = () => { this.onPlay(); this.pause(); this.seek(Number(this.slider.value)); };
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.pause(); });
    this.manifest = fetch(`${this.assetRoot}manifest.json`).then((r) => {
      if (!r.ok) throw new Error('Simulation manifest unavailable');
      return r.json();
    });
    // Defer a manifest failure to load(), where a visible error is shown.
    this.manifest.catch(() => {});
  }

  initialiseRenderer() {
    if (this.renderer) return;
    this.renderer = new THREE.WebGLRenderer({
      canvas:this.element.querySelector('canvas'),
      antialias:true,
      preserveDrawingBuffer:Boolean(this.options.preserveDrawingBuffer),
    });
    this.renderer.setPixelRatio(this.options.pixelRatio ?? Math.min(devicePixelRatio, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#e8ece8');
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa38d, 2.1));
    const light = new THREE.DirectionalLight(0xffffff, 2.3);
    light.position.set(2,-3,5);
    this.scene.add(light);
    this.camera = new THREE.PerspectiveCamera(42,1,0.01,30);
    this.camera.up.set(0,0,1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 8;
    this.controls.addEventListener('change', () => this.draw());
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.stage);
  }

  makeMeshResources(meshes, buffer) {
    return meshes.map((mesh) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(buffer,mesh.position.offset,mesh.position.count),3));
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer,mesh.index.offset,mesh.index.count),1));
      if (mesh.vertexColor) {
        geometry.setAttribute('color',new THREE.BufferAttribute(
          new Uint8Array(buffer,mesh.vertexColor.offset,mesh.vertexColor.count),3,true));
      }
      geometry.computeVertexNormals();
      const material = (mesh.materials || [{color:mesh.color}]).map((entry) => new THREE.MeshStandardMaterial({
        color:mesh.vertexColor ? 0xffffff : new THREE.Color(...entry.color),
        vertexColors:Boolean(mesh.vertexColor),
        roughness:entry.roughness ?? 0.72,
        metalness:entry.metalness ?? 0.08,
        emissive:new THREE.Color(...(entry.emissive || [0,0,0])),
        side:THREE.DoubleSide,
      }));
      for (const group of mesh.groups || [{start:0,count:mesh.index.count,material:0}]) {
        geometry.addGroup(group.start,group.count,group.material);
      }
      return { body:mesh.body, geometry, material };
    });
  }

  loadShared(path) {
    if (!path) return Promise.resolve([]);
    if (this.sharedPath === path && this.sharedPromise) return this.sharedPromise;
    this.sharedPath = path;
    this.sharedPromise = fetch(this.assetRoot + path).then(async (response) => {
      if (!response.ok) throw new Error('Shared robot metadata unavailable');
      const data = await response.json();
      const binary = await fetch(this.assetRoot + data.binary);
      if (!binary.ok) throw new Error('Shared robot geometry unavailable');
      return this.makeMeshResources(data.meshes,await binary.arrayBuffer());
    });
    return this.sharedPromise;
  }

  async load(id, calibration) {
    const sequence = ++this.sequence;
    this.lastError = null;
    this.pause();
    this.ready = false;
    this.clear();
    this.player.hidden = true;
    this.placeholder.hidden = false;
    this.placeholder.textContent = 'Loading simulation…';
    try {
      const manifest = await this.manifest;
      if (sequence !== this.sequence) return;
      const rollouts = manifest.rollouts || manifest;
      if (!rollouts[id]) { this.placeholder.textContent = 'Coming soon'; return; }
      const response = await fetch(this.assetRoot + rollouts[id]);
      if (!response.ok) throw new Error('Simulation metadata unavailable');
      const data = await response.json();
      const [binary,sharedResources] = await Promise.all([
        fetch(this.assetRoot + data.binary),
        this.loadShared(manifest.shared),
      ]);
      if (!binary.ok) throw new Error('Simulation geometry unavailable');
      const buffer = await binary.arrayBuffer();
      if (sequence !== this.sequence) return;
      this.initialiseRenderer();
      this.data = data;
      this.poses = new Float32Array(buffer,data.poses.offset,data.poses.count);
      this.root = new THREE.Group();
      this.bodies = data.bodies.map(() => { const body = new THREE.Group(); this.root.add(body); return body; });
      for (const resource of this.makeMeshResources(data.meshes,buffer)) {
        const object = new THREE.Mesh(resource.geometry,resource.material);
        (resource.body < 0 ? this.root : this.bodies[resource.body]).add(object);
      }
      for (const resource of sharedResources) {
        const body = data.bodies.indexOf(resource.body);
        if (body < 0) throw new Error(`Missing shared robot body: ${resource.body}`);
        const object = new THREE.Mesh(resource.geometry,resource.material);
        object.userData.sharedAsset = true;
        this.bodies[body].add(object);
      }
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(5,5),new THREE.MeshStandardMaterial({color:0xdce1d8,roughness:1}));
      this.root.add(floor);
      this.scene.add(this.root);
      this.duration = data.timestamps.at(-1);
      this.slider.max = String(this.duration);
      this.taskSamples = data.taskFrames.flatMap((frame,index) => frame < 0 ? [] : [{frame,time:data.timestamps[index]}]);
      this.configureCamera(calibration);
      this.placeholder.hidden = true;
      this.player.hidden = false;
      this.ready = true;
      this.resize();
      this.seek(0);
    } catch (error) {
      if (sequence !== this.sequence) return;
      console.error(error);
      this.lastError = error;
      this.ready = false;
      this.clear();
      this.player.hidden = true;
      this.placeholder.hidden = false;
      this.placeholder.textContent = 'Simulation could not load. Select this task to retry.';
    }
  }

  seek(time) {
    if (!this.ready) return;
    this.time = THREE.MathUtils.clamp(time,0,this.duration);
    const times = this.data.timestamps;
    let lo=0, hi=times.length-1;
    while (lo+1<hi) { const mid=(lo+hi)>>1; if(times[mid]<=this.time)lo=mid;else hi=mid; }
    const alpha = THREE.MathUtils.clamp((this.time-times[lo])/(times[hi]-times[lo]),0,1);
    const [a, b, qa, qb] = this.scratch;
    this.bodies.forEach((body,index) => {
      const i=(lo*this.bodies.length+index)*7, j=(hi*this.bodies.length+index)*7, p=this.poses;
      a.fromArray(p,i); b.fromArray(p,j); body.position.copy(a).lerp(b,alpha);
      qa.set(p[i+4],p[i+5],p[i+6],p[i+3]); qb.set(p[j+4],p[j+5],p[j+6],p[j+3]);
      body.quaternion.copy(qa).slerp(qb,alpha);
    });
    this.slider.value = String(this.time);
    this.output.textContent = `${this.time.toFixed(1)} / ${this.duration.toFixed(1)} s`;
    const sample = alpha === 1 ? hi : lo;
    const label = this.data.phaseNames[this.data.phases[sample]] || '';
    this.phase.textContent = label ? label[0].toUpperCase()+label.slice(1) : '';
    this.draw();
  }

  seekTaskFrame(frame) {
    if (!this.ready || !this.taskSamples.length) return;
    this.pause();
    const samples = this.taskSamples;
    let index = samples.findIndex((sample) => sample.frame >= frame);
    if (index < 0) index = samples.length-1;
    const next = samples[index], prev = samples[Math.max(0,index-1)];
    const alpha = next.frame === prev.frame ? 0 : THREE.MathUtils.clamp((frame-prev.frame)/(next.frame-prev.frame),0,1);
    this.seek(THREE.MathUtils.lerp(prev.time,next.time,alpha));
  }

  tick(stamp) {
    if (!this.playing) return;
    if (this.stamp === null) this.stamp = stamp;
    const dt = (stamp-this.stamp)/1000;
    if (dt >= 1/30) { this.seek((this.time+Math.min(dt,0.15))%this.duration); this.stamp=stamp; }
    this.raf = requestAnimationFrame((next) => this.tick(next));
  }

  updateButton() {
    this.button.classList.toggle('is-playing',this.playing);
    const label = this.playing ? 'Pause simulation' : 'Play simulation';
    this.button.setAttribute('aria-label',label);
    this.button.title = label;
  }
  pause() {
    if (!this.playing) return;
    this.playing=false; cancelAnimationFrame(this.raf); this.updateButton();
  }
  configureCamera(calibration) {
    this.calibration = calibration;
    const base = new THREE.Vector3(...this.data.robotBase);
    const target = base.clone().add(new THREE.Vector3(0.3,0,0.18));
    this.controls.dispose();
    if (calibration) {
      const m = calibration.cameraToRobot;
      // C2R is robot-base-relative; recorded simulation poses are world-relative.
      this.camera.position.set(m[0][3],m[1][3],m[2][3]).add(base);
      const forward = new THREE.Vector3(m[0][2],m[1][2],m[2][2]).normalize();
      this.camera.up.set(-m[0][1],-m[1][1],-m[2][1]).normalize();
      const depth = Math.max(target.clone().sub(this.camera.position).dot(forward),0.3);
      target.copy(this.camera.position).addScaledVector(forward,depth);
    } else {
      this.camera.up.set(0,0,1);
      this.camera.position.copy(base).add(new THREE.Vector3(1.25,-1.65,1.05));
    }
    this.camera.lookAt(target);
    // OrbitControls caches the up-axis transform when constructed.
    this.controls = new OrbitControls(this.camera,this.renderer.domElement);
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 8;
    this.controls.target.copy(target);
    this.controls.addEventListener('change', () => this.draw());
    this.controls.update();
  }

  resize() {
    if (!this.renderer || !this.stage.clientWidth || !this.stage.clientHeight) return;
    this.renderer.setSize(this.stage.clientWidth,this.stage.clientHeight,false);
    this.camera.aspect=this.stage.clientWidth/this.stage.clientHeight;
    const k = this.calibration?.intrinsics;
    if (k) {
      const viewHeight = Math.max(k.height,k.width/this.camera.aspect);
      const viewWidth = viewHeight*this.camera.aspect;
      this.camera.fov = THREE.MathUtils.radToDeg(2*Math.atan(viewHeight/(2*k.fy)));
      this.camera.updateProjectionMatrix();
      this.camera.projectionMatrix.elements[0] = 2*k.fx/viewWidth;
      this.camera.projectionMatrix.elements[8] = (k.width-2*k.cx)/viewWidth;
      this.camera.projectionMatrix.elements[9] = (2*k.cy-k.height)/viewHeight;
      this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    } else {
      this.camera.fov = 42;
      this.camera.updateProjectionMatrix();
    }
    this.draw();
  }
  draw() { if(this.renderer && !this.player.hidden && !document.hidden)this.renderer.render(this.scene,this.camera); }
  clear() {
    if (!this.root) return;
    this.root.traverse((object) => {
      if (object.userData.sharedAsset) return;
      object.geometry?.dispose();
      if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
      else object.material?.dispose();
    });
    this.scene.remove(this.root); this.root=null; this.poses=null; this.bodies=[];
  }
}
