<h1 align="center" style="font-size: 2.0em; font-weight: bold; margin-bottom: 0; border: none; border-bottom: none;">CASTER: Formalizing Zero-Shot Manipulation from Synthetic Videos as Constrained Optimization</h1>

#####
<div align="center">
    <a href="https://collab.me.vt.edu/CASTER/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Website&color=blue"></a> &ensp;
    <a href="https://huggingface.co/datasets/JiecaoBoy/CASTER-scenes"><img src="https://img.shields.io/static/v1?label=Dataset&message=Hugging%20Face&color=orange" alt="Dataset: CASTER paper scenes on Hugging Face"></a> &ensp;
    <!-- <a href=""><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red"></a> &ensp;  -->
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
</div>

> Implementation of CASTER, a framework that extract task-relevant trajectory features from synthetic videos and finetune them in a digital twin with constrained optimization.



https://github.com/user-attachments/assets/7cdc37f8-71ef-4e11-857c-0ee05947b315



<video src="assets/CASTER_vi.mp4" autoplay loop muted playsinline controls width="100%"></video>

## Setup Instructions

From the repository root, initialize the submodules and install its
uv environment:

```bash
git submodule update --init
uv sync
```

<details>
<summary>Submodules</summary>

[DA3], [DEVA], [TrackCraft3R], and M2T2 share Caster's uv environment. M2T2's PointNet++ operators compile
on first use. [DA3]'s Open3D is pinned to 0.19 for the tested glibc 2.34 cluster nodes.
Model downloads use the repository-level `models/` directory for Hugging Face, ModelScope, Torch, and [DEVA]. [TrackCraft3R] uses its documented `checkpoints/` layout when present.

</details>

<details>
<summary>SAM 3D</summary>

SAM 3D Objects requires a separate environment. Follow its
[official installation and checkpoint download instructions](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md), including requesting model access and authenticating
with Hugging Face. Download the model and move the downloaded `checkpoints/`
directory to `src/sam-3d-objects/checkpoints/hf/` relative to the CASTER root.

</details>

<details>
<summary>Isaac</summary>

Simulation uses Isaac Sim 6.0.1 in a separate container. First prepare and enter
a GPU-enabled Isaac Sim container following the
[official container installation instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html),
using the `nvcr.io/nvidia/isaac-sim:6.0.1` image.
Mount the CASTER repository and dataset at the same absolute paths used on the host.
The commands below run inside that container, from the repository root, and
assume Isaac Sim is installed at `/isaac-sim`; adjust that path for your image.

<details>
<summary>Apptainer on HPC clusters</summary>

See the [Isaac Lab cluster guide](https://isaac-sim.github.io/IsaacLab/main/source/deployment/cluster.html).

</details>
<br>

Install Isaac Lab and CASTER's simulation dependencies in the container's Python
environment. See the [official Isaac Lab binary installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)
for details:

```bash
source /isaac-sim/setup_python_env.sh
ln -s /isaac-sim src/IsaacLab/_isaac_sim  # Create once.
cd src/IsaacLab
./isaaclab.sh --install core
cd ../..
/isaac-sim/python.sh -m pip install -r src/caster/sim_requirements.txt
```

</details>

<details>
<summary>APIs</summary>

Caster uses Gemini Robotics ER for open vocabulary object detection and grasp
contact queries. Task-feature parsing uses a text model of your choice through
an OpenAI-compatible API. Create a `.env` file in the repository root:

```dotenv
# Gemini Robotics ER
GENAI_API_KEY=your-google-api-key

# Task-feature parsing
OPENAI_API_KEY=your-endpoint-api-key
OPENAI_MODEL=your-model-name

# Optional: set this when using a custom OpenAI-compatible endpoint.
# OPENAI_API_BASE=https://your-server.example/v1
```

</details>

## Building Dataset

With your RGB-D camera server running, capture scenes using your calibrated
camera-to-robot transform:

```bash
uv run python src/caster/real/capture_dataset.py c2r=/path/to/C2R.npy
```

Open the printed browser URL and click **Capture scene** to save a new
`dataset/scene_N/assets/` directory. Override `connect=...` for your camera server
and `intrinsics=...` for your camera's intrinsics; the default is
[config/camera_intrinsics.yaml](src/caster/config/camera_intrinsics.yaml).

Provide the scene inputs under `assets/`; reconstruction adds `masks` and `meshes`:

```text
dataset/scene_<scene_num>/assets/
├── rgb.png              # Input RGB image
├── pointcloud.ply       # Input point cloud
├── C2R.npy              # Camera-to-robot transform
├── masks/               # Object masks
└── meshes/              # Object meshes and transformations
```

<details>
<summary>Asset formats</summary>

- **C2R.npy:** a NumPy file containing a finite `4 × 4` homogeneous transform
  from camera to robot coordinates: `p_robot = R @ p_camera + t`.
- **rgb.png:** an 8-bit, three-channel color PNG at the resolution specified by
  the intrinsics YAML. Depth must be registered to this RGB image.
- **pointcloud.ply:** a binary little-endian point cloud with float32 `x`, `y`,
  and `z` coordinates in metres, with no colors or faces. Points use the camera
  frame: X right, Y down, Z forward; C2R must use
  this same camera frame.

</details>

## Video generation

This repo uses HunyuanVideo 1.5 to generate synthetic videos conditioned on initial frame through
[Diffusers](https://huggingface.co/docs/diffusers/v0.40.0/api/pipelines/hunyuan_video15).
It uses the [480p step-distilled checkpoint](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled).

```bash
uv run python src/caster/video_gen.py video_gen_prompt=pour scene_num=1 'seeds=[0,1]'
```

Settings live in [config/video_gen.yaml](src/caster/config/video_gen.yaml), and task prompts
live in [config/video_gen_prompt](src/caster/config/video_gen_prompt). Each generated video is validated against its prompt with Qwen3.5-9B. Scores are saved by seed in `validation_results.json` after each video.

## Digital twin reconstruction

`caster.reconstruct` creates object meshes and their transformation in the robot frame. We first use Gemini Robotics ER to identify objects and LangSAM to segment them. Put
`GENAI_API_KEY` in the repository-root `.env`, or export it before running.

Start the SAM 3D server in a separate terminal. The server uses its own
environment separate from Caster's uv environment.

```bash
sam3d-objects/bin/python src/caster/servers/sam3d_server.py
```

Run the scene reconstruction:

```bash
uv run python src/caster/reconstruct.py scene_num=1
```

The default camera calibration is
[config/camera_intrinsics.yaml](src/caster/config/camera_intrinsics.yaml);
override `camera_intrinsics` for your own camera. Reconstruction also predicts grasps and saves them in
`assets/meshes/grasp.json`. 

## Extract trajectory features

The pipeline runs `caster.reconstruct` first if no OBJ mesh files are yet available `assets/meshes`. The SAM 3D server must be running if reconstruction may be needed.
For task-feature reasoning, configure `OPENAI_API_KEY` and `OPENAI_MODEL` as
described in [Setup Instructions](#setup-instructions); 

Start [DA3], [DEVA], and [TrackCraft3R] services with one command:

```bash
uv run python src/caster/servers/servers.py
```

Then process the existing videos:

```bash
uv run python src/caster/pipeline.py video_gen_prompt=pour scene_num=1
```

After tracking, the pipeline combines the selected videos' relative object
positions and orientations into numerical task features. It saves
`tf_<index>_trajectory.json` and `tf_<index>_trajectory_weights.png` in
`dataset/scene_<scene_num>/cf_<cf_index>_<task>/`. Settings live under `weighting` in
[config/pipeline.yaml](src/caster/config/pipeline.yaml).

<details>
<summary>Visualize aligned depth and trajectories</summary>

Replay recovered trajectories and aligned depth in a browser, you can check by opening the URL printed by Viser (usually `http://localhost:8080`):

```bash
uv run python src/caster/visualize_dataset.py scene_num=1
```

</details>



## Simulation replay

This matches the "retarget" baseline in paper. Inside the prepared Isaac Sim container, start the simulation server with the
environment configured as described above in [Setup Instructions](#setup-instructions);:

```bash
source /isaac-sim/setup_python_env.sh
export PYTHONPATH="$PWD/src:$PWD/src/IsaacLab/source/isaaclab:$PWD/src/IsaacLab/source/isaaclab_assets:$PYTHONPATH"
/isaac-sim/python.sh src/caster/sim_env/sim_server.py --headless --camera-config src/caster/config/camera/replay.yaml
```

Replay the trajectory:

```bash
uv run python src/caster/optimization/replay.py scene_num=1 task_name=pour cf_index=1 'demo_indices=[1,2]'
```

Replay converts tracked object motion into rigidly attached TCP position and orientation commands. The results are saved under `cf_<index>_<task>/optimization/replay/demo_<index>_seed_<seed>/`.


## Run optimization

Inside the prepared Isaac Sim container, start the simulation server with batched
environments:

```bash
/isaac-sim/python.sh src/caster/sim_env/sim_server.py --headless --set num_envs=1050 --camera-config src/caster/config/camera/optimization.yaml
```

In another terminal, run either cost function through the same entry point. `feature` matches CASTER implementation while `trajectory` matches the "mimic" basline in paper :

```bash
uv run python src/caster/optimization/run_optimization.py scene_num=1 task_name=pour cf_index=1 'demo_indices=[1,2]' cost_function=[feature/trajectory]
```

Settings live in [config/optimization/optimization.yaml](src/caster/config/optimization/optimization.yaml).
Outputs are saved under
`cf_<index>_<task>/optimization/<feature_cost_function|trajectory_cost_function>/demo_<index>_seed_<seed>/`,
including `optimized_trajectory.json`, `optimized_trajectory.npz`,
`optimized_sim_rollout.mp4`, and the final feasible-candidate archive. 

<!-- the original spline fitting, geometric L-BFGS-B stage, 1024-candidate CEM
refinement over 10 generations, checkpoint selection, and final validation.
Each frame uses one 60 Hz control step, followed by the configured release check. -->

## Real-world execution

On the robot control computer, configure the network and camera settings in
[real/config/default.yaml](src/caster/real/config/default.yaml).

```bash
uv run python src/caster/real/replay_trajectory.py scene_num=1 task_name=pour cf_index=1 execute=true
```

The default `method=feature` uses feature-optimized rollouts. Set
`method=trajectory` for trajectory optimization or `method=replay` for simulation
replay. The script allows selecting demos interactively; if a demo has multiple seeds, use
its directory name, such as `demo=demo_1_seed_0`.

[DA3]: https://github.com/ByteDance-Seed/Depth-Anything-3
[DEVA]: https://github.com/hkchengrex/Tracking-Anything-with-DEVA
[TrackCraft3R]: https://github.com/cvlab-kaist/TrackCraft3r

