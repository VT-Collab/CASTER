# Isaac simulation details

See the README for [container setup](../../../README.md#setup-instructions),
[simulation replay](../../../README.md#simulation-replay), and
[optimization commands and outputs](../../../README.md#run-optimization).

## Runtime and server options

Wait for `Simulation server ready` before starting a client, and keep the server
terminal running. Settings live in
[config/simulation.yaml](../config/simulation.yaml).
UMI fingers are the default; stock Franka fingers are available through
`--set gripper_variant=stock`.
The [replay camera config](../config/camera/replay.yaml) frames the table
for a single environment. Omitting `--camera-config` keeps the original camera
settings. Camera rotations use WXYZ quaternions.
