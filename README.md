# Robotic Car Painting Cell Simulation

**Name:** Satvik Tajane
**Software used:** ROS 2 Jazzy, Gazebo Harmonic, MoveIt 2 (OMPL, Cartesian planning, pick_ik), Python, Open3D
**Robot selected:** KUKA KR 20 R3100 (KR IONTEC), mounted upside down on an overhead XY gantry

Yuktii AI Labs, Robotics Simulation Intern Assessment.

A single industrial robot paints the exterior of a 4.8 x 1.9 x 1.6 m car body standing on a skid in a paint booth.
Paint paths are generated offline from a scanned car body; MoveIt 2 executes them as collision-checked straight-line
strokes at a constant 0.5 m/s gun speed, with paint coverage drawn live in Gazebo and RViz.

**Result:** 67 of 83 generated stripes painted (81 %): roof 10/11, hood 9/12, sides 48/60.

---

## Repository layout

```
src/
  yuktiikuka/                  robot, cell and painting package
    urdf/                      KR 20 R3100 macro, gantry macro, spray gun and TCP, ros2_control
    worlds/paint_booth.sdf     Gazebo world: booth, pillars, glass walls, car on skid, conveyor
    meshes/                    KR 20 R3100 meshes, car body (visual 100k and collision 5k triangles)
    config/cell_config.yaml    single source of truth: car pose, paint parameters, section rules, obstacles
    config/strokes.yaml        generated paint paths (output of tools/path_generator.py)
    config/kr20_controllers.yaml
    launch/gazebo.launch.py    Gazebo, robot, controllers
    scripts/planning_scene_setup.py   loads the car and booth into MoveIt's planning scene
    scripts/paint_executor.py         runs the painting process
  yuktiikuka_moveit_config/    MoveIt 2 configuration (groups arm, gantry, gantry_arm)
    launch/moveitgazebo.launch.py     move_group + RViz + planning scene, against the running Gazebo
tools/
  prepare_car.py               scale and decimate the scanned car body (Open3D)
  measure_profile.py           centreline height profile, used to set section boundaries
  measure_skid_points.py       skid support points on the sill edges
  path_generator.py            offline paint path generation, writes strokes.yaml
  reach_check.py, cellcheck.py early reach and coverage studies for the cell layout
  requirements.txt             Python packages for the tools
docs/
  DESIGN.md                    requirements, architecture, decisions, failure policy
  EXPERIMENTS.md               every painting experiment, configuration and finding
  images/                      figures
```

## Requirements

Ubuntu 24.04 with ROS 2 Jazzy, plus:

```bash
sudo apt install ros-jazzy-moveit ros-jazzy-ros-gz ros-jazzy-gz-ros2-control \
                 ros-jazzy-ros2-controllers ros-jazzy-pick-ik ros-jazzy-ros2controlcli
```

## Build

```bash
cd ~/Yuktii_Assesment
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Run

Three terminals, each after `source install/setup.bash`:

```bash
# 1. Gazebo: booth, car, gantry and robot with controllers
ros2 launch yuktiikuka gazebo.launch.py

# 2. MoveIt + RViz + planning scene (after the three controllers are active)
QT_QPA_PLATFORM=xcb ros2 launch yuktiikuka_moveit_config moveitgazebo.launch.py

# 3. Paint the whole car, or one section
ros2 run yuktiikuka paint_executor.py
ros2 run yuktiikuka paint_executor.py --ros-args -p section:=roof
```

In RViz, add MarkerArray displays on `/paint_plan` (planned stripes) and `/paint_coverage` (painted areas) to watch
the process. `QT_QPA_PLATFORM=xcb` avoids an RViz window error under Wayland with NVIDIA drivers.

Regenerating the paint paths (optional, the generated `strokes.yaml` is included):

```bash
cd tools && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 path_generator.py              # all sections; opens a preview window
```

## How it works

1. **Cell.** An 8 x 5 m booth between entry and exit zones. A gantry runs on rails carried by the booth's four pillars
   (x +-3.4 m, y +-1.5 m); the KR 20 R3100 hangs from its carriage with a 0.30 m spray gun. The tool centre point sits
   25 cm in front of the nozzle, where the paint lands.
2. **Paths (offline).** The car mesh is ray cast along zigzag stripe lines 10 cm apart (20 cm spray band, 50 % overlap).
   Each point gets a gun orientation pointing into the surface; stripes split at sharp turns and edges; approach and
   retreat points sit off the panel so the gun is at full speed over the paint.
3. **Execution (online).** For each section: arm to a tucked transit pose, gantry to a station chosen from the robot's
   kinematics, then each stripe as one collision-checked Cartesian path, retimed for a constant 0.5 m/s at the gun.
   Roof and hood are painted with the gantry locked; the sides use line tracking (the gantry moves with the gun).
4. **Collision checking.** MoveIt checks every motion against the car's collision mesh, the skid, floor, pillars and
   walls, and the robot's own links.
5. **Failure policy.** A stripe is painted fully or not at all. Planning failures are skipped and reported; failures
   during motion stop the robot. Every unpainted stripe is listed at the end of a run.


## Credits

- KUKA KR 20 R3100 description and meshes adapted from
  [kroshu/kuka_robot_descriptions](https://github.com/kroshu/kuka_robot_descriptions), Apache License 2.0
  (see `src/yuktiikuka/LICENSE`).
- Car body: "Car body" scan by [Artec 3D](https://www.artec3d.com/3d-models/car-body), licensed CC BY 4.0;
  scaled to 4.8 x 1.9 m and decimated.
