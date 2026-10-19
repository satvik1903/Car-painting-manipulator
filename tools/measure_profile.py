"""
measure_profile.py
Walks along the car's centreline (y = 0) from the front bumper to the rear,
shooting a ray straight DOWN every 2 cm, and records the surface height.
Then it computes the slope (how fast the height changes) and marks the places
where the surface turns from flat to steep or back: hood -> windshield,
windshield -> roof, roof -> rear window, and so on.

Heights are reported above the booth floor (car mesh + skid lift).
Also saves profile.png, a plot for the technical report.

Run (inside the tools venv):
    cd ~/Yuktii_Assesment/tools
    source .venv/bin/activate
    pip install matplotlib        # once
    python3 measure_profile.py
"""
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")              # write a file, no window needed
import matplotlib.pyplot as plt

SKID_LIFT = 0.337                  # body bottom height in the booth (cell_config.yaml)
STEP = 0.02                        # metres between rays
STEEP = 0.30                       # |slope| above this counts as "steep" (about 17 degrees)
Y_LINE = 0.0                       # centreline

MESH = Path(__file__).resolve().parent.parent / "src/yuktiikuka/meshes/car/car_visual.stl"

# ---- Build the ray-casting scene ---------------------------------------------
mesh = o3d.io.read_triangle_mesh(str(MESH))
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

# ---- One downward ray per x, from 3 m above the car's bottom -------------------
xs = np.arange(2.45, -2.45, -STEP)                     # front (+x) to rear (-x)
rays = np.array([[x, Y_LINE, 3.0, 0, 0, -1] for x in xs], dtype=np.float32)
t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
hit = np.isfinite(t)
z = np.where(hit, 3.0 - t + SKID_LIFT, np.nan)         # height above the booth floor

# ---- Slope: change in height per metre along the car ---------------------------
# Moving toward the rear, so a POSITIVE slope means the surface is climbing.
slope = np.full_like(z, np.nan)
slope[1:] = (z[1:] - z[:-1]) / STEP

# ---- Classify each point and find where the class changes ----------------------
def label(s):
    if np.isnan(s):
        return "none"
    if s > STEEP:
        return "climbing"
    if s < -STEEP:
        return "dropping"
    return "flat"

labels = [label(s) for s in slope]
print(f"{'x (m)':>7} {'height (m)':>11}   change")
prev = None
for x, zz, lab in zip(xs, z, labels):
    if lab != prev and lab != "none":
        print(f"{x:7.2f} {zz:11.3f}   -> {lab}")
        prev = lab

# ---- A coarse table every 20 cm, to read the shape --------------------------
print("\nprofile every 20 cm:")
for x, zz, s in zip(xs[::10], z[::10], slope[::10]):
    zs = "   -   " if np.isnan(zz) else f"{zz:6.3f}"
    ss = "   -  " if np.isnan(s) else f"{s:6.2f}"
    print(f"  x {x:6.2f}   z {zs}   slope {ss}")

# ---- Plot for the report ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(xs, z, lw=2)
ax.invert_xaxis()                                      # front of the car on the left
ax.set_xlabel("x along the car (m), front on the left")
ax.set_ylabel("surface height (m)")
ax.set_title("Car centreline profile (ray cast every 2 cm)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parent / "profile.png", dpi=150)
print("\nsaved tools/profile.png")
