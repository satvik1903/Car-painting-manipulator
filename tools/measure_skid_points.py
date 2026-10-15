"""
measure_skid_points.py
Finds how high the car body's underside is above each candidate skid support
point, by shooting a ray straight UP from below the car and seeing where it
first hits the body mesh.

Uses the processed visual mesh (metres, bottom of body at z = 0), so every
height printed here must be LIFTED by the skid height (0.337) to get the
real height above the booth floor.

Run (inside the tools venv):
    cd ~/Yuktii_Assesment/tools
    source .venv/bin/activate
    python3 measure_skid_points.py
"""
from pathlib import Path
import numpy as np
import open3d as o3d

SKID_LIFT = 0.337   # body bottom sits this high above the floor in the world

MESH = Path(__file__).resolve().parent.parent / "src/yuktiikuka/meshes/car/car_visual.stl"

# Candidate (x, y) support points to test, in the car frame.
#   current  : where the posts are now (inside the wheel arches)
#   rocker   : under the sills between the arches, near the body's outer edge
CANDIDATES = {
    "current front-left ": ( 1.45,  0.55),
    "current front-right": ( 1.45, -0.55),
    "current rear-left  ": (-1.45,  0.55),
    "current rear-right ": (-1.45, -0.55),
    "rocker  front-left ": ( 0.90,  0.85),
    "rocker  front-right": ( 0.90, -0.85),
    "rocker  rear-left  ": (-0.90,  0.85),
    "rocker  rear-right ": (-0.90, -0.85),
}

mesh = o3d.io.read_triangle_mesh(str(MESH))
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

# One ray per point: start 1 m below the car, point straight up (+z).
rays = np.array([[x, y, -1.0, 0.0, 0.0, 1.0] for (x, y) in CANDIDATES.values()],
                dtype=np.float32)
hits = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()   # distance travelled

print(f"{'point':<22}{'x':>6}{'y':>7}   underside above floor")
for (name, (x, y)), t in zip(CANDIDATES.items(), hits):
    if np.isinf(t):
        print(f"{name:<22}{x:>6.2f}{y:>7.2f}   NO SURFACE (ray went through a gap)")
    else:
        z_body = -1.0 + t                    # height in the car frame
        print(f"{name:<22}{x:>6.2f}{y:>7.2f}   {z_body + SKID_LIFT:.3f} m")

# Find the lowest point of the skin in a thin slice at each support x, per side.
v = np.asarray(mesh.vertices)
print(f"\n{'support point':<22}{'x':>6}{'y':>8}   post top height above floor")
for x0 in (0.9, -0.9):
    for side, name in ((1, "left "), (-1, "right")):
        sel = v[(np.abs(v[:, 0] - x0) < 0.05) & (np.sign(v[:, 1]) == side)]
        low = sel[np.argmin(sel[:, 2])]          # lowest skin point in that slice
        print(f"{'sill ' + ('front ' if x0 > 0 else 'rear  ') + name:<22}"
              f"{low[0]:>6.2f}{low[1]:>8.3f}   {low[2] + SKID_LIFT:.3f} m")