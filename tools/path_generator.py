"""
path_generator.py  (offline half of the architecture, DESIGN.md section 4)

Reads cell_config.yaml and the car mesh, and generates zigzag paint stripes for
one section: points ON the car surface (where paint lands = paint_tcp targets)
with the gun pointing straight into the surface at each point.

Output: src/yuktiikuka/config/strokes.yaml (the contract with the executor),
plus an Open3D preview window so the stripes can be checked by eye.

Run (inside the tools venv):
    cd ~/Yuktii_Assesment/tools
    source .venv/bin/activate
    python3 path_generator.py            # default section: hood
    python3 path_generator.py hood
"""
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

WS = Path(__file__).resolve().parent.parent
PKG = WS / "src" / "yuktiikuka"
CONFIG = PKG / "config" / "cell_config.yaml"
OUTPUT = PKG / "config" / "strokes.yaml"

SAMPLE_STEP = 0.05     # m between points along a stripe (small, so straight segments hug the curve: R3)
SCAN_MARGIN = 0.30     # m: scan this far past the car's sides to find where the surface ends
MIN_FACING = 0.7       # an "up" surface must have normal z > 0.5 (within 60 deg of vertical)
SMOOTH = 5             # average normals over this many neighbouring points (scan noise)
TURNAROUND_S = 1.0     # rough time per off-panel turnaround, for the time estimate


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def quat_from_matrix(R):
    """3x3 rotation matrix -> quaternion [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def gun_orientation(normal, travel_dir):
    """Orientation of paint_tcp at one point.

    The gun's z axis (tool0 / paint_tcp z) must point INTO the surface: z = -normal.
    Spin about z is free (round spray), so x is chosen along the travel direction:
    this keeps the spin steady along a stripe, which helps the IK stay smooth.
    """
    z = -normal / np.linalg.norm(normal)
    x = travel_dir - np.dot(travel_dir, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return quat_from_matrix(np.column_stack([x, y, z]))


def smooth_normals(normals, k):
    """Moving average along the stripe, then renormalise (removes scan noise)."""
    if len(normals) < k:
        return normals
    pad = k // 2
    padded = np.pad(normals, ((pad, pad), (0, 0)), mode="edge")
    out = np.array([padded[i:i + k].mean(axis=0) for i in range(len(normals))])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def longest_true_run(mask):
    """Indices [start, end) of the longest run of True values."""
    best, start = (0, 0), None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    section = sys.argv[1] if len(sys.argv) > 1 else "hood"
    cfg = yaml.safe_load(open(CONFIG))
    rule = cfg["section_rules"][section]
    paint = cfg["paint"]
    if rule["facing"] != "up" or rule["stripe_axis"] != "y":
        raise SystemExit("This first version handles up-facing sections with stripes along y")

    # Stripe spacing from the spray band and overlap (R6, D2): 0.20 * (1 - 0.5) = 0.10 m
    spacing = paint["spray_width"] * (1.0 - paint["overlap"])
    lead = paint["approach_distance"]

    # Car mesh, moved to where it sits in the booth (on the skid)
    mesh = o3d.io.read_triangle_mesh(str(PKG / cfg["car"]["visual_mesh"]))
    mesh.translate(cfg["car"]["position"])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    bb = mesh.get_axis_aligned_bounding_box()
    top = bb.max_bound[2] + 1.0                         # rays start 1 m above the car

    # Stripe positions along x: evenly spread, never more than `spacing` apart.
    # Round UP the number of gaps (never leave bare metal), then +1 for the end stripe.
    x0, x1 = rule["x_range"]
    n_stripes = math.ceil((x1 - x0) / spacing) + 1
    xs = np.linspace(x1, x0, n_stripes)                 # front edge first

    ys = np.arange(bb.min_bound[1] - SCAN_MARGIN, bb.max_bound[1] + SCAN_MARGIN, SAMPLE_STEP)

    stripes, total_paint_len = [], 0.0
    for i, x in enumerate(xs):
        # 1. Drop a ray straight down at every sample along this stripe line
        rays = np.array([[x, y, top, 0, 0, -1] for y in ys], dtype=np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()
        normals = ans["primitive_normals"].numpy().astype(float)

        # Scan triangles can be wound either way: make every normal face the ray (point up)
        normals[normals[:, 2] < 0] *= -1

        # 2. Keep points that hit the car on an up-facing surface; take the longest unbroken run
        ok = np.isfinite(t) & (normals[:, 2] > MIN_FACING)
        a, b = longest_true_run(ok)
        if b - a < 3:
            continue
        pts = np.column_stack([np.full(b - a, x), ys[a:b], top - t[a:b]])
        nrm = smooth_normals(normals[a:b], SMOOTH)

        # 3. Zigzag (D1): every other stripe runs the opposite way
        if i % 2 == 1:
            pts, nrm = pts[::-1], nrm[::-1]
        travel = pts[-1] - pts[0]
        travel /= np.linalg.norm(travel)

        # 4. Gun orientation at every point: pointing into the surface
        quats = [gun_orientation(n, travel) for n in nrm]
        points = [[*map(float, p), *map(float, q)] for p, q in zip(pts, quats)]

        # 5. Approach and retreat: off the panel, along the travel line, same height and angle
        approach = [*map(float, pts[0] - lead * travel), *map(float, quats[0])]
        retreat = [*map(float, pts[-1] + lead * travel), *map(float, quats[-1])]

        length = float(np.linalg.norm(pts[-1] - pts[0]))
        total_paint_len += length
        stripes.append({"id": len(stripes), "length": round(length, 3),
                        "approach": approach, "points": points, "retreat": retreat})

    # ---- Write the contract file ------------------------------------------------
    out = {
        "frame": cfg["frame"],
        "pose_format": "[x, y, z, qx, qy, qz, qw] of paint_tcp; z axis points into the surface",
        "sections": [{"name": section, "stripe_axis": rule["stripe_axis"],
                      "stripe_spacing": round(spacing, 3), "stripes": stripes}],
    }
    with open(OUTPUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)

    # ---- Report -----------------------------------------------------------------
    travel_len = total_paint_len + 2 * lead * len(stripes)
    est = travel_len / paint["speed"] + TURNAROUND_S * len(stripes)
    print(f"section {section}: {len(stripes)} stripes, spacing {spacing:.3f} m")
    print(f"  painted length {total_paint_len:.2f} m, estimated time {est:.0f} s")
    print(f"  written to {OUTPUT}")

    # ---- Preview: car, stripes (blue/orange alternating), gun directions -----------
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.75, 0.75, 0.78))
    geoms = [mesh]
    for s in stripes:
        p = np.array([pt[:3] for pt in s["points"]])
        line = o3d.geometry.LineSet(o3d.utility.Vector3dVector(p),
                                    o3d.utility.Vector2iVector([[k, k + 1] for k in range(len(p) - 1)]))
        line.paint_uniform_color((0.1, 0.4, 0.9) if s["id"] % 2 == 0 else (0.95, 0.5, 0.1))
        geoms.append(line)
        # every 6th point: a short line from the surface UP to where the gun nozzle is (25 cm)
        for pt in s["points"][::6]:
            pos = np.array(pt[:3])
            q = pt[3:]
            Rm = o3d.geometry.get_rotation_matrix_from_quaternion([q[3], q[0], q[1], q[2]])
            nozzle = pos - paint["standoff"] * Rm[:, 2]     # back along the gun axis
            seg = o3d.geometry.LineSet(o3d.utility.Vector3dVector([pos, nozzle]),
                                       o3d.utility.Vector2iVector([[0, 1]]))
            seg.paint_uniform_color((0.1, 0.7, 0.2))
            geoms.append(seg)
    o3d.visualization.draw_geometries(geoms, window_name=f"Paint stripes: {section}")


if __name__ == "__main__":
    main()
