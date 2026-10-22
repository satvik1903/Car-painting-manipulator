"""
path_generator.py  (offline half of the architecture, DESIGN.md section 4)

Generates zigzag paint stripes for every section in cell_config.yaml
(section_order + section_rules): points ON the car surface (= paint_tcp targets)
with the gun pointing straight into the surface, written to strokes.yaml.

v2: all sections in one file; up-facing and side-facing (+y / -y) surfaces;
stripes along x or y; fixed spin reference per section (no 180 deg wrist spin
at zigzag turnarounds); gantry station per section.

Run (inside the tools venv):
    cd ~/Yuktii_Assesment/tools && source .venv/bin/activate
    python3 path_generator.py              # all sections in section_order
    python3 path_generator.py hood roof    # only these
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

SAMPLE_STEP = 0.05     # m between points along a stripe
SCAN_MARGIN = 0.30     # m past the car's bounding box when a stripe has no range on its axis
MIN_FACING = 0.7       # surface must face the section's direction within ~45 deg
SMOOTH = 5             # average normals over this many neighbours (scan noise)
TURNAROUND_S = 1.0     # rough time per turnaround, for the estimate
MAX_TURN_DEG = 12.0    # surface direction may change at most this much between neighbours (5 cm)
MAX_GAP = 2.0          # a jump bigger than this many sample steps between neighbours splits the stripe
MIN_PIECE = 0.25       # m: pieces shorter than this are dropped

AXIS = {"x": 0, "y": 1, "z": 2}
FACING = {"up": np.array([0.0, 0.0, 1.0]),
          "+y": np.array([0.0, 1.0, 0.0]),
          "-y": np.array([0.0, -1.0, 0.0])}


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def quat_from_matrix(R):
    """3x3 rotation matrix -> quaternion [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def gun_orientation(normal, spin_ref):
    """Gun z axis points INTO the surface (z = -normal). Spin about z is free
    (round spray), so x is aligned with ONE fixed reference direction for the
    whole section. Same spin on every stripe = no wrist spin at turnarounds."""
    z = -normal / np.linalg.norm(normal)
    x = spin_ref - np.dot(spin_ref, z) * z
    x /= np.linalg.norm(x)
    return quat_from_matrix(np.column_stack([x, np.cross(z, x), z]))


def smooth_normals(normals, k):
    if len(normals) < k:
        return normals
    pad = k // 2
    padded = np.pad(normals, ((pad, pad), (0, 0)), mode="edge")
    out = np.array([padded[i:i + k].mean(axis=0) for i in range(len(normals))])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def longest_true_run(mask):
    best, start = (0, 0), None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def continuous_pieces(ok, hits, normals, step):
    """Split one stripe line into pieces the gun can follow smoothly.

    A piece ends where the ray misses or hits the wrong kind of surface (ok is
    False), where the surface direction turns more than MAX_TURN_DEG between
    neighbouring points (rails, mirrors, creases), or where the hit point jumps
    (an edge). Each piece becomes its own stripe: gun off between pieces, so a
    stripe is still painted fully or not at all.
    """
    pieces, start = [], None
    cos_lim = math.cos(math.radians(MAX_TURN_DEG))
    for i in range(len(ok)):
        if not ok[i]:
            if start is not None:
                pieces.append((start, i))
                start = None
            continue
        if start is None:
            start = i
            continue
        turn = float(np.dot(normals[i], normals[i - 1]))
        jump = float(np.linalg.norm(hits[i] - hits[i - 1]))
        if turn < cos_lim or jump > MAX_GAP * step:
            pieces.append((start, i))
            start = i
    if start is not None:
        pieces.append((start, len(ok)))
    return [(a, b) for a, b in pieces
            if b - a >= 3 and np.linalg.norm(hits[b - 1] - hits[a]) >= MIN_PIECE]


def spaced(a, b, spacing):
    """Evenly spread values from a to b, never more than `spacing` apart
    (round the gap count UP, +1 for the end: no bare strip)."""
    n = math.ceil(abs(b - a) / spacing) + 1
    return np.linspace(a, b, n)


# -----------------------------------------------------------------------------
# One section
# -----------------------------------------------------------------------------
def generate_section(name, rule, scene, bb, paint):
    facing = FACING[rule["facing"]]
    s_ax = AXIS[rule["stripe_axis"]]                     # axis the stripes run along
    rng = rule.get("range", {})
    spacing = paint["spray_width"] * (1.0 - paint["overlap"])     # R6, D2
    lead = paint["approach_distance"]
    lo, hi = bb.min_bound, bb.max_bound

    def limits(axis_name):
        a = AXIS[axis_name]
        return rng.get(axis_name, [lo[a] - SCAN_MARGIN, hi[a] + SCAN_MARGIN])

    if rule["facing"] == "up":
        # rays straight down; stripes along s_ax, offsets along the other horizontal axis
        o_name = "y" if rule["stripe_axis"] == "x" else "x"
        direction = np.array([0.0, 0.0, -1.0])
        ray_ax, start_coord = 2, hi[2] + 1.0
        if o_name in rng:
            offsets = spaced(*rng[o_name], spacing)
        else:
            offsets = np.arange(lo[AXIS[o_name]], hi[AXIS[o_name]], spacing)
        offsets = np.sort(offsets)[::-1]                 # start from the +x / +y end
    else:
        # rays horizontally toward the car; stripes along x, offsets in z, TOP first (R8)
        o_name = "z"
        direction = -facing
        ray_ax = 1
        start_coord = (lo[1] - 1.0) if rule["facing"] == "-y" else (hi[1] + 1.0)
        z0, z1 = limits("z")
        offsets = spaced(max(z0, z1), min(z0, z1), spacing)
    o_ax = AXIS[o_name]

    a0, a1 = limits(rule["stripe_axis"])
    samples = np.arange(min(a0, a1), max(a0, a1) + 1e-9, SAMPLE_STEP)
    spin_ref = np.zeros(3)
    spin_ref[s_ax] = 1.0                                 # fixed for the whole section

    stripes, painted, lines_done = [], 0.0, 0
    for off in offsets:
        origins = np.zeros((len(samples), 3))
        origins[:, s_ax] = samples
        origins[:, o_ax] = off
        origins[:, ray_ax] = start_coord
        rays = np.hstack([origins, np.tile(direction, (len(samples), 1))]).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()
        normals = ans["primitive_normals"].numpy().astype(float)
        normals[(normals @ direction) > 0] *= -1         # make every normal face the ray
        hits = origins + np.where(np.isfinite(t), t, 0)[:, None] * direction

        ok = np.isfinite(t) & ((normals @ facing) > MIN_FACING)
        for ax_name, (r0, r1) in rng.items():           # stay inside the section's box
            v = hits[:, AXIS[ax_name]]
            ok &= (v >= min(r0, r1)) & (v <= max(r0, r1))
        smooth = np.zeros_like(normals)
        good = np.where(ok)[0]
        if len(good):
            smooth[good] = smooth_normals(normals[good], SMOOTH)
        pieces = continuous_pieces(ok, hits, smooth, SAMPLE_STEP)
        line_reversed = lines_done % 2 == 1               # zigzag (D1), per stripe LINE
        if line_reversed:
            pieces = pieces[::-1]
        for a, b in pieces:
            pts, nrm = hits[a:b], smooth[a:b]
            if line_reversed:
                pts, nrm = pts[::-1], nrm[::-1]
            travel = pts[-1] - pts[0]
            travel /= np.linalg.norm(travel)
            quats = [gun_orientation(n, spin_ref) for n in nrm]
            points = [[*map(float, p), *map(float, q)] for p, q in zip(pts, quats)]
            approach = [*map(float, pts[0] - lead * travel), *map(float, quats[0])]
            retreat = [*map(float, pts[-1] + lead * travel), *map(float, quats[-1])]
            length = float(np.linalg.norm(pts[-1] - pts[0]))
            painted += length
            stripes.append({"id": len(stripes), "line": int(lines_done), "length": round(length, 3),
                            "approach": approach, "points": points, "retreat": retreat})
        if pieces:
            lines_done += 1

    est = (painted + 2 * lead * len(stripes)) / paint["speed"] + TURNAROUND_S * len(stripes)
    print(f"  {name:<12} {lines_done:>3} lines {len(stripes):>3} stripes   painted {painted:6.2f} m   ~{est:4.0f} s")
    entry = {"name": name, "facing": rule["facing"], "stripe_axis": rule["stripe_axis"],
             "stripe_spacing": round(spacing, 3), "stripes": stripes}
    if "station" in rule:
        entry["station"] = [float(v) for v in rule["station"]]
    if "stroke_group" in rule:
        entry["stroke_group"] = rule["stroke_group"]
    return entry, est


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    cfg = yaml.safe_load(open(CONFIG))
    rules = cfg["section_rules"]
    wanted = sys.argv[1:] or [s for s in cfg["section_order"] if s in rules]

    mesh = o3d.io.read_triangle_mesh(str(PKG / cfg["car"]["visual_mesh"]))
    mesh.translate(cfg["car"]["position"])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    bb = mesh.get_axis_aligned_bounding_box()

    print("generating:")
    sections, total = [], 0.0
    for name in wanted:
        entry, est = generate_section(name, rules[name], scene, bb, cfg["paint"])
        sections.append(entry)
        total += est
    print(f"  total painting estimate ~{total / 60:.1f} min (plus gantry moves)")

    out = {"frame": cfg["frame"],
           "pose_format": "[x, y, z, qx, qy, qz, qw] of paint_tcp; z axis points into the surface",
           "sections": sections}
    with open(OUTPUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None)
    print(f"written to {OUTPUT}")

    # Preview: car, stripes (blue/orange alternating), gun lines every 6th point
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.75, 0.75, 0.78))
    geoms = [mesh]
    standoff = cfg["paint"]["standoff"]
    for s in sections:
        for st in s["stripes"]:
            p = np.array([pt[:3] for pt in st["points"]])
            line = o3d.geometry.LineSet(o3d.utility.Vector3dVector(p),
                                        o3d.utility.Vector2iVector([[k, k + 1] for k in range(len(p) - 1)]))
            line.paint_uniform_color((0.1, 0.4, 0.9) if st["id"] % 2 == 0 else (0.95, 0.5, 0.1))
            geoms.append(line)
            for pt in st["points"][::6]:
                q = pt[3:]
                Rm = o3d.geometry.get_rotation_matrix_from_quaternion([q[3], q[0], q[1], q[2]])
                pos = np.array(pt[:3])
                seg = o3d.geometry.LineSet(o3d.utility.Vector3dVector([pos, pos - standoff * Rm[:, 2]]),
                                           o3d.utility.Vector2iVector([[0, 1]]))
                seg.paint_uniform_color((0.1, 0.7, 0.2))
                geoms.append(seg)
    o3d.visualization.draw_geometries(geoms, window_name="Paint stripes: " + ", ".join(wanted))


if __name__ == "__main__":
    main()