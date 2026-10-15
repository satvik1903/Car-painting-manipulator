"""
prepare_car.py
Turns the raw Artec 3D scan (Mercedes E350 body, millimetres, ~2.1M triangles)
into two meshes for the painting cell:

  car_visual.stl     ~100k triangles  -> drawn in Gazebo/RViz, used for paint paths
  car_collision.stl  ~5k triangles    -> used by MoveIt for fast collision checks

Target size (from the Yuktii brief): 4.8 m long, 1.9 m wide.
Height keeps the body's proportions; the skid lifts it to 1.6 m overall.

Frame after processing: car centred in x and y, bottom of the body at z = 0,
x along the length, z up (same as the booth frame).

Run (inside the tools venv):
    cd ~/Yuktii_Assesment/tools
    source .venv/bin/activate
    python3 prepare_car.py
Model credit: "Car body" by Artec 3D (https://www.artec3d.com/), CC BY 4.0.
"""
from pathlib import Path
import numpy as np
import open3d as o3d

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
TARGET_LENGTH = 4.8          # m, along x
TARGET_WIDTH = 1.9           # m, along y
OVERALL_HEIGHT = 1.6         # m, body + skid (used only to report the skid height)
VISUAL_TRIANGLES = 100_000
COLLISION_TRIANGLES = 5_000
SHOW_RESULT = True           # open a viewer at the end

# Paths are worked out from where this script lives, so it runs from anywhere.
TOOLS_DIR = Path(__file__).resolve().parent
WS_DIR = TOOLS_DIR.parent
RAW_DIR = WS_DIR / "models_raw"
OUT_DIR = WS_DIR / "src" / "yuktiikuka" / "meshes" / "car"


def size_of(mesh):
    """Length, width, height of a mesh's bounding box."""
    return mesh.get_axis_aligned_bounding_box().get_extent()


def clean(mesh):
    """Remove broken bits a scan often has, so decimation behaves."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def main():
    # 1. Find and load the raw OBJ ------------------------------------------------
    obj_files = sorted(RAW_DIR.rglob("*.OBJ"))
    if not obj_files:
        raise SystemExit(f"No .obj found under {RAW_DIR}")
    raw_path = obj_files[0]
    print(f"Loading {raw_path} ...")
    mesh = o3d.io.read_triangle_mesh(str(raw_path))
    print(f"  raw: {len(mesh.triangles):,} triangles, size (mm) {np.round(size_of(mesh), 1)}")

    # 2. Scale: millimetres -> metres, then fit the brief --------------------------
    v = np.asarray(mesh.vertices) * 0.001                    # now in metres
    raw_size_m = v.max(axis=0) - v.min(axis=0)
    sx = TARGET_LENGTH / raw_size_m[0]
    sy = TARGET_WIDTH / raw_size_m[1]
    sz = sx                                                  # height keeps the length's proportions
    v = v * np.array([sx, sy, sz])                           # broadcasting: one factor per axis
    print(f"  scale factors x {sx:.4f}  y {sy:.4f}  z {sz:.4f}")

    # 3. Position: centred in x and y, bottom of the body on z = 0 -----------------
    centre_xy = (v.max(axis=0) + v.min(axis=0)) / 2
    v[:, 0] -= centre_xy[0]
    v[:, 1] -= centre_xy[1]
    v[:, 2] -= v[:, 2].min()                                 # lift until the lowest point touches z = 0

    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh = clean(mesh)
    print(f"  scaled and placed: size (m) {np.round(size_of(mesh), 3)}")

    # 4. Visual mesh (~100k triangles) ------------------------------------------
    print(f"Decimating to {VISUAL_TRIANGLES:,} triangles (can take a minute) ...")
    visual = mesh.simplify_quadric_decimation(target_number_of_triangles=VISUAL_TRIANGLES)
    visual = clean(visual)

    # 5. Collision mesh (~5k triangles), made from the visual one (much faster) ----
    print(f"Decimating to {COLLISION_TRIANGLES:,} triangles ...")
    collision = visual.simplify_quadric_decimation(target_number_of_triangles=COLLISION_TRIANGLES)
    collision = clean(collision)

    # 6. Normals: STL files store one normal per triangle; vertex normals make the viewer shade nicely
    for m in (visual, collision):
        m.compute_triangle_normals()
        m.compute_vertex_normals()

    # 7. Save --------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    visual_path = OUT_DIR / "car_visual.stl"
    collision_path = OUT_DIR / "car_collision.stl"
    o3d.io.write_triangle_mesh(str(visual_path), visual)
    o3d.io.write_triangle_mesh(str(collision_path), collision)

    # 8. Report ------------------------------------------------------------------
    body_h = size_of(visual)[2]
    print("\nDone.")
    print(f"  {visual_path.name:<18} {len(visual.triangles):>8,} triangles  "
          f"size {np.round(size_of(visual), 3)}  {visual_path.stat().st_size / 1e6:.1f} MB")
    print(f"  {collision_path.name:<18} {len(collision.triangles):>8,} triangles  "
          f"size {np.round(size_of(collision), 3)}  {collision_path.stat().st_size / 1e6:.1f} MB")
    print(f"  body height {body_h:.3f} m  ->  skid height for {OVERALL_HEIGHT} m overall: "
          f"{OVERALL_HEIGHT - body_h:.3f} m")
    print(f"  saved to {OUT_DIR}")

    if SHOW_RESULT:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)   # 1 m arrows at the origin
        collision_view = o3d.geometry.LineSet.create_from_triangle_mesh(collision)
        collision_view.translate((0, 2.5, 0))                                 # show it beside the visual one
        collision_view.paint_uniform_color((0.8, 0.2, 0.2))
        visual.paint_uniform_color((0.75, 0.75, 0.75))
        o3d.visualization.draw_geometries(
            [visual, collision_view, frame],
            window_name="Left: visual mesh   Right (red wireframe): collision mesh",
        )


if __name__ == "__main__":
    main()