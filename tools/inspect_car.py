import open3d as o3d

mesh = o3d.io.read_triangle_mesh("../models_raw/Car_body.OBJ")
bb = mesh.get_axis_aligned_bounding_box()

print("triangles:", len(mesh.triangles), " vertices:", len(mesh.vertices))
print("min corner:", bb.min_bound)
print("max corner:", bb.max_bound)
print("size x y z:", bb.get_extent())
print("watertight (fully closed):", mesh.is_watertight())

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(bb.get_extent()) * 0.2)
o3d.visualization.draw_geometries([mesh, frame])