#!/usr/bin/env python3
"""
planning_scene_setup.py
Adds the car body and the booth's static obstacles to MoveIt's planning scene,
so every plan avoids them (requirement R2).

Everything comes from cell_config.yaml (the single source of truth), so the
planning scene and the Gazebo world are described by the same numbers.

How it works:
  1. Read cell_config.yaml.
  2. Build one CollisionObject per obstacle (boxes) plus the car (a mesh).
  3. Send them all to move_group in ONE request on /apply_planning_scene,
     then exit. move_group keeps them for as long as it runs.
"""
import os
import struct

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive

PKG = "yuktiikuka"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_binary_stl(path):
    """Read a binary STL into (vertices N x 3, triangles M x 3).

    Binary STL layout: 80-byte header, uint32 triangle count, then one
    50-byte record per triangle: normal (3 floats), 3 vertices (9 floats),
    and a 2-byte attribute. Open3D writes this format by default.
    """
    with open(path, "rb") as f:
        data = f.read()
    count = struct.unpack("<I", data[80:84])[0]
    if 84 + count * 50 != len(data):
        raise ValueError(f"{path} is not a binary STL (ASCII STL is not supported here)")
    record = np.dtype([("normal", "<f4", 3), ("v", "<f4", 9), ("attr", "<u2")])
    tris = np.frombuffer(data, dtype=record, count=count, offset=84)
    vertices = tris["v"].reshape(-1, 3)                 # 3 vertices per triangle, not shared
    triangles = np.arange(len(vertices)).reshape(-1, 3)
    return vertices, triangles


def make_pose(xyz):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)
    pose.orientation.w = 1.0                           # no rotation
    return pose


def box_object(name, size, xyz, frame):
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = name
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [float(s) for s in size]           # full lengths x, y, z
    obj.primitives = [box]
    obj.primitive_poses = [make_pose(xyz)]
    obj.operation = CollisionObject.ADD
    return obj


def mesh_object(name, stl_path, xyz, frame):
    vertices, triangles = load_binary_stl(stl_path)
    mesh = Mesh()
    mesh.vertices = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in vertices]
    mesh.triangles = [MeshTriangle(vertex_indices=[int(a), int(b), int(c)]) for a, b, c in triangles]
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = name
    obj.meshes = [mesh]
    obj.mesh_poses = [make_pose(xyz)]
    obj.operation = CollisionObject.ADD
    return obj, len(triangles)


# -----------------------------------------------------------------------------
# Node
# -----------------------------------------------------------------------------
class PlanningSceneSetup(Node):
    def __init__(self):
        super().__init__("planning_scene_setup")
        share = get_package_share_directory(PKG)
        default_cfg = os.path.join(share, "config", "cell_config.yaml")
        cfg_path = self.declare_parameter("config_file", default_cfg).value
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.share = share
        self.client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")

    def build_objects(self):
        frame = self.cfg["frame"]
        objects = []

        car = self.cfg["car"]
        stl = os.path.join(self.share, car["collision_mesh"])
        car_obj, n_tris = mesh_object("car_body", stl, car["position"], frame)
        objects.append(car_obj)
        self.get_logger().info(f"car_body: mesh with {n_tris} triangles at {car['position']}")

        for ob in self.cfg["obstacles"]:
            objects.append(box_object(ob["name"], ob["size"], ob["position"], frame))
        self.get_logger().info(f"{len(self.cfg['obstacles'])} box obstacles")
        return objects

    def run(self):
        # move_group must be up first: wait for its service.
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("waiting for /apply_planning_scene (is move_group running?)")

        scene = PlanningScene()
        scene.is_diff = True                            # ADD to the scene, don't replace it
        scene.world.collision_objects = self.build_objects()

        future = self.client.call_async(ApplyPlanningScene.Request(scene=scene))
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is not None and future.result().success:
            self.get_logger().info("planning scene loaded: car, skid, floor, pillars, walls")
        else:
            self.get_logger().error("failed to apply the planning scene")


def main():
    rclpy.init()
    node = PlanningSceneSetup()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
