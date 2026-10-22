#!/usr/bin/env python3
"""
paint_executor.py  (online half of the architecture, DESIGN.md section 4)

Reads strokes.yaml (the plan) and cell_config.yaml (the settings) and paints
each section through MoveIt, following the process in DESIGN.md section 2:

  arm -> transit pose (D9)                      MoveGroup action, joint goal
  gantry -> section station, then held (D6)     MoveGroup action, joint goal
  arm -> first approach point                   MoveGroup action, pose goal (free move)
  for every stripe:
      straight line: [turnaround] approach -> points -> retreat
                                                /compute_cartesian_path (avoids collisions)
      quality check: whole line planned?        failure policy: skip + report
      retime the line for a constant 0.5 m/s at the gun (R5), using /compute_fk
      execute                                   /execute_trajectory action
      gun on/off + coverage markers             RViz (/paint_coverage) and Gazebo
  arm -> transit, gantry -> parked, arm -> home

Run (Gazebo and MoveIt already running):
    ros2 run yuktiikuka paint_executor.py
    ros2 run yuktiikuka paint_executor.py --ros-args -p section:=hood
In RViz: Add -> MarkerArray -> /paint_coverage (and /paint_plan).
"""
import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MotionPlanRequest,
                             OrientationConstraint, PositionConstraint, RobotState)
from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

PKG = "yuktiikuka"
ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]
GANTRY_JOINTS = ["gantry_x_joint", "gantry_y_joint"]
TCP = "paint_tcp"

CART_STEP = 0.02          # m between points in the Cartesian path (and FK samples)
MIN_SPEED_FRACTION = 0.2  # a stretch that must slow below 20 % of target speed = wrist flip: fail it
PAINT_COLOR = (0.05, 0.25, 0.75)
SETTLE_S = 0.3           # s to wait before planning, so the gantry has stopped creeping


class SafetyStop(Exception):
    """Failure policy: safety failures stop all motion, no automatic recovery."""


class PlanFailure(Exception):
    """Failure policy: a quality failure. No motion happened, so skip and carry on."""


# MoveIt error codes that mean the robot was MOVING when it went wrong (safety).
# Everything else (no plan, no IK, goal in collision, ...) happened before any motion.
EXECUTION_ERRORS = {-4: "CONTROL_FAILED", -6: "TIMED_OUT", -7: "PREEMPTED"}


def pose_from_list(p):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = p[0], p[1], p[2]
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = p[3:7]
    return pose


def secs(t):
    s = int(t)
    return Duration(sec=s, nanosec=int((t - s) * 1e9))


class PaintExecutor(Node):
    def __init__(self):
        super().__init__("paint_executor")
        share = get_package_share_directory(PKG)
        self.cfg = yaml.safe_load(open(os.path.join(share, "config", "cell_config.yaml")))
        self.strokes = yaml.safe_load(open(os.path.join(share, "config", "strokes.yaml")))
        self.frame = self.cfg["frame"]
        self.paint = self.cfg["paint"]
        self.section_param = self.declare_parameter("section", "all").value
        self.gazebo_markers = self.declare_parameter("gazebo_markers", True).value
        # Line tracking: strokes are planned for gantry + arm (8 joints), so the
        # gantry can slide along with the gun. Use "arm" for station-locked painting.
        self.stroke_group = self.declare_parameter("stroke_group", "gantry_arm").value
        self.active_group = self.stroke_group

        self.move_group = ActionClient(self, MoveGroup, "/move_action")
        self.execute = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.cartesian = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.fk = self.create_client(GetPositionFK, "/compute_fk")
        self.params = self.create_client(GetParameters, "/move_group/get_parameters")

        self.coverage_pub = self.create_publisher(MarkerArray, "/paint_coverage", 10)
        self.plan_pub = self.create_publisher(MarkerArray, "/paint_plan", 10)
        self.joint_state = None
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)

        self.coverage = MarkerArray()
        self.gz_id = 0
        self.report = []                       # (section, stripe, status)

    # ------------------------------------------------------------------ plumbing
    def _on_joints(self, msg):
        self.joint_state = msg

    def wait(self, future, timeout=60.0):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def wait_for_interfaces(self):
        for name, ok in [("/move_action", lambda: self.move_group.wait_for_server(2.0)),
                         ("/execute_trajectory", lambda: self.execute.wait_for_server(2.0)),
                         ("/compute_cartesian_path", lambda: self.cartesian.wait_for_service(2.0)),
                         ("/compute_fk", lambda: self.fk.wait_for_service(2.0))]:
            while not ok():
                self.get_logger().info(f"waiting for {name} (is move_group running?)")
        while self.joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.2)

    def named_poses(self):
        """Read home / transit / parked joint values from the SRDF (single source of truth)."""
        self.params.wait_for_service()
        res = self.wait(self.params.call_async(GetParameters.Request(names=["robot_description_semantic"])))
        srdf = ET.fromstring(res.values[0].string_value)
        poses = {}
        for gs in srdf.iter("group_state"):
            poses[gs.get("name")] = {j.get("name"): float(j.get("value")) for j in gs.iter("joint")}
        return poses

    # ------------------------------------------------------------------ motions
    def settle(self):
        """Let the robot come to rest so planning starts from where it really is.
        The heavy gantry creeps its last millimetres after a move in Gazebo; a plan
        started before that is rejected as 'start point deviates from current state'."""
        time.sleep(SETTLE_S)

    def _send_move(self, group, constraints, label):
        self.settle()
        req = MotionPlanRequest()
        req.group_name = group
        req.num_planning_attempts = 5
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.5
        req.max_acceleration_scaling_factor = 0.5
        req.start_state.is_diff = True
        req.goal_constraints = [constraints]
        goal = MoveGroup.Goal(request=req)
        goal.planning_options.plan_only = False
        handle = self.wait(self.move_group.send_goal_async(goal))
        if handle is None or not handle.accepted:
            raise SafetyStop(f"{label}: MoveGroup goal rejected")
        result = self.wait(handle.get_result_async(), timeout=120.0)
        code = result.result.error_code.val
        if code in EXECUTION_ERRORS:
            raise SafetyStop(f"{label}: failed during motion ({EXECUTION_ERRORS[code]})")
        if code != 1:                          # 1 = SUCCESS; no motion happened
            raise PlanFailure(f"{label}: no plan (MoveIt error code {code})")
        self.get_logger().info(f"{label}: done")

    def move_joints(self, group, values, label):
        c = Constraints()
        for name, pos in values.items():
            c.joint_constraints.append(JointConstraint(joint_name=name, position=pos,
                                                       tolerance_above=0.001, tolerance_below=0.001,
                                                       weight=1.0))
        self._send_move(group, c, label)

    def move_tcp_to(self, pose_list, label):
        target = pose_from_list(pose_list)
        pc = PositionConstraint()
        pc.header.frame_id = self.frame
        pc.link_name = TCP
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005])
        pc.constraint_region.primitives = [sphere]
        pc.constraint_region.primitive_poses = [target]
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = self.frame
        oc.link_name = TCP
        oc.orientation = target.orientation
        tol = math.radians(self.paint["tilt_tolerance_deg"])
        oc.absolute_x_axis_tolerance = tol
        oc.absolute_y_axis_tolerance = tol
        # Spin is free for the PAINT (A2), but the approach must match the stripe's spin,
        # or the first stretch of the straight line becomes a wrist spin in place.
        oc.absolute_z_axis_tolerance = tol
        oc.weight = 1.0
        goal = Constraints(position_constraints=[pc], orientation_constraints=[oc])
        try:
            self._send_move(self.active_group, goal, label)
        except PlanFailure:
            if self.active_group == "arm":
                raise
            # 8-joint free moves are a harder search; the arm alone often finds one
            self.get_logger().info(f"{label}: retrying with the arm group only")
            self._send_move("arm", goal, label + " (arm only)")

    # ------------------------------------------------------------------ one stroke
    def plan_line(self, waypoints):
        self.settle()
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.start_state.is_diff = True
        req.group_name = self.active_group
        req.link_name = TCP
        req.waypoints = [pose_from_list(p) for p in waypoints]
        req.max_step = CART_STEP
        req.jump_threshold = 0.0               # flips are caught in retime() instead
        req.avoid_collisions = True            # R2
        res = self.wait(self.cartesian.call_async(req))
        return res.solution.joint_trajectory, res.fraction

    def tcp_position(self, names, positions):
        req = GetPositionFK.Request()
        req.header.frame_id = self.frame
        req.fk_link_names = [TCP]
        req.robot_state = RobotState(joint_state=JointState(name=names, position=list(positions)),
                                     is_diff=True)
        res = self.wait(self.fk.call_async(req), timeout=5.0)
        p = res.pose_stamped[0].pose.position
        return np.array([p.x, p.y, p.z])

    def retime(self, traj, first_pt, last_pt):
        """R5: give every point a time so the GUN moves at a constant speed.

        MoveIt times paths by joint limits, which makes the gun speed up and slow
        down. Here: FK gives the gun position at every point; each stretch gets
        time = distance / 0.5 m/s, stretched only where a joint would exceed its
        limit. A stretch that must slow below 20 % of the target speed means the
        wrist is flipping: the stroke is rejected (quality failure).
        """
        speed = self.paint["speed"]
        # speed cap per joint for painting moves: gantry slides in m/s, arm joints in rad/s
        vmax = np.array([0.5 if n.startswith("gantry") else 1.5 for n in traj.joint_names])
        q = np.array([pt.positions for pt in traj.points])
        x = np.array([self.tcp_position(traj.joint_names, row) for row in q])
        # the painted part runs from the point nearest the first surface point to the
        # point nearest the last one; the lead-in and turnaround are off the panel
        i0 = int(np.argmin(np.linalg.norm(x - np.array(first_pt[:3]), axis=1)))
        i1 = int(np.argmin(np.linalg.norm(x - np.array(last_pt[:3]), axis=1)))
        times, slowest = [0.0], 1.0
        for i in range(1, len(q)):
            dist = float(np.linalg.norm(x[i] - x[i - 1]))
            t_gun = dist / speed
            t_joint = float(np.max(np.abs(q[i] - q[i - 1]) / vmax))
            dt = max(t_gun, t_joint, 1e-3)
            if i0 < i <= i1 and dist > 1e-4:
                slowest = min(slowest, t_gun / dt)
            times.append(times[-1] + dt)
        if slowest < MIN_SPEED_FRACTION:
            return None, slowest
        t = np.array(times)
        vel = np.zeros_like(q)                  # finite-difference velocities, zero at both ends
        vel[1:-1] = (q[2:] - q[:-2]) / (t[2:] - t[:-2])[:, None]
        for pt, ti, vi in zip(traj.points, t, vel):
            pt.time_from_start = secs(float(ti))
            pt.velocities = vi.tolist()
            pt.accelerations = []
        return traj, slowest

    def run_trajectory(self, traj, label):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory.joint_trajectory = traj
        handle = self.wait(self.execute.send_goal_async(goal))
        if handle is None or not handle.accepted:
            raise SafetyStop(f"{label}: execution rejected")
        result = self.wait(handle.get_result_async(), timeout=300.0)
        if result.result.error_code.val != 1:
            # failure policy: a controller error during motion is a SAFETY failure
            raise SafetyStop(f"{label}: execution failed (code {result.result.error_code.val})")

    # ------------------------------------------------------------------ visuals
    def disc_marker(self, mid, p, ns, color, alpha):
        m = Marker()
        m.header.frame_id = self.frame
        m.ns, m.id, m.type, m.action = ns, mid, Marker.CYLINDER, Marker.ADD
        # disc centred on the surface point, flat on it: its axis is the gun axis (= -normal)
        m.pose = pose_from_list(p)
        m.scale.x = m.scale.y = self.paint["spray_width"]   # the spray band (D2)
        m.scale.z = 0.003
        m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=alpha)
        return m

    def publish_plan(self, sections):
        arr = MarkerArray()
        for s in sections:
            for st in s["stripes"]:
                m = Marker()
                m.header.frame_id = self.frame
                m.ns, m.id, m.type, m.action = f"plan_{s['name']}", st["id"], Marker.LINE_STRIP, Marker.ADD
                m.pose.orientation.w = 1.0
                m.scale.x = 0.01
                m.color = ColorRGBA(r=0.95, g=0.6, b=0.1, a=0.9)
                m.points = [Point(x=p[0], y=p[1], z=p[2]) for p in st["points"]]
                arr.markers.append(m)
        self.plan_pub.publish(arr)

    def paint_stripe_markers(self, section, stripe):
        """Gun was ON along this stripe: add paint discs at every point."""
        new = []
        for k, p in enumerate(stripe["points"]):
            mid = stripe["id"] * 1000 + k
            new.append(self.disc_marker(mid, p, f"paint_{section}", PAINT_COLOR, 0.85))
        self.coverage.markers.extend(new)
        self.coverage_pub.publish(self.coverage)
        if self.gazebo_markers:
            self.gazebo_paint(stripe["points"])

    def gazebo_paint(self, points):
        """Draw the same paint discs in Gazebo's 3D view (visual only, no physics).
        Gazebo does not simulate paint, so the coverage is drawn as markers."""
        r, g, b = PAINT_COLOR
        items = []
        for p in points:
            self.gz_id += 1
            items.append(
                "marker { action: ADD_MODIFY ns: \"paint\" id: %d type: CYLINDER "
                "pose { position { x: %.4f y: %.4f z: %.4f } orientation { x: %.5f y: %.5f z: %.5f w: %.5f } } "
                "scale { x: %.3f y: %.3f z: 0.003 } "
                "material { ambient { r: %.2f g: %.2f b: %.2f a: 1 } diffuse { r: %.2f g: %.2f b: %.2f a: 1 } } }"
                % (self.gz_id, *p[:7], self.paint["spray_width"], self.paint["spray_width"], r, g, b, r, g, b))
        cmd = ["gz", "service", "-s", "/marker_array", "--reqtype", "gz.msgs.Marker_V",
               "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", " ".join(items)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
        except Exception as e:                  # visuals must never stop the painting
            self.get_logger().warn(f"Gazebo markers skipped: {e}")

    # ------------------------------------------------------------------ the process
    def paint_section(self, section, poses):
        name, stripes = section["name"], section["stripes"]
        if not stripes:
            self.get_logger().warn(f"{name}: no stripes generated, skipping")
            return
        # per-section choice (cell_config section_rules.<name>.stroke_group), else the node parameter
        self.active_group = section.get("stroke_group", self.stroke_group)
        if "station" in section:
            sx, sy = section["station"]
        else:
            pts = np.array([p[:3] for st in stripes for p in st["points"]])
            sx, sy = float(pts[:, 0].mean()), 0.0
        station = {"gantry_x_joint": sx, "gantry_y_joint": sy}
        self.get_logger().info(f"=== {name}: {len(stripes)} stripes, station ({sx:.2f}, {sy:.2f}), "
                               f"strokes planned for '{self.active_group}'")

        try:
            self.move_joints("arm", poses["transit"], f"{name}: arm to transit")         # D9
            self.move_joints("gantry", station, f"{name}: gantry to station")            # D6
            self.move_tcp_to(stripes[0]["approach"], f"{name}: approach first stripe")
        except PlanFailure as e:
            self.get_logger().warn(f"{e}: skipping section {name}")
            self.report.extend((name, st["id"], "FAILED") for st in stripes)
            return

        failed, t0 = 0, time.time()
        for i, st in enumerate(stripes):
            label = f"{name} stripe {st['id']}"
            waypoints = [st["approach"], *st["points"], st["retreat"]]
            traj, fraction = self.plan_line(waypoints)
            ok = fraction > 0.999
            if ok:
                traj, slowest = self.retime(traj, st["points"][0], st["points"][-1])
                ok = traj is not None
                if not ok:
                    self.get_logger().warn(f"{label}: wrist flip detected (speed fell to {slowest:.0%})")
            else:
                self.get_logger().warn(f"{label}: only {fraction:.0%} of the line is plannable")

            if not ok:
                # failure policy: gun stays off, skip the stripe, re-position for the next one
                failed += 1
                self.report.append((name, st["id"], "FAILED"))
                if failed > self.paint["max_failed_stripe_fraction"] * len(stripes):
                    self.get_logger().error(f"{name}: more than 10 % of stripes failed, stopping section")
                    break
                if i + 1 < len(stripes):
                    try:
                        self.move_tcp_to(stripes[i + 1]["approach"], f"{label}: re-position")
                    except PlanFailure as e:
                        # no motion happened; the next stripe's line starts from where we are
                        self.get_logger().warn(f"{e}: trying the next stripe from here")
                continue

            self.get_logger().info(f"{label}: gun ON, {st['length']:.2f} m at {self.paint['speed']} m/s")
            self.run_trajectory(traj, label)
            self.get_logger().info(f"{label}: gun OFF")
            self.paint_stripe_markers(name, st)
            self.report.append((name, st["id"], "painted"))

        self.get_logger().info(f"=== {name}: finished in {time.time() - t0:.0f} s, {failed} failed stripes")

    def run(self):
        self.wait_for_interfaces()
        poses = self.named_poses()
        sections = self.strokes["sections"]
        if self.section_param != "all":                  # R13: one chosen section
            sections = [s for s in sections if s["name"] == self.section_param]
        self.publish_plan(sections)
        start = time.time()
        try:
            for s in sections:
                self.paint_section(s, poses)
            try:
                self.move_joints("arm", poses["transit"], "finish: arm to transit")
                self.move_joints("gantry", poses["parked"], "finish: gantry parked")
                self.move_joints("arm", poses["home"], "finish: arm home")
                self.get_logger().info("job done: car can leave the booth")
            except PlanFailure as e:
                self.get_logger().error(f"{e}: robot not parked, check before releasing the car")
        except SafetyStop as e:
            self.get_logger().error(f"SAFETY STOP: {e}. Gun off, no further motion.")
        painted = sum(1 for r in self.report if r[2] == "painted")
        failed = [r for r in self.report if r[2] == "FAILED"]
        self.get_logger().info(f"report: {painted} stripes painted, {len(failed)} failed, "
                               f"total {time.time() - start:.0f} s")
        for sec, sid, _ in failed:
            self.get_logger().info(f"  not painted: {sec} stripe {sid}")


def main():
    rclpy.init()
    node = PaintExecutor()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()