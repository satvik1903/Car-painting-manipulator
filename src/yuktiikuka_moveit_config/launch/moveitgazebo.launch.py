# =============================================================================
# moveit_gazebo.launch.py
# Starts MoveIt (move_group) and RViz for the robot that gazebo.launch.py
# is ALREADY running. It does not start a robot, controllers or Gazebo.
#
# Run, in a second terminal, after gazebo.launch.py is up:
#     ros2 launch yuktiikuka_moveit_config moveit_gazebo.launch.py
# =============================================================================
import os

from ament_index_python.packages import  get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    """
    Load the moveit configurator """

    robot_xacro=os.path.join(
        get_package_share_directory("yuktiikuka"),"urdf", "kr20_r3100.urdf.xacro"
    )

    moveit_config=(
        MoveItConfigsBuilder("kr20_r3100", package_name="yuktiikuka_moveit_config")
        .robot_description(file_path=robot_xacro)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    sim_time={"use_sim_time":True}

    #move group: Moveits main program
    #This holds the robot model and then runs IK and the planner

    move_group=Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), sim_time,
        {"trajectory_execution.allowed_start_tolerance": 0.05},
        ]
    )

    #Rviz with the MoveIt motion planning panel

    rviz_config=os.path.join(
        get_package_share_directory("yuktiikuka_moveit_config"), "config",
        "moveit.rviz"
    )
    rviz=Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            sim_time,
        ],
    )
        # Loads the car and booth obstacles into MoveIt's planning scene (R2).
    # Waits for move_group by itself, so the start order doesn't matter.
    scene_setup = Node(
        package="yuktiikuka",
        executable="planning_scene_setup.py",
        output="screen",
        parameters=[sim_time],
    )

    return LaunchDescription([move_group, rviz, scene_setup])