# gazebo.launch.py starts gazebo harmonic, spawns the KUKA 20 R3100

import os
from ament_index_python.packages import get_package_share_directory 
from launch import LaunchDescription #The todo list for ROS

from launch.actions import(
    IncludeLaunchDescription, #running gazebos launch file in this one 
    AppendEnvironmentVariable, 
    RegisterEventHandler, 
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource #include a python launch file
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue 

def generate_launch_description():
    pkg_share=get_package_share_directory("yuktiikuka")
    xacro_file=os.path.join(pkg_share, "urdf", "kr20_r3100.urdf.xacro")
    #telling gazebo where the meshes are 
    set_mesh_path=AppendEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=os.path.join(pkg_share,".."),
    )
    #xacro to urdf
    robot_description=ParameterValue(
        Command([FindExecutable(name="xacro")," ",xacro_file]),
        value_type=str,
    )
    #loading world file
    world_file=os.path.join(pkg_share, "worlds", "paint_booth.sdf")
    #start gazebo
    gazebo=IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"

            )
        ),
        launch_arguments={"gz_args": f"-r {world_file}"}.items(),
    )

    #robot state publisher
    robot_state_publisher=Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    #spawn robot in gz

    spawn_robot=Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", "kr20_r3100"],  
    )

    #Clock bridge

    clock_bridge=Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
    )

    #Controller Spawner

    joint_state_broadcaster_spawner=Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    #Take trajectories
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

        # Drives the two gantry slide joints (gantry_x_joint, gantry_y_joint).
    gantry_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=["gantry_controller", "--controller-manager", "/controller_manager"],
    )
 
    start_jsb_after_spawn = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )
    start_arm_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )
    start_gantry_after_arm = RegisterEventHandler(
        OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gantry_controller_spawner],
        )
    )
 

    # 10. Everything to launch

    return LaunchDescription(
        [
            set_mesh_path,
            gazebo,
            robot_state_publisher,
            clock_bridge,
            spawn_robot,
            start_jsb_after_spawn,
            start_arm_after_jsb,
            start_gantry_after_arm,
        ]
    )
 