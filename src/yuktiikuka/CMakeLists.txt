cmake_minimum_required(VERSION 3.8)
project(yuktiikuka)
find_package(ament_cmake REQUIRED)
install(DIRECTORY config launch meshes urdf worlds
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS scripts/planning_scene_setup.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
