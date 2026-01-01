cmake_minimum_required(VERSION 3.8)
project(grasp_pose_detection)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Generate the service interface
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetBestGrasp.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

# Install the python package
ament_python_install_package(${PROJECT_NAME})

# Install the runnable node
install(PROGRAMS
  scripts/hggd_grasp_service_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
