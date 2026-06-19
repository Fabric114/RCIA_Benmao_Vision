cmake_minimum_required(VERSION 3.10)
project(rcia_energy_trace_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

get_filename_component(RCIA_ENERGY_TRACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

set(USE_REALSENSE OFF CACHE BOOL "Disable M3T RealSense backend for ROS frame bridge" FORCE)
set(USE_AZURE_KINECT OFF CACHE BOOL "Disable M3T Azure Kinect backend for ROS frame bridge" FORCE)
set(USE_GTEST OFF CACHE BOOL "Disable upstream M3T tests" FORCE)
set(BUILD_M3T_EXAMPLES OFF CACHE BOOL "Disable upstream M3T examples" FORCE)
set(BUILD_M3T_DOC OFF CACHE BOOL "Disable upstream M3T docs" FORCE)
add_subdirectory("${RCIA_ENERGY_TRACE_ROOT}/third_party/m3t" "${CMAKE_CURRENT_BINARY_DIR}/m3t")

add_library(ros_frame_cameras
  src/axis_normal_viewer.cpp
  src/ros_frame_cameras.cpp
)
target_include_directories(ros_frame_cameras PUBLIC
  include
  "${RCIA_ENERGY_TRACE_ROOT}/third_party/m3t/include"
)
target_link_libraries(ros_frame_cameras m3t)
ament_target_dependencies(ros_frame_cameras
  rclcpp
  sensor_msgs
  cv_bridge
)

add_executable(m3t_ros_tracking_node
  src/m3t_ros_tracking_node.cpp
)
target_include_directories(m3t_ros_tracking_node PRIVATE
  include
  "${RCIA_ENERGY_TRACE_ROOT}"
  "${RCIA_ENERGY_TRACE_ROOT}/third_party/m3t/include"
)
target_link_libraries(m3t_ros_tracking_node
  ros_frame_cameras
  m3t
)
target_compile_definitions(m3t_ros_tracking_node PRIVATE
  RCIA_PROJECT_ROOT="${RCIA_ENERGY_TRACE_ROOT}"
)
ament_target_dependencies(m3t_ros_tracking_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
)

add_executable(realsense_rgbd_publisher_node
  src/realsense_rgbd_publisher_node.cpp
)
target_link_libraries(realsense_rgbd_publisher_node
  ${realsense2_LIBRARY}
  ${OpenCV_LIBRARIES}
)
target_include_directories(realsense_rgbd_publisher_node PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(realsense_rgbd_publisher_node
  rclcpp
  sensor_msgs
)

install(TARGETS
  ros_frame_cameras
  m3t_ros_tracking_node
  realsense_rgbd_publisher_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/foundationpose_init_node.py
  scripts/pipeline_visualizer_node.py
  scripts/save_rgbd_frame_node.py
  scripts/tracking_health_debug_node.py
  scripts/tracking_supervisor_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
