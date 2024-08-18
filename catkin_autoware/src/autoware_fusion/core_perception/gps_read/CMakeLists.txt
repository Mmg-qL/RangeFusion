cmake_minimum_required(VERSION 2.8.12)
project(gps_reader)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  gps_common
  pcl_ros
  tf  
  tf_conversions #mmg 2023-12-5
  message_filters
)

find_package(Eigen3 REQUIRED)  #mmg 2023-12-5


catkin_package(
  INCLUDE_DIRS 
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs gps_common
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(gps_reader src/gps_reader.cpp src/gps_reader_main.cpp)

target_link_libraries(gps_reader
  ${catkin_LIBRARIES}
  ${EIGEN3_INCLUDE_DIR}
)

install(
  TARGETS 
  gps_reader 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

