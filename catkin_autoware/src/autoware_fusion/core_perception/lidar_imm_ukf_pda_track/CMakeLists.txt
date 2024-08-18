cmake_minimum_required(VERSION 2.8.3)
project(imm_ukf_pda_track)

find_package(autoware_build_flags REQUIRED)
find_package(OpenCV REQUIRED)  #mmg 2023-12-5
find_package(Eigen3 REQUIRED)  #mmg 2023-12-5

find_package(catkin REQUIRED COMPONENTS
  amathutils_lib
  autoware_msgs
  geometry_msgs
  pcl_ros
  roscpp
  roslint
  tf  #mmg 2023-12-5
  vector_map
  tf_conversions #mmg 2023-12-5
  nav_msgs #mmg 2023-12-18
  message_filters #mmg 2023-12-18
  sensor_msgs #mmg 2023-12-18
  geometry_msgs #mmg 2023-12-18
  std_msgs #mmg 2023-12-26
  gps_common
)


set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

catkin_package(
  CATKIN_DEPENDS
    vector_map
    amathutils_lib
)

set(ROSLINT_CPP_OPTS "--filter=-build/c++14")


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR} #mmg 2023-12-26
)

#imm_ukf_pda
add_executable(imm_ukf_pda
  nodes/imm_ukf_pda/imm_ukf_pda_main.cpp
  nodes/imm_ukf_pda/imm_ukf_pda.cpp
  nodes/imm_ukf_pda/ukf.cpp
)
target_link_libraries(imm_ukf_pda
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES} #mmg 2023-12-5
)
add_dependencies(imm_ukf_pda
  ${catkin_EXPORTED_TARGETS}
)




install(
  TARGETS
    imm_ukf_pda
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

if (CATKIN_ENABLE_TESTING)
  roslint_add_test()
endif()
