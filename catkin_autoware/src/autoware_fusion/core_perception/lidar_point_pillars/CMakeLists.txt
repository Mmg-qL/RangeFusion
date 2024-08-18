cmake_minimum_required(VERSION 3.10.0)
project(lidar_point_pillars)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# set flags for CUDA availability
find_package(CUDA)
find_package(autoware_build_flags REQUIRED)

#mmg 2024-01-12
set(TensorRT_ROOT "/home/gmm/file/TensorRT-8.6.1.6")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    pcl_ros
    autoware_msgs)

catkin_package(
    CATKIN_DEPENDS
      roslib
      pcl_ros
      autoware_msgs)

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} #mmg 2024-01-12
    ${TensorRT_ROOT}/include #mmg 2024-01-12
    ${TensorRT_ROOT}/samples/common #mmg 2024-01-12
    ${catkin_INCLUDE_DIRS}
  )

link_directories(
  ${CUDA_LIBRARIES}
  ${TensorRT_ROOT}/lib #mmg 2024-01-12
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64 #mmg 2024-01-12
)


set(SOURCE_FILES
  nodes/lidar_point_pillars_node.cpp
  nodes/point_pillars_ros.cpp
)

add_executable(lidar_point_pillars
  ${SOURCE_FILES}
)

add_dependencies(lidar_point_pillars
  ${catkin_EXPORTED_TARGETS}
)

file(GLOB_RECURSE cpp_cuda_srcs
    nodes/preprocess_points_cuda.cu
    nodes/anchor_mask_cuda.cu
    nodes/scatter_cuda.cu
    nodes/postprocess_cuda.cu
    nodes/nms_cuda.cu
    ${TensorRT_ROOT}/samples/common/logger.cpp 
    ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
    ${TensorRT_ROOT}/samples/common/sampleUtils.cpp
)
cuda_add_library(gpu_point_pillars_lib SHARED ${cpp_cuda_srcs})

target_link_libraries(gpu_point_pillars_lib
  ${CUDA_LIBRARIES}
)

add_library(point_pillars_lib
  nodes/point_pillars.cpp
  nodes/preprocess_points.cpp
)
  
target_link_libraries(point_pillars_lib
  # ${NVINFER}
  # ${NVONNXPARSERS}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDNN_LIBRARY}
  nvinfer
  nvonnxparser 
  nvinfer_plugin
  gpu_point_pillars_lib
)

target_link_libraries(lidar_point_pillars
  ${catkin_LIBRARIES}
  point_pillars_lib
)

install(
  TARGETS
    gpu_point_pillars_lib
    point_pillars_lib
    lidar_point_pillars
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/${PROJECT_NAME}/
  PATTERN ".svn" EXCLUDE
)

if (CATKIN_ENABLE_TESTING AND EXISTS /proc/driver/nvidia/)
  find_package(rostest REQUIRED)
  catkin_add_gtest(test-point_pillars test/src/test_point_pillars.cpp)
  target_link_libraries(test-point_pillars ${catkin_LIBRARIES} point_pillars_lib)
endif()

