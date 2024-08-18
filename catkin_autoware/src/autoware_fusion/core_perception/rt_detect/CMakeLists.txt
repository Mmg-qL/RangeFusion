cmake_minimum_required(VERSION 3.10.0)
project(rt_detect)
add_definitions(-std=c++14 -w)

# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspaces)
# set(CMAKE_INSTALL_PREFIX ${EXECUTABLE_OUTPUT_PATH}/install/) # make install时的存储路径

set(CMAKE_BUILD_TYPE "Release") # 运行项目时开启

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_89,code=compute_89")

# mmg 2024-01-11
find_package(catkin REQUIRED COMPONENTS
            roscpp
            sensor_msgs
            cv_bridge
            image_transport
            roslib
            autoware_msgs
            tf
            tf_conversions)
            
find_package(CUDA REQUIRED) 
find_package(OpenCV REQUIRED) 
find_package(Eigen3 REQUIRED) 

catkin_package(CATKIN_DEPENDS roscpp sensor_msgs cv_bridge autoware_msgs) # mmg 2024-01-11

set(TensorRT_ROOT "/home/gmm/file/TensorRT-8.6.1.6") # 设置tensorrt8.xx根目录，改为你自己的即可

# 5. include所有要用到的hpp文件路径
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS} # 追踪要用到

    # tensorrt
    ${TensorRT_ROOT}/include
    ${TensorRT_ROOT}/samples/common # 导入这个主要是为了适应于trt多版本[v7.xx,v8.xx]的logger导入

    # 项目里要用到的
    ${PROJECT_SOURCE_DIR}/utils
    ${PROJECT_SOURCE_DIR}/application
)

link_directories(
    ${CUDA_LIBRARIES}
    # ${CUDA_cublas_LIBRARY}
    # ${CUDA_cudart_static_LIBRARY}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64

    # tensorrt
    ${TensorRT_ROOT}/lib
)

# # 7. 将utils里写好的cu文件和cpp文件编译成so库，方便后面调用
file(GLOB_RECURSE cpp_cuda_srcs
    ${PROJECT_SOURCE_DIR}/utils/*.cpp
    ${PROJECT_SOURCE_DIR}/application/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/*.cu
    ${TensorRT_ROOT}/samples/common/logger.cpp 
    ${TensorRT_ROOT}/samples/common/sampleOptions.cpp 
    ${TensorRT_ROOT}/samples/common/sampleUtils.cpp
)
cuda_add_library(utils_cu_cpp SHARED ${cpp_cuda_srcs})

add_executable(rt_detect # mmg 2024-01-11
            src/rt_detect.cpp 
            src/rt_detect_node.cpp
            src/main_rtdetr.cpp
            src/det3Dbox.cpp
            ) # mmg 2024-01-11
            

# 8. 链接要所有要用到的so库
target_link_libraries(rt_detect
    ${catkin_LIBRARIES}
    utils_cu_cpp 
    cuda
    cudart
    cudnn
    pthread
    ${CUDA_LIBRARIES} # mmg 2024-01-11
    ${OpenCV_LIBS}
    nvinfer 
    nvinfer_plugin
    nvonnxparser   
)

# # make install 时需要用到
install(TARGETS rt_detect utils_cu_cpp
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)

  
install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/utils/
        ${PROJECT_SOURCE_DIR}/application/
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh")

# 通过make auto -j 来编译和运行程序
# add_custom_target(
#     auto
#     DEPENDS infer
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspaces
#     COMMAND ./infer
# )
