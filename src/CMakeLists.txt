cmake_minimum_required(VERSION 3.10)
project(CSVToVideoCUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

cuda_add_executable(csv_to_video_cuda csv_to_video_cuda.cpp)

target_link_libraries(csv_to_video_cuda ${OpenCV_LIBS} ${CUDA_LIBRARIES})

set_target_properties(csv_to_video_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)