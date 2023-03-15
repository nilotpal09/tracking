cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GraphBuilder LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CUDA_FLAGS "-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

enable_language("CUDA")
add_executable(main ${CMAKE_SOURCE_DIR}/apps/main.cu ${gpu_source_files})