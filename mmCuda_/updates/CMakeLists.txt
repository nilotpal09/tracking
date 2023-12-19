cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(GraphBuilder LANGUAGES CUDA CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CUDA_FLAGS "-DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

# Set CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")

# Set flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -ffast-math -O3")
MESSAGE(STATUS "Setting g++ flags for Release configuration")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s")  ## Strip binary

add_subdirectory(common)
set_target_properties(common PROPERTIES LINKER_LANGUAGE CUDA)
add_subdirectory(host)
add_subdirectory(device)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
add_executable(GraphBuilder ${CMAKE_SOURCE_DIR}/apps/main.cu)
set_target_properties(GraphBuilder PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                             RUNTIME_OUTPUT_DIRECTORY
                                 "${CMAKE_SOURCE_DIR}/build")
target_link_libraries(GraphBuilder PRIVATE host device common)