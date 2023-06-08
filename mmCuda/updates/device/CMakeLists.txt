file(GLOB device_source_files "${CMAKE_SOURCE_DIR}/device/src/*.cu")
file(GLOB device_header_files "${CMAKE_SOURCE_DIR}/device/include/*.cuh")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

set(CMAKE_CUDA_FLAGS "-rdc=true -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT")
add_library(device STATIC "${device_source_files}" "${device_header_files}")