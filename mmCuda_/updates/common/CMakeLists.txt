file(GLOB common_source_files "${CMAKE_SOURCE_DIR}/common/src/*.cu")
file(GLOB common_header_files "${CMAKE_SOURCE_DIR}/common/include/*.cuh")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

add_library(common STATIC "${common_source_files}" "${common_header_files}")