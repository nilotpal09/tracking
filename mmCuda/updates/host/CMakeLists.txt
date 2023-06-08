file(GLOB host_source_files "${CMAKE_SOURCE_DIR}/host/src/*.cu")
file(GLOB host_header_files "${CMAKE_SOURCE_DIR}/host/include/*.cuh")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})

add_library(host STATIC "${host_source_files}" "${host_header_files}")