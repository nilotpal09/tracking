# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /srv01/agrp/nilotpal/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /srv01/agrp/nilotpal/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build

# Include any dependencies generated for this target.
include common/CMakeFiles/common.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include common/CMakeFiles/common.dir/compiler_depend.make

# Include the progress variables for this target.
include common/CMakeFiles/common.dir/progress.make

# Include the compile flags for this target's objects.
include common/CMakeFiles/common.dir/flags.make

# Object files for target common
common_OBJECTS =

# External object files for target common
common_EXTERNAL_OBJECTS =

common/libcommon.a: common/CMakeFiles/common.dir/build.make
common/libcommon.a: common/CMakeFiles/common.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CUDA static library libcommon.a"
	cd /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build/common && $(CMAKE_COMMAND) -P CMakeFiles/common.dir/cmake_clean_target.cmake
	cd /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build/common && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/common.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
common/CMakeFiles/common.dir/build: common/libcommon.a
.PHONY : common/CMakeFiles/common.dir/build

common/CMakeFiles/common.dir/clean:
	cd /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build/common && $(CMAKE_COMMAND) -P CMakeFiles/common.dir/cmake_clean.cmake
.PHONY : common/CMakeFiles/common.dir/clean

common/CMakeFiles/common.dir/depend:
	cd /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/common /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build/common /srv01/agrp/nilotpal/projects/tracking/mmCuda/updates/build/common/CMakeFiles/common.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : common/CMakeFiles/common.dir/depend
