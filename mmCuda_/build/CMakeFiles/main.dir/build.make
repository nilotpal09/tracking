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
CMAKE_SOURCE_DIR = /srv01/agrp/nilotpal/projects/tracking/mmCuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /srv01/agrp/nilotpal/projects/tracking/mmCuda/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/apps/main.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/apps/main.cu.o: CMakeFiles/main.dir/includes_CUDA.rsp
CMakeFiles/main.dir/apps/main.cu.o: /srv01/agrp/nilotpal/projects/tracking/mmCuda/apps/main.cu
CMakeFiles/main.dir/apps/main.cu.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/main.dir/apps/main.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/main.dir/apps/main.cu.o -MF CMakeFiles/main.dir/apps/main.cu.o.d -x cu -c /srv01/agrp/nilotpal/projects/tracking/mmCuda/apps/main.cu -o CMakeFiles/main.dir/apps/main.cu.o

CMakeFiles/main.dir/apps/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/main.dir/apps/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/apps/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/main.dir/apps/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/cuda_kernels.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/cuda_kernels.cu.o: CMakeFiles/main.dir/includes_CUDA.rsp
CMakeFiles/main.dir/src/cuda_kernels.cu.o: /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/cuda_kernels.cu
CMakeFiles/main.dir/src/cuda_kernels.cu.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/main.dir/src/cuda_kernels.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/main.dir/src/cuda_kernels.cu.o -MF CMakeFiles/main.dir/src/cuda_kernels.cu.o.d -x cu -c /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/cuda_kernels.cu -o CMakeFiles/main.dir/src/cuda_kernels.cu.o

CMakeFiles/main.dir/src/cuda_kernels.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/cuda_kernels.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/cuda_kernels.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/cuda_kernels.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/datatypes.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/datatypes.cu.o: CMakeFiles/main.dir/includes_CUDA.rsp
CMakeFiles/main.dir/src/datatypes.cu.o: /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/datatypes.cu
CMakeFiles/main.dir/src/datatypes.cu.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/main.dir/src/datatypes.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/main.dir/src/datatypes.cu.o -MF CMakeFiles/main.dir/src/datatypes.cu.o.d -x cu -c /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/datatypes.cu -o CMakeFiles/main.dir/src/datatypes.cu.o

CMakeFiles/main.dir/src/datatypes.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/datatypes.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/datatypes.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/datatypes.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/device_kernals.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/device_kernals.cu.o: CMakeFiles/main.dir/includes_CUDA.rsp
CMakeFiles/main.dir/src/device_kernals.cu.o: /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/device_kernals.cu
CMakeFiles/main.dir/src/device_kernals.cu.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/main.dir/src/device_kernals.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/main.dir/src/device_kernals.cu.o -MF CMakeFiles/main.dir/src/device_kernals.cu.o.d -x cu -c /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/device_kernals.cu -o CMakeFiles/main.dir/src/device_kernals.cu.o

CMakeFiles/main.dir/src/device_kernals.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/device_kernals.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/device_kernals.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/device_kernals.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/memoryscheduler.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/memoryscheduler.cu.o: CMakeFiles/main.dir/includes_CUDA.rsp
CMakeFiles/main.dir/src/memoryscheduler.cu.o: /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/memoryscheduler.cu
CMakeFiles/main.dir/src/memoryscheduler.cu.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/main.dir/src/memoryscheduler.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/main.dir/src/memoryscheduler.cu.o -MF CMakeFiles/main.dir/src/memoryscheduler.cu.o.d -x cu -c /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/memoryscheduler.cu -o CMakeFiles/main.dir/src/memoryscheduler.cu.o

CMakeFiles/main.dir/src/memoryscheduler.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/memoryscheduler.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/memoryscheduler.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/memoryscheduler.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/selection.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/selection.cu.o: CMakeFiles/main.dir/includes_CUDA.rsp
CMakeFiles/main.dir/src/selection.cu.o: /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/selection.cu
CMakeFiles/main.dir/src/selection.cu.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CUDA object CMakeFiles/main.dir/src/selection.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/main.dir/src/selection.cu.o -MF CMakeFiles/main.dir/src/selection.cu.o.d -x cu -c /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/selection.cu -o CMakeFiles/main.dir/src/selection.cu.o

CMakeFiles/main.dir/src/selection.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/selection.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/selection.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/selection.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/triplet_finder.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/triplet_finder.cu.o: CMakeFiles/main.dir/includes_CUDA.rsp
CMakeFiles/main.dir/src/triplet_finder.cu.o: /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/triplet_finder.cu
CMakeFiles/main.dir/src/triplet_finder.cu.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CUDA object CMakeFiles/main.dir/src/triplet_finder.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/main.dir/src/triplet_finder.cu.o -MF CMakeFiles/main.dir/src/triplet_finder.cu.o.d -x cu -c /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/triplet_finder.cu -o CMakeFiles/main.dir/src/triplet_finder.cu.o

CMakeFiles/main.dir/src/triplet_finder.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/triplet_finder.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/triplet_finder.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/triplet_finder.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/main.dir/src/utility.cu.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/utility.cu.o: CMakeFiles/main.dir/includes_CUDA.rsp
CMakeFiles/main.dir/src/utility.cu.o: /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/utility.cu
CMakeFiles/main.dir/src/utility.cu.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/main.dir/src/utility.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/main.dir/src/utility.cu.o -MF CMakeFiles/main.dir/src/utility.cu.o.d -x cu -c /srv01/agrp/nilotpal/projects/tracking/mmCuda/src/utility.cu -o CMakeFiles/main.dir/src/utility.cu.o

CMakeFiles/main.dir/src/utility.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/main.dir/src/utility.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/main.dir/src/utility.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/main.dir/src/utility.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/apps/main.cu.o" \
"CMakeFiles/main.dir/src/cuda_kernels.cu.o" \
"CMakeFiles/main.dir/src/datatypes.cu.o" \
"CMakeFiles/main.dir/src/device_kernals.cu.o" \
"CMakeFiles/main.dir/src/memoryscheduler.cu.o" \
"CMakeFiles/main.dir/src/selection.cu.o" \
"CMakeFiles/main.dir/src/triplet_finder.cu.o" \
"CMakeFiles/main.dir/src/utility.cu.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/apps/main.cu.o
main: CMakeFiles/main.dir/src/cuda_kernels.cu.o
main: CMakeFiles/main.dir/src/datatypes.cu.o
main: CMakeFiles/main.dir/src/device_kernals.cu.o
main: CMakeFiles/main.dir/src/memoryscheduler.cu.o
main: CMakeFiles/main.dir/src/selection.cu.o
main: CMakeFiles/main.dir/src/triplet_finder.cu.o
main: CMakeFiles/main.dir/src/utility.cu.o
main: CMakeFiles/main.dir/build.make
main: CMakeFiles/main.dir/linkLibs.rsp
main: CMakeFiles/main.dir/objects1.rsp
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CUDA executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /srv01/agrp/nilotpal/projects/tracking/mmCuda/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /srv01/agrp/nilotpal/projects/tracking/mmCuda /srv01/agrp/nilotpal/projects/tracking/mmCuda /srv01/agrp/nilotpal/projects/tracking/mmCuda/build /srv01/agrp/nilotpal/projects/tracking/mmCuda/build /srv01/agrp/nilotpal/projects/tracking/mmCuda/build/CMakeFiles/main.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/main.dir/depend

