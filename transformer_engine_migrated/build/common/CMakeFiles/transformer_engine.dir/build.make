# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build

# Include any dependencies generated for this target.
include common/CMakeFiles/transformer_engine.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include common/CMakeFiles/transformer_engine.dir/compiler_depend.make

# Include the progress variables for this target.
include common/CMakeFiles/transformer_engine.dir/progress.make

# Include the compile flags for this target's objects.
include common/CMakeFiles/transformer_engine.dir/flags.make

common/CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o: common/CMakeFiles/transformer_engine.dir/flags.make
common/CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o: /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/transformer_engine.cpp
common/CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o: common/CMakeFiles/transformer_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object common/CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o -MF CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o.d -o CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o -c /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/transformer_engine.cpp

common/CMakeFiles/transformer_engine.dir/transformer_engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/transformer_engine.dir/transformer_engine.cpp.i"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/transformer_engine.cpp > CMakeFiles/transformer_engine.dir/transformer_engine.cpp.i

common/CMakeFiles/transformer_engine.dir/transformer_engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/transformer_engine.dir/transformer_engine.cpp.s"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/transformer_engine.cpp -o CMakeFiles/transformer_engine.dir/transformer_engine.cpp.s

common/CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o: common/CMakeFiles/transformer_engine.dir/flags.make
common/CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o: /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/fused_attn/fused_attn.cpp
common/CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o: common/CMakeFiles/transformer_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object common/CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o -MF CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o.d -o CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o -c /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/fused_attn/fused_attn.cpp

common/CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.i"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/fused_attn/fused_attn.cpp > CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.i

common/CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.s"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/fused_attn/fused_attn.cpp -o CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.s

common/CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o: common/CMakeFiles/transformer_engine.dir/flags.make
common/CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o: /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/layer_norm/ln_api.cpp
common/CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o: common/CMakeFiles/transformer_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object common/CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o -MF CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o.d -o CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o -c /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/layer_norm/ln_api.cpp

common/CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.i"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/layer_norm/ln_api.cpp > CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.i

common/CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.s"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/layer_norm/ln_api.cpp -o CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.s

common/CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o: common/CMakeFiles/transformer_engine.dir/flags.make
common/CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o: /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/rmsnorm/rmsnorm_api.cpp
common/CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o: common/CMakeFiles/transformer_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object common/CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o -MF CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o.d -o CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o -c /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/rmsnorm/rmsnorm_api.cpp

common/CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.i"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/rmsnorm/rmsnorm_api.cpp > CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.i

common/CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.s"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/rmsnorm/rmsnorm_api.cpp -o CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.s

common/CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o: common/CMakeFiles/transformer_engine.dir/flags.make
common/CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o: /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/cuda_driver.cpp
common/CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o: common/CMakeFiles/transformer_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object common/CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o -MF CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o.d -o CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o -c /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/cuda_driver.cpp

common/CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.i"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/cuda_driver.cpp > CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.i

common/CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.s"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/cuda_driver.cpp -o CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.s

common/CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o: common/CMakeFiles/transformer_engine.dir/flags.make
common/CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o: /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/cuda_runtime.cpp
common/CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o: common/CMakeFiles/transformer_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object common/CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o -MF CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o.d -o CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o -c /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/cuda_runtime.cpp

common/CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.i"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/cuda_runtime.cpp > CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.i

common/CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.s"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/cuda_runtime.cpp -o CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.s

common/CMakeFiles/transformer_engine.dir/util/rtc.cpp.o: common/CMakeFiles/transformer_engine.dir/flags.make
common/CMakeFiles/transformer_engine.dir/util/rtc.cpp.o: /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/rtc.cpp
common/CMakeFiles/transformer_engine.dir/util/rtc.cpp.o: common/CMakeFiles/transformer_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object common/CMakeFiles/transformer_engine.dir/util/rtc.cpp.o"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/CMakeFiles/transformer_engine.dir/util/rtc.cpp.o -MF CMakeFiles/transformer_engine.dir/util/rtc.cpp.o.d -o CMakeFiles/transformer_engine.dir/util/rtc.cpp.o -c /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/rtc.cpp

common/CMakeFiles/transformer_engine.dir/util/rtc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/transformer_engine.dir/util/rtc.cpp.i"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/rtc.cpp > CMakeFiles/transformer_engine.dir/util/rtc.cpp.i

common/CMakeFiles/transformer_engine.dir/util/rtc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/transformer_engine.dir/util/rtc.cpp.s"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/rtc.cpp -o CMakeFiles/transformer_engine.dir/util/rtc.cpp.s

common/CMakeFiles/transformer_engine.dir/util/system.cpp.o: common/CMakeFiles/transformer_engine.dir/flags.make
common/CMakeFiles/transformer_engine.dir/util/system.cpp.o: /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/system.cpp
common/CMakeFiles/transformer_engine.dir/util/system.cpp.o: common/CMakeFiles/transformer_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object common/CMakeFiles/transformer_engine.dir/util/system.cpp.o"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT common/CMakeFiles/transformer_engine.dir/util/system.cpp.o -MF CMakeFiles/transformer_engine.dir/util/system.cpp.o.d -o CMakeFiles/transformer_engine.dir/util/system.cpp.o -c /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/system.cpp

common/CMakeFiles/transformer_engine.dir/util/system.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/transformer_engine.dir/util/system.cpp.i"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/system.cpp > CMakeFiles/transformer_engine.dir/util/system.cpp.i

common/CMakeFiles/transformer_engine.dir/util/system.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/transformer_engine.dir/util/system.cpp.s"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common/util/system.cpp -o CMakeFiles/transformer_engine.dir/util/system.cpp.s

# Object files for target transformer_engine
transformer_engine_OBJECTS = \
"CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o" \
"CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o" \
"CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o" \
"CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o" \
"CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o" \
"CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o" \
"CMakeFiles/transformer_engine.dir/util/rtc.cpp.o" \
"CMakeFiles/transformer_engine.dir/util/system.cpp.o"

# External object files for target transformer_engine
transformer_engine_EXTERNAL_OBJECTS =

common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/transformer_engine.cpp.o
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/fused_attn/fused_attn.cpp.o
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/layer_norm/ln_api.cpp.o
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/rmsnorm/rmsnorm_api.cpp.o
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/util/cuda_driver.cpp.o
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/util/cuda_runtime.cpp.o
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/util/rtc.cpp.o
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/util/system.cpp.o
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/build.make
common/libtransformer_engine.so: common/CMakeFiles/transformer_engine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX shared library libtransformer_engine.so"
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/transformer_engine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
common/CMakeFiles/transformer_engine.dir/build: common/libtransformer_engine.so
.PHONY : common/CMakeFiles/transformer_engine.dir/build

common/CMakeFiles/transformer_engine.dir/clean:
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common && $(CMAKE_COMMAND) -P CMakeFiles/transformer_engine.dir/cmake_clean.cmake
.PHONY : common/CMakeFiles/transformer_engine.dir/clean

common/CMakeFiles/transformer_engine.dir/depend:
	cd /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/common /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common /home/intel/project/transformer_engine_v1.7_oneapi2025/transformer_engine/migrated/build/common/CMakeFiles/transformer_engine.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : common/CMakeFiles/transformer_engine.dir/depend

