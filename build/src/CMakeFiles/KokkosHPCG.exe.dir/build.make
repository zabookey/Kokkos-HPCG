# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /projects/install/rhel6-x86_64/sems/utility/cmake/2.8.12/bin/cmake

# The command to remove a file.
RM = /projects/install/rhel6-x86_64/sems/utility/cmake/2.8.12/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zabooke/Project/KokkosHPCG

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zabooke/Project/KokkosHPCG/build

# Include any dependencies generated for this target.
include src/CMakeFiles/KokkosHPCG.exe.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/KokkosHPCG.exe.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/KokkosHPCG.exe.dir/flags.make

src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o: src/CMakeFiles/KokkosHPCG.exe.dir/flags.make
src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o: ../src/main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/zabooke/Project/KokkosHPCG/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o"
	cd /home/zabooke/Project/KokkosHPCG/build/src && /projects/install/rhel6-x86_64/sems/compiler/gcc/4.9.2/openmpi/1.6.5/bin/mpicxx   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o -c /home/zabooke/Project/KokkosHPCG/src/main.cpp

src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KokkosHPCG.exe.dir/main.cpp.i"
	cd /home/zabooke/Project/KokkosHPCG/build/src && /projects/install/rhel6-x86_64/sems/compiler/gcc/4.9.2/openmpi/1.6.5/bin/mpicxx  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/zabooke/Project/KokkosHPCG/src/main.cpp > CMakeFiles/KokkosHPCG.exe.dir/main.cpp.i

src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KokkosHPCG.exe.dir/main.cpp.s"
	cd /home/zabooke/Project/KokkosHPCG/build/src && /projects/install/rhel6-x86_64/sems/compiler/gcc/4.9.2/openmpi/1.6.5/bin/mpicxx  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/zabooke/Project/KokkosHPCG/src/main.cpp -o CMakeFiles/KokkosHPCG.exe.dir/main.cpp.s

src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o.requires:
.PHONY : src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o.requires

src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o.provides: src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/KokkosHPCG.exe.dir/build.make src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o.provides.build
.PHONY : src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o.provides

src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o.provides.build: src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o

# Object files for target KokkosHPCG.exe
KokkosHPCG_exe_OBJECTS = \
"CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o"

# External object files for target KokkosHPCG.exe
KokkosHPCG_exe_EXTERNAL_OBJECTS =

src/KokkosHPCG.exe: src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o
src/KokkosHPCG.exe: src/CMakeFiles/KokkosHPCG.exe.dir/build.make
src/KokkosHPCG.exe: src/libkokkoshpcglib.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetraext.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetrainout.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetra.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libkokkostsqr.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetrakernels.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetraclassiclinalg.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetraclassicnodeapi.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetraclassic.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetraext.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetrainout.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetra.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libkokkostsqr.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetrakernels.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetraclassiclinalg.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetraclassicnodeapi.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libtpetraclassic.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchoskokkoscomm.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchoskokkoscompat.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchosremainder.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchosnumerics.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchoscomm.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchosparameterlist.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchoscore.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchoskokkoscomm.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchoskokkoscompat.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchosremainder.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchosnumerics.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchoscomm.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchosparameterlist.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libteuchoscore.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libkokkosalgorithms.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libkokkoscontainers.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libkokkoscore.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libkokkosalgorithms.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libkokkoscontainers.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libkokkoscore.a
src/KokkosHPCG.exe: /home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build/lib/libgtest.a
src/KokkosHPCG.exe: /usr/lib64/liblapack.so
src/KokkosHPCG.exe: /usr/lib64/libblas.so
src/KokkosHPCG.exe: src/CMakeFiles/KokkosHPCG.exe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable KokkosHPCG.exe"
	cd /home/zabooke/Project/KokkosHPCG/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/KokkosHPCG.exe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/KokkosHPCG.exe.dir/build: src/KokkosHPCG.exe
.PHONY : src/CMakeFiles/KokkosHPCG.exe.dir/build

src/CMakeFiles/KokkosHPCG.exe.dir/requires: src/CMakeFiles/KokkosHPCG.exe.dir/main.cpp.o.requires
.PHONY : src/CMakeFiles/KokkosHPCG.exe.dir/requires

src/CMakeFiles/KokkosHPCG.exe.dir/clean:
	cd /home/zabooke/Project/KokkosHPCG/build/src && $(CMAKE_COMMAND) -P CMakeFiles/KokkosHPCG.exe.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/KokkosHPCG.exe.dir/clean

src/CMakeFiles/KokkosHPCG.exe.dir/depend:
	cd /home/zabooke/Project/KokkosHPCG/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zabooke/Project/KokkosHPCG /home/zabooke/Project/KokkosHPCG/src /home/zabooke/Project/KokkosHPCG/build /home/zabooke/Project/KokkosHPCG/build/src /home/zabooke/Project/KokkosHPCG/build/src/CMakeFiles/KokkosHPCG.exe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/KokkosHPCG.exe.dir/depend

