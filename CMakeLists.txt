cmake_minimum_required(VERSION 2.8)

#These next few lines are ripped from the trilinos demo almost exactly
set(Trilinos_PREFIX
"/home/zabooke/Project/Trilinos_builds/trilinos_kokkos_tpetra/build")
set(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})

message("Trilinos prefix: ${Trilinos_PREFIX}")

find_package(Trilinos REQUIRED)

#Use same compilers and flags as Trilinos
set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
message("Compiler: ${CMAKE_CXX_COMPILER}")
set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
set(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER})

project(KokkosHPCG)

set(CMAKE_CXX_FLAGS "${Trilinos_CXX_COMPILER_FLAGS} -pg")
set(CMAKE_C_FLAGS  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_Fortran_FLAGS  "${Trilinos_Fortran_COMPILER_FLAGS} ${CMAKE_Fortran_FLAGS}")

# MPI check 
MESSAGE("-- Checking if MPI is enabled in Trilinos:")
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(HPCG_NOMPI OFF)
ELSE()
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(HPCG_NOMPI ON)
  ADD_DEFINITIONS(-DHPCG_NOMPI)
ENDIF()

set(HPCG_NOOPENMP OFF)
ADD_DEFINITIONS(-DHPCG_NOMPI)

# KokkosCore check will exit if Kokkos is not found
message("--Looking for Kokkos:") 
list(FIND Trilinos_PACKAGE_LIST Kokkos Kokkos_List_ID)
IF (Kokkos_List_ID GREATER -1)
  message("--Looking for Kokkos: -- found. Proceeding...")
ELSE()
  message(FATAL_ERROR "Kokkos not found... CMake exiting")
ENDIF()

# Change this definition to specify where to execute the code.
# -DHPCG_Kokkos_OpenMP
# -DHPCG_Kokkos_Cuda
# -DHPCG_Kokkos_Serial (Default)
ADD_DEFINITIONS(-DHPCG_Kokkos_OpenMP)

# Change this definition to specify which Gauss Seidel to use
# -DSYMGS_COLOR Solve using graph Colouring. (Doesn't Produce valid results)
# -DSYMGS_LEVEL Level solve.
# -DSYMGS_INEXACT Jacobi Trisolve. (Only works if you use enough iterations
#		for your problem size.)
# Default is serial Gauss Seidel used in Vanilla HPCG
# If nothing is specified the defualt serial Gauss Seidel will be used.
ADD_DEFINITIONS(-DSYMGS_LEVEL)

include_directories("${Trilinos_PREFIX}/include")
#This will run the cmakelists in the src directory. This is where I want things to compile.
add_subdirectory(src)


