//KokkosSetup.hpp

/*
This file is for the intention of creating things needed by Kokkos.
*/

#ifndef KOKKOS_SETUP
#define KOKKOS_SETUP

#include <Kokkos_Core.hpp>

//Find a way to change this at compile time.
typedef Kokkos::OpenMP execution_space;
typedef Kokkos::View<double *> kokkos_type;
#endif
