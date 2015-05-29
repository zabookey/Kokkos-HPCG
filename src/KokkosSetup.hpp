//KokkosSetup.hpp

/*
This file is for the intention of creating things needed by Kokkos.
*/

#ifndef KOKKOS_SETUP
#define KOKKOS_SETUP

#include <Kokkos_Core.hpp>

typedef Kokkos::View<double *> kokkos_type;
typedef Kokkos::OpenMP execution_space;

#endif
