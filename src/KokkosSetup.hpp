//KokkosSetup.hpp

/*
This file is for the intention of creating things needed by Kokkos.
*/

#ifndef KOKKOS_SETUP
#define KOKKOS_SETUP

#include <Kokkos_Core.hpp>
#include "Geometry.hpp" // Just so we have the local_int_t and global_int_t definitions.

//TODO Find a way to change this at compile time.
typedef Kokkos::OpenMP execution_space;

//Normal types.
typedef Kokkos::View<int *> int_1d_type;
typedef Kokkos::View<double *> double_1d_type;
typedef Kokkos::View<double **> double_2d_type;
typedef Kokkos::View<local_int_t *> local_int_1d_type;
typedef Kokkos::View<local_int_t **> local_int_2d_type;
typedef Kokkos::View<global_int_t *> global_int_1d_type;
typedef Kokkos::View<global_int_t **> global_int_2d_type;
typedef Kokkos::View<char *> char_1d_type;
//Const values
typedef Kokkos::View<const double *> const_double_1d_type;
typedef Kokkos::View<const double **> const_double_2d_type;
typedef Kokkos::View<const local_int_t *> const_local_int_1d_type;
typedef Kokkos::View<const global_int_t **> const_global_int_2d_type;
typedef Kokkos::View<const char *> const_char_1d_type;
//Mirrors
typedef int_1d_type::HostMirror host_int_1d_type;
typedef double_1d_type::HostMirror host_double_1d_type;
typedef double_2d_type::HostMirror host_double_2d_type;
typedef local_int_1d_type::HostMirror host_local_int_1d_type;
typedef local_int_2d_type::HostMirror host_local_int_2d_type;
typedef global_int_1d_type::HostMirror host_global_int_1d_type;
typedef global_int_2d_type::HostMirror host_global_int_2d_type;
typedef char_1d_type::HostMirror host_char_1d_type;
//Const Mirrors
typedef const_double_1d_type::HostMirror host_const_double_1d_type;
typedef const_double_2d_type::HostMirror host_const_double_2d_type;
typedef const_local_int_1d_type::HostMirror host_const_local_int_1d_type;
typedef const_global_int_2d_type::HostMirror host_const_global_int_2d_type;
typedef const_char_1d_type::HostMirror host_const_char_1d_type;
#endif
