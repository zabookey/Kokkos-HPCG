//KokkosSetup.hpp

/*
This file is for the intention of creating things needed by Kokkos.
*/

#ifndef KOKKOS_SETUP
#define KOKKOS_SETUP

#include <Kokkos_Core.hpp>
#include <Kokkos_Sparse.hpp>
#include "Kokkos_UnorderedMap.hpp"
#include "Geometry.hpp" // Just so we have the local_int_t and global_int_t definitions.

//TODO Find a way to change this at compile time.
typedef Kokkos::Serial execution_space;
#ifdef HPCG_Kokkos_OpenMP
typedef Kokkos::OpenMP execution_space;
#endif
#ifdef HPCG_Kokkos_Cuda
typedef Kokkos::Cuda execution_space;
#endif
//View typedefs.
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
typedef Kokkos::View<const local_int_t **> const_local_int_2d_type;
typedef Kokkos::View<const global_int_t *> const_global_int_1d_type;
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
typedef const_local_int_2d_type::HostMirror host_const_local_int_2d_type;
typedef const_global_int_1d_type::HostMirror host_const_global_int_1d_type;
typedef const_global_int_2d_type::HostMirror host_const_global_int_2d_type;
typedef const_char_1d_type::HostMirror host_const_char_1d_type;
//CrsMatrix typedefs
//CrsMatrix types
typedef Kokkos::CrsMatrix<double, local_int_t, execution_space> local_matrix_type;
typedef Kokkos::CrsMatrix<double, global_int_t, execution_space> global_matrix_type;
typedef local_matrix_type::values_type values_type; // View for matrix values, similar to double_1d_type.
typedef local_matrix_type::index_type local_index_type; // View for column Indices, similar to local_int_1d_type.
typedef global_matrix_type::index_type global_index_type;
typedef local_matrix_type::row_map_type row_map_type; //View for row_map, similar to const_local_int_1d_type.
typedef local_matrix_type::StaticCrsGraphType StaticCrsGraphType; // The graph type held by matrix_type.
typedef global_matrix_type::StaticCrsGraphType globalStaticCrsGraphType;
typedef Kokkos::View<StaticCrsGraphType::size_type *, StaticCrsGraphType::array_layout,
	StaticCrsGraphType::device_type> non_const_row_map_type; // Used specifically for setup of row_map_type.
//CrsMatrix Mirrors
typedef local_matrix_type::HostMirror host_matrix_type; // I don't think this will get used. Never used in Practice
typedef values_type::HostMirror host_values_type;
typedef local_index_type::HostMirror host_local_index_type;
typedef global_index_type::HostMirror host_global_index_type;
typedef row_map_type::HostMirror host_row_map_type;
typedef StaticCrsGraphType::HostMirror host_StaticCrsGraphType;
typedef non_const_row_map_type::HostMirror host_non_const_row_map_type;
//UnorderedMap typedefs.
typedef Kokkos::UnorderedMap<global_int_t, local_int_t, execution_space> map_type;
typedef map_type::HostMirror host_map_type;
#endif
