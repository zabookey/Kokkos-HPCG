
#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cassert>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include "Geometry.hpp"
#include "KokkosSetup.hpp"

using Kokkos::create_mirror_view;

struct Vector_STRUCT {
  local_int_t localLength;  //!< length of local portion of the vector
  double_1d_type values;          //!< view of values
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;

	bool isInitialized = false; // Use this variable instead of checking if the address = 0.

};
typedef struct Vector_STRUCT Vector;

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
inline void InitializeVector(Vector & v, local_int_t localLength) {
  v.localLength = localLength;
  v.values = double_1d_type("Vector Values", localLength);
  v.optimizationData = 0;
	v.isInitialized = true; //Use this variable only when looking to be deleted. MIGHT NOT EVEN BE NECESSARY...
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
//This whole method may be redundant since views are automatically initalized with zeros.
inline void ZeroVector(Vector & v){
	local_int_t localLength = v.localLength;
	host_double_1d_type vv = create_mirror_view(v.values);
	for(int i = 0; i < localLength; ++i) vv(i) = 0.0;
	Kokkos::deep_copy(v.values, vv);
	return;
}
/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
inline void ScaleVectorValue(Vector & v, local_int_t index, double value) {
  assert(index>=0 && index < v.localLength);
  host_double_1d_type vv = create_mirror_view(v.values);
	vv(index) *= value;
	Kokkos::deep_copy(v.values, vv);
	return;
}
/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
inline void FillRandomVector(Vector & v){
	local_int_t localLength = v.localLength;
	host_double_1d_type vv = create_mirror_view(v.values);
	for(int i = 0; i < localLength; ++i) vv(i) = rand() / (double)(RAND_MAX) + 1.0;
	Kokkos::deep_copy(v.values, vv);
	return;
}
/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
inline void CopyVector(const Vector & v, Vector & w) {
	local_int_t localLength = v.localLength;
	assert(w.localLength >= localLength);
	host_const_double_1d_type vv = create_mirror_view(v.values);
	host_double_1d_type wv = create_mirror_view(w.values);
	for(int i = 0; i < localLength; ++i) wv(i) = vv(i);
	w.isInitialized = v.isInitialized;
	Kokkos::deep_copy(w.values, wv);
	return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteVector(Vector & v) {

  v.localLength = 0;
	v.isInitialized = false;
  return;
}

#endif // VECTOR_HPP
