
#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cassert>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include "Geometry.hpp"
#include "KokkosSetup.hpp"

using Kokkos::create_mirror_view;
using Kokkos::deep_copy;

struct Vector_STRUCT {
  local_int_t localLength;  //!< length of local portion of the vector
  double_1d_type values;          //!< view of values
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;
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
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
//This whole method may be redundant since views are automatically initalized with zeros.
inline void ZeroVector(Vector & v){
	deep_copy(v.values, 0.0);	// This should zero out v.values.
	return;
}
/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
KOKKOS_INLINE_FUNCTION
void ScaleVectorValue(const Vector & v, local_int_t index, double value) {
  assert(index>=0 && index < v.localLength);
	double_1d_type vv = v.values;
	vv(index) *= value;
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
	deep_copy(v.values, vv);
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
	// Because it's not a guarantee that v and w are the same length I can't just deep_copy.
	host_const_double_1d_type vv = create_mirror_view(v.values);
	host_double_1d_type wv = create_mirror_view(w.values);
	deep_copy(vv, v.values);
	deep_copy(wv, w.values); //Have to copy the whole thing in case w is longer than v, this way we retain the end that v can't touch.
	for(int i = 0; i < localLength; ++i) wv(i) = vv(i);
	Kokkos::deep_copy(w.values, wv);
	return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteVector(Vector & v) {

  v.localLength = 0;
  return;
}

#endif // VECTOR_HPP
