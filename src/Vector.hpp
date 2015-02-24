// This will be the updated vector class which contains views. 
//Will create a copy directory of HPCG so I can maintain a working HPCG while I modify this one.

#ifndef VECTOR_HPP
#define VECTOR_HPP
#include <cassert>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include "Geometry.hpp"
#include "KokkosSetup.hpp"

struct Vector_STRUCT {
  local_int_t localLength;  //!< length of local portion of the vector
  kokkos_type values;          //!< view of values
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
  v.values = kokkos_type("A", localLength);
  v.optimizationData = 0;
  return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
inline void ZeroVector(Vector & v) {
  local_int_t localLength = v.localLength;
  kokkos_type vv = v.values;
  for (int i=0; i<localLength; ++i) vv(i) = 0.0;
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
  kokkos_type vv = v.values;
  vv(index) *= value;
  return;
}
/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
inline void FillRandomVector(Vector & v) {
  local_int_t localLength = v.localLength;
  kokkos_type vv = v.values;
  for (int i=0; i<localLength; ++i) vv(i) = rand() / (double)(RAND_MAX) + 1.0;
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
  kokkos_type vv = v.values;
  kokkos_type wv = w.values;
  for (int i=0; i<localLength; ++i) wv(i) = vv(i);
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
