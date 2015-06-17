
#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <map>
#include <vector>
#include <cassert>
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"
#include "KokkosSetup.hpp"

using Kokkos::create_mirror_view;
using Kokkos::deep_copy;

struct SparseMatrix_STRUCT {
	char_1d_type title; //!< name of the sparse matrix.
	Geometry * geom; //!< geometry associated with this matrix NO LONGER A POINTER
	global_int_t totalNumberOfRows; //!< total number of matrix rows across all processes
	global_int_t totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes
	local_int_t localNumberOfRows; //!< number of rows local to this process
	local_int_t localNumberOfColumns;  //!< number of columns local to this process
	local_int_t localNumberOfNonzeros;  //!< number of nonzeros local to this process
	char_1d_type nonzerosInRow;  //!< The number of nonzeros in a row will always be 27 or fewer
	double_1d_type matrixDiagonal; //!< Indices of matrix diagonal values.
	std::map< global_int_t, local_int_t > globalToLocalMap; //!< global-to-local mapping
//	std::vector< global_int_t > localToGlobalMap; //!< local-to-global mapping

	local_matrix_type localMatrix; // This is the CrsMatrix that will hold our values and corresponding mtx indices.
	global_matrix_type globalMatrix; // Same values and row_map as local matrix but entries will be the corresponding global mtx indices.
	//!< matrix indices as local values are in localMatrix.graph.entries.

	global_int_1d_type localToGlobalMap;
	mutable bool isDotProductOptimized;
	mutable bool isSpmvOptimized;
	mutable bool isMgOptimized;
	mutable bool isWaxpbyOptimized;
	/*!
	 This is for storing optimized data structres created in OptimizeProblem and
	 used inside optimized ComputeSPMV().
	 */
	mutable struct SparseMatrix_STRUCT * Ac; // Coarse grid matrix 
	mutable MGData * mgData; // Pointer to the coarse level data for this fine matrix
	void * optimizationData;  // pointer that can be used to store implementation-specific data

#ifndef HPCG_NOMPI
	local_int_t numberOfExternalValues; //!< number of entries that are external to this process
	int numberOfSendNeighbors; //!< number of neighboring processes that will be send local data
	local_int_t totalToBeSent; //!< total number of entries to be sent
	local_int_1d_type elementsToSend; //!< elements to send to neighboring processes
	int_1d_type neighbors; //!< neighboring processes
	local_int_1d_type receiveLength; //!< lenghts of messages received from neighboring processes
	local_int_1d_type sendLength; //!< lenghts of messages sent to neighboring processes
	double * sendBuffer; //!< send buffer for non-blocking sends. Still a pointer for EXCHANGEHALO MPI requests. Will change when I figure out how
#endif

	mutable bool isInitialized = false;
};
typedef struct SparseMatrix_STRUCT SparseMatrix;

inline void InitializeSparseMatrix(SparseMatrix & A, Geometry & geom) {
  //A.title = 0;
  A.geom = &geom;
  A.totalNumberOfRows = 0;
  A.totalNumberOfNonzeros = 0;
  A.localNumberOfRows = 0;
  A.localNumberOfColumns = 0;
  A.localNumberOfNonzeros = 0;
  //A.nonzerosInRow = 0;
  //A.mtxIndG = 0;
  //A.matrixDiagonal = 0;

	//TODO This might fix the segmentation fault caused in deleteMatrix
	A.optimizationData = 0;

  // Optimization is ON by default. The code that switches it OFF is in the
  // functions that are meant to be optimized.
  A.isDotProductOptimized = true;
  A.isSpmvOptimized       = true;
  A.isMgOptimized      = true;
  A.isWaxpbyOptimized     = true;

#ifndef HPCG_NOMPI
  A.numberOfExternalValues = 0;
  A.numberOfSendNeighbors = 0;
  A.totalToBeSent = 0;
  //A.elementsToSend = 0;
  //A.neighbors = 0;
  //A.receiveLength = 0;
  //A.sendLength = 0;
  A.sendBuffer = 0;
#endif
  A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
  A.Ac =0; 
	A.isInitialized = true;
  return;
}

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
inline void CopyMatrixDiagonal(SparseMatrix & A, Vector & diagonal){
	assert(A.localNumberOfRows==diagonal.localLength);
	host_const_double_1d_type valuesA = create_mirror_view(A.localMatrix.values);
	host_const_double_1d_type curDiagA = create_mirror_view(A.matrixDiagonal);
	host_double_1d_type dv = create_mirror_view(diagonal.values);
	deep_copy(valuesA, A.localMatrix.values); // Copy the values into the mirror.
	deep_copy(curDiagA, A.matrixDiagonal);
	for(local_int_t i = 0; i < A.localNumberOfRows; ++i) dv(i) = valuesA((int)curDiagA(i));
	deep_copy(diagonal.values, dv);
	return;
}
/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
inline void ReplaceMatrixDiagonal(SparseMatrix & A, Vector & diagonal){
	assert(A.localNumberOfRows == diagonal.localLength);
	host_double_1d_type valuesA = create_mirror_view(A.localMatrix.values);
	host_const_double_1d_type curDiagA = create_mirror_view(A.matrixDiagonal);
	host_const_double_1d_type dv = create_mirror_view(diagonal.values);
	deep_copy(valuesA, A.localMatrix.values);
	deep_copy(curDiagA, A.matrixDiagonal);
	deep_copy(dv, diagonal.values);
	for(local_int_t i = 0; i < A.localNumberOfRows; ++i) valuesA((int)curDiagA(i)) = dv(i);
	deep_copy(A.localMatrix.values, valuesA);
	return;
}
/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteMatrix(SparseMatrix & A) {
//The views should just deallocate themselves once they go out of scope
/*	for (local_int_t i = 0; i< A.localNumberOfRows; ++i) {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndG[i];
    delete [] A.mtxIndL[i];
  }
*/
//  if (A.title)                  delete [] A.title;
//  if (A.nonzerosInRow)             delete [] A.nonzerosInRow;
//  if (A.mtxIndG) delete [] A.mtxIndG;
//  if (A.matrixValues) delete [] A.matrixValues;
//  if (A.matrixDiagonal)           delete [] A.matrixDiagonal;

#ifndef HPCG_NOMPI
//  if (A.elementsToSend)       delete [] A.elementsToSend;
//  if (A.neighbors)              delete [] A.neighbors;
//  if (A.receiveLength)            delete [] A.receiveLength;
//  if (A.sendLength)            delete [] A.sendLength;
  if (A.sendBuffer)            delete [] A.sendBuffer;
#endif

  if (A.geom!=0) { delete A.geom; A.geom = 0;}
  if (A.Ac!=0) { DeleteMatrix(*A.Ac); delete A.Ac; A.Ac = 0;} // Delete coarse matrix
	if (A.mgData!=0) {DeleteMGData(*A.mgData); delete A.mgData; A.mgData = 0;} // Delete MG data

	if(A.optimizationData != 0)
		delete[] (double *) A.optimizationData;
  return;
}

#endif // SPARSEMATRIX_HPP
