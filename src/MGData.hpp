
//@HEADER
// ***************************************************
//
// Kokkos HPCG: Refactored code to replace most/all
// allocated arrays with views.
// 
// 
// 
// 
//
// ***************************************************
//@HEADER

/*!
 @file MGData.hpp

 HPCG data structure
 */

#ifndef MGDATA_HPP
#define MGDATA_HPP

#include <cassert>
#include "SparseMatrix.hpp"
#include "Vector.hpp"

//#include "KokkosSetup.hpp"

struct MGData_STRUCT {
  int numberOfPresmootherSteps; // Call ComputeSYMGS this many times prior to coarsening
  int numberOfPostsmootherSteps; // Call ComputeSYMGS this many times after coarsening
  local_int_1d_type f2cOperator; //!< 1D array containing the fine operator local IDs that will be injected into coarse space.
  Vector * rc; // coarse grid residual vector
  Vector * xc; // coarse grid solution vector
  Vector * Axf; // fine grid residual vector
  /*!
   This is for storing optimized data structres created in OptimizeProblem and
   used inside optimized ComputeSPMV().
   */
  void * optimizationData;

};
typedef struct MGData_STRUCT MGData;

/*!
 Constructor for the data structure of CG vectors.

 @param[in] Ac - Fully-formed coarse matrix
 @param[in] f2cOperator -
 @param[out] data the data structure for CG vectors that will be allocated to get it ready for use in CG iterations
 */
inline void InitializeMGData(local_int_1d_type f2cOperator, Vector & rc, Vector & xc, Vector & Axf, MGData & data) {
  data.numberOfPresmootherSteps = 1;
  data.numberOfPostsmootherSteps = 1;
  data.f2cOperator = f2cOperator; // Space for injection operator
  data.rc = &rc;
  data.xc = &xc;
  data.Axf = &Axf;
  return;
}

/*!
 Destructor for the CG vectors data.

 @param[inout] data the MG data structure whose storage is deallocated
 */
inline void DeleteMGData(MGData & data) {

	DeleteVector(*data.Axf);
	DeleteVector(*data.rc);
	DeleteVector(*data.xc);
	delete data.Axf;
	delete data.rc;
	delete data.xc;
  return;
}

#endif // MGDATA_HPP
