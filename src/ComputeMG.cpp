#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"

#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include <cassert>
#include <iostream>
/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  A.isMgOptimized = false;
	ZeroVector(x);

	int ierr = 0;
	if(A.mgData!=0){
		int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
		for(int i = 0; i < numberOfPresmootherSteps; ++i) ierr+= ComputeSYMGS(A, r, x);
		if(ierr != 0) return (ierr);
		ierr = ComputeSPMV(A, x, *A.mgData->Axf); if(ierr!=0) return(ierr);
		ierr = ComputeRestriction_ref(A, r); if(ierr!=0) return(ierr);
		ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc); if(ierr != 0) return(ierr);
		ierr = ComputeProlongation_ref(A,x); if(ierr!=0) return (ierr);
		int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
		for(int i = 0; i < numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
		if(ierr!=0) return (ierr);
	}
	else{
		ierr = ComputeSYMGS(A,r,x);
		if(ierr!=0) return (ierr);
	}
	return(0);

}
