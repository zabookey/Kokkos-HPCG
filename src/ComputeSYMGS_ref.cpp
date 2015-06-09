
#ifndef HPCG_NOMPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS_ref.hpp"
#include <cassert>

using Kokkos::create_mirror_view;
using Kokkos::subview;
using Kokkos::ALL;
/*!
  Computes one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.
  - We then perform one back sweep.
  - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.


  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSYMGS
*/
int ComputeSYMGS_ref(const SparseMatrix & A, const Vector & r, Vector & x){

	assert(x.localLength == A.localNumberOfColumns); // Make sure x contains space for halo values

#ifndef HPCG_NOMPI
	ExchangeHalo(A,x);
#endif

	const local_int_t nrow = A.localNumberOfRows;
	host_double_1d_type matrixDiagonal = create_mirror_view(A.matrixDiagonal); //Host Mirror to A.matrixDiagonal.
	host_const_double_1d_type rv = create_mirror_view(r.values); // Host Mirror to r.values
	const host_double_1d_type xv = create_mirror_view(x.values); // Host Mirror to x.values
	const host_const_char_1d_type nonZerosInRow = create_mirror_view(A.nonzerosInRow); // Host Mirror to A.nonZerosInRow.
//Easier to Mirror it once than mirror in every iteration

	for(local_int_t i = 0; i < nrow; i++){
		const host_const_double_1d_type currentValues = create_mirror_view(subview(A.matrixValues, i, ALL()));
		const host_const_local_int_1d_type currentColIndices = create_mirror_view(subview(A.mtxIndL, i, ALL()));
		const int currentNumberOfNonzeros = nonZerosInRow(i);
		const double currentDiagonal = currentValues((int)matrixDiagonal(i)); //This works only if matrixDiagonal is the indices of the diagonal and not the value.If its the values remove currentValues( ).
		double sum = rv(i);

		for(int j = 0; j < currentNumberOfNonzeros; j++){
			local_int_t curCol = currentColIndices(j);
			sum -= currentValues(j) * xv(curCol);
		}
		sum += xv(i)*currentDiagonal;

		xv(i) = sum/currentDiagonal;
	}

	// Now the back sweep.

	for(local_int_t i = nrow-1; i >= 0; i--){
		const host_const_double_1d_type currentValues = create_mirror_view(subview(A.matrixValues, i, ALL()));
		const host_const_local_int_1d_type currentColIndices = create_mirror_view(subview(A.mtxIndL, i, ALL()));
		const int currentNumberOfNonzeros = nonZerosInRow(i);
		const double currentDiagonal = currentValues((int)matrixDiagonal(i)); // Same reason as the last loop. Change if needed.
		double sum = rv(i);

		for(int j = 0; j < currentNumberOfNonzeros; j++){
			local_int_t curCol = currentColIndices(j);
			sum -= currentValues(j) * xv(curCol);
		}
		sum += xv(i)*currentDiagonal;

		xv(i) = sum/currentDiagonal;
	}

	Kokkos::deep_copy(x.values, xv); // Copy the updated xv on the host back to x.values.
	// All of the mirrors should go out of scope here and deallocate themselves.
	return(0);
}
