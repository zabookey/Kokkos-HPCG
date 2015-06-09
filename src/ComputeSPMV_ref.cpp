
#include "ComputeSPMV_ref.hpp"

#ifndef HPCG_NOMPI
#include "ExchangeHalo.hpp"
#endif
/*
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
*/
#include <cassert>

#include <Kokkos_Core.hpp>

class SPMV{
	private:
	SparseMatrix mat;
	double_1d_type xValues;
	double_1d_type yValues;
	
	public:
	SPMV(const SparseMatrix & A, double_1d_type xv, double_1d_type yv){
		mat = A;
		xValues = xv;
		yValues = yv;
	}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int& i) const{
		double sum = 0.0;
	// TODO These should use auto.
		const const_double_1d_type cur_vals = Kokkos::subview(mat.matrixValues, i, Kokkos::ALL());
		const const_local_int_1d_type cur_inds = Kokkos::subview(mat.mtxIndL, i, Kokkos::ALL());
		const int cur_nnz = mat.nonzerosInRow(i);

		for(int j = 0; j < cur_nnz; j++)
			sum += cur_vals(j)*xValues(cur_inds(j));
		yValues(i) = sum;
	}
};

int ComputeSPMV_ref(const SparseMatrix & A, Vector & x, Vector & y){
	
	assert(x.localLength>=A.localNumberOfColumns);
	assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NOMPI
	ExchangeHalo(A,x);
#endif
	const local_int_t nrow = A.localNumberOfRows;
	double_1d_type xv = x.values;
	double_1d_type yv = y.values;
/*
	Kokkos::parallel_for(nrow, [=](const int&i){
		double sum = 0;
		const double * const cur_vals = A.matrixValues[i];
		const local_int_t * const cur_inds = A.mtxIndL[i];
		const int cur_nnz = A.nonzerosInRow[i];
		for(int j = 0; j < cur_nnz; j++){
			sum += cur_vals[j]*xv(cur_inds[j]);
		}
		yv(i) = sum;
	});
*/
	Kokkos::parallel_for((int) nrow, SPMV(A, xv, yv));

return (0);
}
