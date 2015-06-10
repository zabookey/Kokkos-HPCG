
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
	public:
	const_double_2d_type matrixValues;
	const_local_int_2d_type mtxIndL;
	const_char_1d_type nonzerosInRow;
	double_1d_type xv;
	double_1d_type yv;
	
	SPMV(const const_double_2d_type &matrixValues_,const const_local_int_2d_type &mtxIndL_, 
		const const_char_1d_type &nonzerosInRow_, double_1d_type &xv_, double_1d_type &yv_):
		matrixValues(matrixValues_), mtxIndL(mtxIndL_), nonzerosInRow(nonzerosInRow_), xv(xv_), yv(yv_)
		{}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i) const{
		double sum = 0.0;
		const auto cur_vals = Kokkos::subview(matrixValues, i, Kokkos::ALL());
		const auto cur_inds = Kokkos::subview(mtxIndL, i, Kokkos::ALL());
		const int cur_nnz = nonzerosInRow(i);
		for(int j = 0; j < cur_nnz; j++)
			sum += cur_vals(j) * xv(cur_inds(j));
		yv(i) = sum;
	}
};

int ComputeSPMV_ref(const SparseMatrix & A, Vector & x, Vector & y){
	
	assert(x.localLength>=A.localNumberOfColumns);
	assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NOMPI
	ExchangeHalo(A,x);
#endif

	Kokkos::parallel_for(A.localNumberOfRows, SPMV(A.matrixValues, A.mtxIndL, A.nonzerosInRow, x.values, y.values));

return (0);
}
