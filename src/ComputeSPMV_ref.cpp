
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
	const local_matrix_type localMatrix;
	const_double_1d_type xv;
	double_1d_type yv;
	
	SPMV(const local_matrix_type &localMatrix_, double_1d_type &xv_, double_1d_type &yv_):
		localMatrix(localMatrix_), xv(xv_), yv(yv_)
		{}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i) const{
		double sum = 0.0;
		int start = localMatrix.graph.row_map(i);
		int end = localMatrix.graph.row_map(i+1);
		for(int j = start; j < end; j++)
			sum += localMatrix.values(j) * xv(localMatrix.graph.entries(j));
		yv(i) = sum;
	}
};

int ComputeSPMV_ref(const SparseMatrix & A, Vector & x, Vector & y){
	
	assert(x.localLength>=A.localNumberOfColumns);
	assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NOMPI
	ExchangeHalo(A,x);
#endif

//	Kokkos::parallel_for(A.localNumberOfRows, SPMV(A.localMatrix, x.values, y.values));
	KokkosSparse::spmv("N", 1.0, A.localMatrix, x.values, 0.0, y.values);
return (0);
}
