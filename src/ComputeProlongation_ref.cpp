/*
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
*/
#include "ComputeProlongation_ref.hpp"

#include <Kokkos_Core.hpp>

class Prolongation{
	public:
		double_1d_type xfv;
		double_1d_type xcv;
		local_int_1d_type f2c;

		Prolongation(double_1d_type &xfValues, double_1d_type &xcValues, local_int_1d_type &f2cO){
			xfv = xfValues;
			xcv = xcValues;
			f2c = f2cO;
		}

		KOKKOS_INLINE_FUNCTION
		void operator() (const int & i) const{
			xfv(f2c(i)) += xcv(i);
		}
};

int ComputeProlongation_ref(const SparseMatrix & Af, Vector & xf) {
	local_int_t nc = Af.mgData->rc->localLength;

	Kokkos::parallel_for(nc, Prolongation(xf.values, Af.mgData->xc->values, Af.mgData->f2cOperator));

	return (0);
}
