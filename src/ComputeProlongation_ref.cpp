//Kokkos is not initialezed here since it is initialized in main.
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif

#include "ComputeProlongation_ref.hpp"

#include <Kokkos_Core.hpp>

class Prolongation{
	public:
		Prolongation(kokkos_type xfValues, kokkos_type xcValues, local_int_t * f2cO){
			xfv = xfValues;
			xcv = xcValues;
			f2c = f2cO;
		}

		KOKKOS_INLINE_FUNCTION
		void operator() (const int & i) const{
			xfv(f2c[i]) += xcv(i);
		}
	private:
		kokkos_type xfv;
		kokkos_type xcv;
		local_int_t * f2c;
};

int ComputeProlongation_ref(const SparseMatrix & Af, Vector & xf) {
	local_int_t nc = Af.mgData->rc->localLength;

	kokkos_type xfv = xf.values;
	kokkos_type xcv = Af.mgData->xc->values;
	local_int_t * f2c = Af.mgData->f2cOperator;

	Kokkos::parallel_for(nc, [=](const int i){
		xfv(f2c[i]) += xcv(i);
	});
/*
	Kokkos::parallel_for((int) nc, Prolongation(xf.values, Af.mgData->xc->values, Af.mgData->f2cOperator));
*/
	return (0);
}
