//Kokkos is not initialezed here since it is initialized in main.
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif

#include "ComputeRestriction_ref.hpp"

#include <Kokkos_Core.hpp>

class Restriction{
	public:
		Restriction(kokkos_type AxfValues, kokkos_type rfValues, kokkos_type rcValues, local_int_t * f2cO){
			Axfv = AxfValues;
			rfv = rfValues;
			rcv = rcValues;
			f2c = f2cO;
		}
		KOKKOS_INLINE_FUNCTION
		void operator() (const int & i) const {
			rcv(i) = rfv(f2c[i]) - Axfv(f2c[i]);
		}

	private:
		kokkos_type Axfv;
		kokkos_type rfv;
		kokkos_type rcv;
		local_int_t * f2c;
};

int ComputeRestriction_ref(const SparseMatrix & A, const Vector & rf){
	local_int_t nc = A.mgData->rc->localLength;

	kokkos_type Axfv = A.mgData->Axf->values;
	kokkos_type rfv = rf.values;
	kokkos_type rcv = A.mgData->rc->values;
	local_int_t * f2c = A.mgData->f2cOperator;

	Kokkos::parallel_for(nc, [&](const int &i){
		rcv(i) = rfv(f2c[i]) - Axfv(f2c[i]);
	});
/*
	Kokkos::parallel_for((int) nc, Restriction(A.mgData->Axf->values, rf.values, A.mgData->rc->values, A.mgData->f2cOperator));
*/
	return (0);
}
