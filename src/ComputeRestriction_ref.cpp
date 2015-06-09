/*
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
*/
#include "ComputeRestriction_ref.hpp"

#include <Kokkos_Core.hpp>

class Restriction{
	public:	
		double_1d_type Axfv;
		double_1d_type rfv;
		double_1d_type rcv;
		local_int_1d_type f2c;

		Restriction(double_1d_type AxfValues, double_1d_type rfValues, double_1d_type rcValues, local_int_1d_type f2cO){
			Axfv = AxfValues;
			rfv = rfValues;
			rcv = rcValues;
			f2c = f2cO;
		}

		KOKKOS_INLINE_FUNCTION
		void operator() (const int & i) const {
			rcv(i) = rfv(f2c(i)) - Axfv(f2c(i));
		}

};

int ComputeRestriction_ref(const SparseMatrix & A, const Vector & rf){
	local_int_t nc = A.mgData->rc->localLength;

	double_1d_type Axfv = A.mgData->Axf->values;
	double_1d_type rfv = rf.values;
	double_1d_type rcv = A.mgData->rc->values;
	local_int_1d_type f2c = A.mgData->f2cOperator;

	Kokkos::parallel_for(nc, Restriction(Axfv, rfv, rcv, f2c));

	return (0);
}
