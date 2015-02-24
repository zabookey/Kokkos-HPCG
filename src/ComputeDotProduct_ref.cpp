//Kokkos is not initialezed here since it is initialized in main.
#ifndef HPCG_NOMPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
#include <cassert>
#include <Kokkos_Core.hpp>
#include "ComputeDotProduct_ref.hpp"

	class Dotproduct {

		private:
		kokkos_type  xv;
		kokkos_type  yv;

		public:
		typedef double value_type;
		Dotproduct(kokkos_type  xValues, kokkos_type  yValues){
			xv = xValues;
			yv = yValues;
		}
		KOKKOS_INLINE_FUNCTION
		void operator()(local_int_t i, double &final)const{
		final += xv(i) * yv(i);
		}

	};

int ComputeDotProduct_ref(const local_int_t n, const Vector & x, const Vector & y,
		 double & result, double & time_allreduce){
	assert(x.localLength >= n);
	assert(y.localLength >= n);

	double local_result = 0.0;
	Kokkos::parallel_reduce(n, Dotproduct(x.values, y.values), local_result);

	#ifndef HPCG_NOMPI
		double t0 = mytimer();
		double global_result = 0.0;
		MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		result = global_result;
		time_allreduce += mytimer() - t0;
	#else
		result = local_result;
	#endif

	return (0);
}

