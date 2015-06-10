#include "ComputeWAXPBY_ref.hpp"
/*
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
*/
#include <cassert>
#include <Kokkos_Core.hpp>

	class Waxpby {
		public:
		const_double_1d_type xv;
		const_double_1d_type yv;
		double_1d_type wv;
		double alpha;
		double beta;

		Waxpby(const double_1d_type &xv_,const double_1d_type &yv_, double_1d_type &wv_,const double alpha_,const double beta_):
			xv(xv_), yv(yv_), wv(wv_), alpha(alpha_), beta(beta_)
			{}
		KOKKOS_INLINE_FUNCTION
		void operator() (const int& i)const{
			wv(i) = alpha * xv(i) + beta * yv(i);
		}
	};


	class AlphaOne {
		public:
		const_double_1d_type xv;
		const_double_1d_type yv;
		double_1d_type wv;
		double beta;
	
		AlphaOne(const double_1d_type &xv_, const double_1d_type &yv_, double_1d_type &wv_, const double beta_):
			xv(xv_), yv(yv_), wv(wv_), beta(beta_)
			{}

		KOKKOS_INLINE_FUNCTION
		void operator() (const int& i)const{
			wv(i) = xv(i) + beta * yv(i);
		}
	};

	class BetaOne {
		public:
		const_double_1d_type xv;
		const_double_1d_type yv;
		double_1d_type wv;
		double alpha;
	
		BetaOne(const double_1d_type &xv_, const double_1d_type &yv_, double_1d_type &wv_, const double alpha_):
			xv(xv_), yv(yv_), wv(wv_), alpha(alpha_)
			{}

		KOKKOS_INLINE_FUNCTION
		void operator() (const int& i)const{
			wv(i) = alpha * xv(i) + yv(i);
		}
	};


int ComputeWAXPBY_ref(const local_int_t n, const double alpha, const Vector & x,
	const double beta, const Vector & y, Vector & w) {

	assert(x.localLength >= n);
	assert(y.localLength >= n); 

	double_1d_type xv = x.values;
  double_1d_type yv = y.values;
	double_1d_type wv = w.values;

	if (alpha == 1.0)
		Kokkos::parallel_for(n, AlphaOne(xv, yv, wv, beta));
	else if(beta == 1.0)
		Kokkos::parallel_for(n, BetaOne(xv, yv, wv, alpha));
	else
		Kokkos::parallel_for(n, Waxpby(xv, yv, wv, alpha, beta));
	
			
	return 0;
}
