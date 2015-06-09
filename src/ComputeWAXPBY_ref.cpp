#include "ComputeWAXPBY_ref.hpp"
/*
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
*/
#include <cassert>
#include <Kokkos_Core.hpp>

	class Waxpby {
	private:
	double_1d_type xv;
	double_1d_type yv;
	double_1d_type wv;
	double a;
	double b;

	public:
		Waxpby(double_1d_type xValues, double_1d_type yValues, double_1d_type wValues, double alpha, double beta){
			xv = xValues;
			yv = yValues;	
			wv = wValues;
			a = alpha;
			b = beta;
		}
		KOKKOS_INLINE_FUNCTION
		void operator() (const int& i)const{
			wv(i) = a * xv(i) + b * yv(i);
		}
	};

int ComputeWAXPBY_ref(const local_int_t n, const double alpha, const Vector & x,
	const double beta, const Vector & y, Vector & w) {

	assert(x.localLength >= n);
	assert(y.localLength >= n); 

	double_1d_type xv = x.values;
  double_1d_type yv = y.values;
	double_1d_type wv = w.values;

	if (alpha == 1.0){
		Kokkos::parallel_for(n, [=](const int & i){
			wv(i) = xv(i) + beta * yv(i);
		});}
	else if(beta == 1.0){
		Kokkos::parallel_for(n, [=](const int & i){
			wv(i) = alpha * xv(i) + yv(i);
		});}
	else{
		Kokkos::parallel_for(n, [=](const int & i){
			wv(i) = alpha * xv(i) + beta * yv(i);
		});}
	
	//Kokkos::parallel_for((int)n, Waxpby(x.values, y.values, w.values, alpha, beta));
			
	return 0;
}
