#ifndef COMPUTEWAXPBY_REF_HPP
#define COMPUTEWAXPBY_REF_HPP
#include "Vector.hpp"
#include "KokkosSetup.hpp"
int ComputeWAXPBY_ref(const local_int_t n, const double alpha, const Vector & x,
		const double beta, const Vector & y, Vector & w);
#endif // COMPUTEWAXPBY_REF_HPP
