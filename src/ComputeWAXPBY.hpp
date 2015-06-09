#ifndef COMPUTEWAXPBY_HPP
#define COMPUTEWAXPBY_HPP
#include "Vector.hpp"
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized);
#endif // COMPUTEWAXPBY_HPP
