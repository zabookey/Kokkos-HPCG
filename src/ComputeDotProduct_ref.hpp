#ifndef COMPUTEDOTPRODUCT_REF_HPP
#define COMPUTEDOTPRODUCT_REF_HPP
#include "Vector.hpp"
#include "KokkosSetup.hpp"
int ComputeDotProduct_ref(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce);

#endif // COMPUTEDOTPRODUCT_REF_HPP
