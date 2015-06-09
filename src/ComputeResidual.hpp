
#ifndef COMPUTERESIDUAL_HPP
#define COMPUTERESIDUAL_HPP
#include "Vector.hpp"
#include "KokkosSetup.hpp"
int ComputeResidual(const local_int_t n, const Vector & v1, const Vector & v2, double & residual);
#endif // COMPUTERESIDUAL_HPP
