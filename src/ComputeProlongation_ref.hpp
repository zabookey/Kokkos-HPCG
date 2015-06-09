#ifndef COMPUTEPROLONGATION_REF_HPP
#define COMPUTEPROLONGATION_REF_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "KokkosSetup.hpp"
int ComputeProlongation_ref(const SparseMatrix & Af, Vector & xf);
#endif // COMPUTEPROLONGATION_REF_HPP
