#ifndef COMPUTESPMV_REF_HPP
#define COMPUTESPMV_REF_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "KokkosSetup.hpp"

int ComputeSPMV_ref(const SparseMatrix & A, Vector & x, Vector & y);

#endif // COMPUTESPMV_REF_HPP
