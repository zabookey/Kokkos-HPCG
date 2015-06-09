#ifndef COMPUTESYMGS_REF_HPP
#define COMPUTESYMGS_REF_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

int ComputeSYMGS_ref(const SparseMatrix & A, const Vector & r, Vector & x);

#endif // COMPUTESYMGS_REF_HPP
