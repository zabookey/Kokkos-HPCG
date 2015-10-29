#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

int ComputeSYMGS(const SparseMatrix & A, const Vector & x, Vector & y);

#endif
