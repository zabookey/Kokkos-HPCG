#ifndef INEXACTSYMGS_HPP
#define INEXACTSYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

int InexactSYMGS(const SparseMatrix & A, const Vector & x, Vector & y);

#endif