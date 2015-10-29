#ifndef LEVELSYMGS_HPP
#define LEVELSYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

int LevelSYMGS(const SparseMatrix & A, const Vector & x, Vector & y);

#endif