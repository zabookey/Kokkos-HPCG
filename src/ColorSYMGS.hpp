#ifndef COLORSYMGS_HPP
#define COLORSYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

int ColorSYMGS(const SparseMatrix & A, const Vector & x, Vector & y);

#endif