#ifndef COMPUTEMG_HPP
#define COMPUTEMG_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x);

#endif // COMPUTEMG_HPP
