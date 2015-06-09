#ifndef COMPUTEMG_REF_HPP
#define COMPUTEMG_REF_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeMG_ref(const SparseMatrix  & A, const Vector & r, Vector & x);

#endif // COMPUTEMG_REF_HPP
