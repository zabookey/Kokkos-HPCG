#ifndef COMPUTESPMV_HPP
#define COMPUTESPMV_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "KokkosSetup.hpp"

int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y);

#endif  // COMPUTESPMV_HPP
