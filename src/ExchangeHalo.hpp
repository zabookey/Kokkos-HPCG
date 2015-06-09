#ifndef EXCHANGEHALO_HPP
#define EXCHANGEHALO_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"
void ExchangeHalo(const SparseMatrix & A, Vector & x);
#endif // EXCHANGEHALO_HPP
