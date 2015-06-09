#ifndef COMPUTERESTRICTION_REF_HPP
#define COMPUTERESTRICTINO_REF_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "KokkosSetup.hpp"
int ComputeRestriction_ref(const SparseMatrix & A, const Vector & rf);
#endif // COMPUTERESTRICTION_REF_HPP
