#ifndef GENERATEPROBLEM_HPP
#define GENERATEPROBLEM_HPP
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "KokkosSetup.hpp"

void GenerateProblem(SparseMatrix & A, Vector & b, Vector & x, Vector & xexact);

void GenerateProblem(SparseMatrix & A);

#endif // GENERATEPROBLEM_HPP
