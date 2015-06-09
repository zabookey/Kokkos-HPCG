#ifndef WRITEPROBLEM_HPP
#define WRITEPROBLEM_HPP
#include "Geometry.hpp"
#include "SparseMatrix.hpp"
#include "KokkosSetup.hpp"

int WriteProblem(const Geometry & geom, const SparseMatrix & A, const Vector b, const Vector x, const Vector xexact);
#endif // WRITEPROBLEM_HPP
