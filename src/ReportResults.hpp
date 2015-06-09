#ifndef REPORTRESULTS_HPP
#define REPORTRESULTS_HPP
#include "SparseMatrix.hpp"
#include "TestCG.hpp"
#include "TestSymmetry.hpp"
#include "TestNorms.hpp"

void ReportResults(const SparseMatrix & A, int numberOfMgLevels, int numberOfCgSets, int refMaxIters, int optMaxIters, double times[],
    const TestCGData & testcg_data, const TestSymmetryData & testsymmetry_data, const TestNormsData & testnorms_data, int global_failure);

#endif // REPORTRESULTS_HPP
