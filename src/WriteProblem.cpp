/*!
 @file WriteProblem.cpp

 HPCG routine
 */

#include <cstdio>
#include "WriteProblem.hpp"
#include "Geometry.hpp"

using Kokkos::create_mirror_view;
using Kokkos::deep_copy;
using Kokkos::subview;
using Kokkos::ALL;
/*!
  Routine to dump:
   - matrix in row, col, val format for analysis with MATLAB
   - x, xexact, b as simple arrays of numbers.

   Writes to A.dat, x.dat, xexact.dat and b.dat, respectivly.

   NOTE:  THIS CODE ONLY WORKS ON SINGLE PROCESSOR RUNS

   Read into MATLAB using:

       load A.dat
       A=spconvert(A);
       load x.dat
       load xexact.dat
       load b.dat

  @param[in] geom   The description of the problem's geometry.
  @param[in] A      The known system matrix
  @param[in] b      The known right hand side vector
  @param[in] x      The solution vector computed by CG iteration
  @param[in] xexact Generated exact solution

  @return Returns with -1 if used with more than one MPI process. Returns with 0 otherwise.

  @see GenerateProblem
*/
int WriteProblem( const Geometry & geom, const SparseMatrix & A,
    const Vector b, const Vector x, const Vector xexact) {

  if (geom.size!=1) return(-1); //TODO Only works on one processor.  Need better error handler
  const global_int_t nrow = A.totalNumberOfRows;

  FILE * fA = 0, * fx = 0, * fxexact = 0, * fb = 0;
  fA = fopen("A.dat", "w");
  fx = fopen("x.dat", "w");
  fxexact = fopen("xexact.dat", "w");
  fb = fopen("b.dat", "w");

  if (! fA || ! fx || ! fxexact || ! fb) {
    if (fb) fclose(fb);
    if (fxexact) fclose(fxexact);
    if (fx) fclose(fx);
    if (fA) fclose(fA);
    return -1;
  }
	//Mirrors and subviews! But mostly just mirrors here.
	const host_const_double_2d_type host_matrixValues = create_mirror_view(A.matrixValues);
	const host_const_global_int_2d_type host_mtxIndG = create_mirror_view(A.mtxIndG);
	const host_const_char_1d_type host_nonzerosInRow = create_mirror_view(A.nonzerosInRow);
	const host_const_double_1d_type host_xvalues = create_mirror_view(x.values);
	const host_const_double_1d_type host_xexactvalues = create_mirror_view(xexact.values);
	const host_const_double_1d_type host_bvalues = create_mirror_view(b.values);
	deep_copy(host_matrixValues, A.matrixValues);
	deep_copy(host_mtxIndG, A.mtxIndG);
	deep_copy(host_nonzerosInRow, A.nonzerosInRow);
	deep_copy(host_xvalues, x.values);
	deep_copy(host_xexactvalues, xexact.values);
	deep_copy(host_bvalues, b.values);
  for (global_int_t i=0; i< nrow; i++) {
    auto currentRowValues = subview(host_matrixValues, i, ALL()); //TODO Should use auto
    auto currentRowIndices = subview(host_mtxIndG, i, ALL()); //TODO Should use auto
    const int currentNumberOfNonzeros = host_nonzerosInRow(i);
    for (int j=0; j< currentNumberOfNonzeros; j++)
#ifdef HPCG_NO_LONG_LONG
      fprintf(fA, " %d %d %22.16e\n",i+1,(global_int_t)(currentRowIndices(j)+1),currentRowValues(j));
#else
      fprintf(fA, " %lld %lld %22.16e\n",i+1,(global_int_t)(currentRowIndices(j)+1),currentRowValues(j));
#endif
    fprintf(fx, "%22.16e\n",host_xvalues(i));
    fprintf(fxexact, "%22.16e\n",host_xexactvalues(i));
    fprintf(fb, "%22.16e\n",host_bvalues(i));
  }

  fclose(fA);
  fclose(fx);
  fclose(fxexact);
  fclose(fb);
  return(0);
}
