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
	const host_global_int_1d_type host_matG_entries = create_mirror_view(A.globalMatrix.graph.entries);
	const host_const_double_1d_type host_xvalues = create_mirror_view(x.values);
	const host_const_double_1d_type host_xexactvalues = create_mirror_view(xexact.values);
	const host_const_double_1d_type host_bvalues = create_mirror_view(b.values);
	const host_values_type host_values = create_mirror_view(A.localMatrix.values);
	const host_row_map_type host_rowMap = create_mirror_view(A.localMatrix.graph.row_map);
	deep_copy(host_matG_entries, A.globalMatrix.graph.entries);
	deep_copy(host_xvalues, x.values);
	deep_copy(host_xexactvalues, xexact.values);
	deep_copy(host_bvalues, b.values);
	deep_copy(host_values, A.localMatrix.values);
	deep_copy(host_rowMap, A.localMatrix.graph.row_map);
  for (global_int_t i=0; i< nrow; i++) {
		int start = host_rowMap(i);
		int end = host_rowMap(i+1);
    for (int j=start; j< end; j++)
#ifdef HPCG_NO_LONG_LONG
      fprintf(fA, " %d %d %22.16e\n",i+1,(global_int_t)(host_matG_entries(j)+1),host_values(j));
#else
      fprintf(fA, " %lld %lld %22.16e\n",i+1,(global_int_t)(host_matG_entries(j)+1),host_values(j));
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
