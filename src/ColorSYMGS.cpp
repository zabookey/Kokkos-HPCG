#ifdef SYMGS_COLOR
#include "ColorSYMGS.hpp"

class colouredForwardSweep{
  public:
  local_int_t colors_row;
  local_int_1d_type colors_ind;

  local_matrix_type A;
  double_1d_type rv, xv;
  int_1d_type matrixDiagonal;

  colouredForwardSweep(const local_int_t colors_row_, const local_int_1d_type& colors_ind_,
    const local_matrix_type& A_, const double_1d_type& rv_, double_1d_type& xv_,
    const int_1d_type matrixDiagonal_):
    colors_row(colors_row_), colors_ind(colors_ind_), A(A_), rv(rv_), xv(xv_),
    matrixDiagonal(matrixDiagonal_) {}

  void operator()(const int & i)const{
    local_int_t currentRow = colors_ind(colors_row + i); // This should tell us what row we're doing SYMGS on.
    int start = A.graph.row_map(currentRow);
    int end = A.graph.row_map(currentRow+1);
    const double currentDiagonal = A.values(matrixDiagonal(currentRow));
    double sum = rv(currentRow);
    for(int j = start; j < end; j++)
      sum -= A.values(j) * xv(A.graph.entries(j));
    sum += xv(currentRow) * currentDiagonal;
    xv(currentRow) = sum/currentDiagonal;
  }
};

class colouredBackSweep{
  public:
  local_int_t colors_row;
  local_int_1d_type colors_ind;
  
  local_matrix_type A;
  double_1d_type rv, xv;
  int_1d_type matrixDiagonal;

  colouredBackSweep(const local_int_t colors_row_, const local_int_1d_type& colors_ind_,
      const local_matrix_type& A_, const double_1d_type& rv_, double_1d_type& xv_,
      const int_1d_type matrixDiagonal_):
      colors_row(colors_row_), colors_ind(colors_ind_), A(A_), rv(rv_), xv(xv_),
      matrixDiagonal(matrixDiagonal_) {}

  void operator()(const int & i)const{
    local_int_t currentRow = colors_ind(colors_row + i); // This should tell us what row we're doing SYMGS on.
    int start = A.graph.row_map(currentRow);
    int end = A.graph.row_map(currentRow+1);
    const double currentDiagonal = A.values(matrixDiagonal(currentRow));
    double sum = rv(currentRow);
    for(int j = start; j < end; j++)
      sum -= A.values(j) * xv(A.graph.entries(j));
    sum += xv(currentRow) * currentDiagonal;
    xv(currentRow) = sum/currentDiagonal;
  }
};

int ColorSYMGS( const SparseMatrix & A, const Vector & r, Vector & x){
assert(x.localLength == A.localNumberOfColumns); // Make sure x contains space for halo values

#ifndef HPCG_NOMPI
  ExchangeHalo(A,x);
#endif
	 // Forward Sweep!
  const int numColors = A.numColors;
  local_int_t dummy = 0;
  for(int i = 0; i < numColors; i++){
    int currentColor = A.f_colors_order(i);
    int start = A.colors_map(currentColor - 1); // Colors start at 1, i starts at 0
    int end = A.colors_map(currentColor);
    dummy += end - start;
    Kokkos::parallel_for(end - start, colouredForwardSweep(start, A.colors_ind, A.localMatrix, r.values, x.values, A.matrixDiagonal));
  }
  assert(dummy == A.localNumberOfRows);
 // Back Sweep!
  for(int i = numColors -1; i >= 0; --i){
    int currentColor = A.f_colors_order(i);
    int start = A.colors_map(currentColor - 1); // Colors start at 1, i starts at 0
    int end = A.colors_map(currentColor);
    Kokkos::parallel_for(end - start, colouredBackSweep(start, A.colors_ind, A.localMatrix, r.values, x.values, A.matrixDiagonal));
  }

return(0);
}
#endif
