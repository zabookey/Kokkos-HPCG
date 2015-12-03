#ifdef SYMGS_COLOR
#include "ColorSYMGS.hpp"

#ifdef KOKKOS_TEAM
typedef Kokkos::TeamPolicy<> team_policy;
typedef typename team_policy::member_type team_member;
class ColouredSweep{
public:
  local_matrix_type A;

  local_int_t color_set_begin;
  local_int_t color_set_end;

  local_int_1d_type colors_ind;

  double_1d_type rv, xv;

  ColouredSweep(const local_int_t color_set_begin_, const local_int_t color_set_end_, 
    const local_matrix_type& A_, const local_int_1d_type& colors_ind_, const double_1d_type& rv_, double_1d_type& xv_):
    color_set_begin(color_set_begin_), color_set_end(color_set_end_), A(A_), colors_ind(colors_ind_), rv(rv_), xv(xv_) {}
KOKKOS_INLINE_FUNCTION
  void operator()(const team_member & teamMember) const{
    int ii = teamMember.league_rank() * teamMember.team_size() + teamMember.team_rank() + color_set_begin;
    if(ii >= color_set_end) return;

	int crow = colors_ind(ii);

    int row_begin = A.graph.row_map(crow);
    int row_end = A.graph.row_map(crow+1);

    bool am_i_the_diagonal = false;
    double diagonal = 1;
    double sum = 0;
    Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, row_end - row_begin),
      [&] (int i, double & valueToUpdate) {
        int adjind = i + row_begin;
        int colIndex = A.graph.entries(adjind);
        double val = A.values(adjind);
        if(colIndex == crow){
          diagonal = val;
          am_i_the_diagonal = true;
        }
        else{
          valueToUpdate += val * xv(colIndex);
        }
      }, sum);

    if(am_i_the_diagonal){
      xv(crow) = (rv(crow) - sum)/diagonal;
    }
  }
};
#else
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

KOKKOS_INLINE_FUNCTION
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
KOKKOS_INLINE_FUNCTION
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
#endif

int ColorSYMGS( const SparseMatrix & A, const Vector & r, Vector & x){
assert(x.localLength == A.localNumberOfColumns); // Make sure x contains space for halo values

#ifndef HPCG_NOMPI
  ExchangeHalo(A,x);
#endif
	 // Forward Sweep!
#ifdef KOKKOS_TEAM
  int vector_size = 64;
  int teamSizeMax = 8;
  for(int i = 0; i < A.numColors; i++){
    int color_index_begin = A.host_colors_map(i);
    int color_index_end = A.host_colors_map(i + 1);
    int numberOfTeams = color_index_end - color_index_begin;
    Kokkos::parallel_for(team_policy(numberOfTeams / teamSizeMax + 1, teamSizeMax, vector_size),
      ColouredSweep(color_index_begin, color_index_end, A.localMatrix, A.colors_ind, r.values, x.values));

    execution_space::fence();
  }
  for(int i = A.numColors - 1; i >= 0; i--){
    int color_index_begin = A.host_colors_map(i);
    int color_index_end = A.host_colors_map(i+1);
    int numberOfTeams = color_index_end - color_index_begin;
    Kokkos::parallel_for(team_policy(numberOfTeams / teamSizeMax + 1, teamSizeMax, vector_size),
      ColouredSweep(color_index_begin, color_index_end, A.localMatrix, A.colors_ind, r.values, x.values));
    execution_space::fence();
  }
#else
  const int numColors = A.numColors;
  local_int_t dummy = 0;
  for(int i = 0; i < numColors; i++){
    int currentColor = A.host_f_colors_order(i);
    int start = A.host_colors_map(currentColor - 1); // Colors start at 1, i starts at 0
    int end = A.host_colors_map(currentColor);
   dummy += end - start;
    Kokkos::parallel_for(end - start, colouredForwardSweep(start, A.colors_ind, A.localMatrix, r.values, x.values, A.matrixDiagonal));
  }
  assert(dummy == A.localNumberOfRows);
 // Back Sweep!
  for(int i = numColors -1; i >= 0; --i){
    int currentColor = A.host_f_colors_order(i);
    int start = A.host_colors_map(currentColor - 1); // Colors start at 1, i starts at 0
    int end = A.host_colors_map(currentColor);
    Kokkos::parallel_for(end - start, colouredBackSweep(start, A.colors_ind, A.localMatrix, r.values, x.values, A.matrixDiagonal));
  }
#endif
return(0);
}
#endif
