
#ifndef HPCG_NOMPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS_ref.hpp"
#include <cassert>

using Kokkos::create_mirror_view;
using Kokkos::deep_copy;
using Kokkos::subview;
using Kokkos::ALL;
/*!
  Computes one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.
  - We then perform one back sweep.
  - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.


  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSYMGS
*/
#ifdef Option_0
class leveledForwardSweep{
	public:
	local_int_t f_lev_start;
	local_int_1d_type f_lev_ind;

	local_matrix_type A;
	double_1d_type rv, zv;
	int_1d_type matrixDiagonal;

	leveledForwardSweep(const local_int_t f_lev_start_, const local_int_1d_type & f_lev_ind_, 
		const local_matrix_type& A_, const double_1d_type& rv_, double_1d_type& zv_,
		const int_1d_type& matrixDiagonal_):
		f_lev_start(f_lev_start_), f_lev_ind(f_lev_ind_), A(A_), rv(rv_), zv(zv_),
		matrixDiagonal(matrixDiagonal_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		local_int_t currentRow = f_lev_ind(f_lev_start+i);
		int start = A.graph.row_map(currentRow);
		const int diagIdx = matrixDiagonal(currentRow);
		double sum = rv(i);
		for(int j = start; j < diagIdx; j++)
			sum -= zv(A.graph.entries(j))*A.values(j);
		zv(i) = sum/A.values(diagIdx);
	}
};

class applyD{
	public:
	local_matrix_type A;
	const_int_1d_type diag;
	double_1d_type z;

	applyD(const local_matrix_type& A_, const const_int_1d_type& diag_, const double_1d_type& z_):
		A(A_), diag(diag_), z(z_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		z(i) = z(i)*A.values(diag(i));
	}
};

class leveledBackSweep{
	public:
	local_int_t b_lev_start;
	local_int_1d_type b_lev_ind;

	local_matrix_type A;
	double_1d_type zv, xv;
	int_1d_type matrixDiagonal;

	leveledBackSweep(const local_int_t b_lev_start_, const local_int_1d_type & b_lev_ind_, 
		const local_matrix_type& A_, const double_1d_type& zv_, double_1d_type& xv_,
		const int_1d_type& matrixDiagonal_):
		b_lev_start(b_lev_start_), b_lev_ind(b_lev_ind_), A(A_), zv(zv_), xv(xv_),
		matrixDiagonal(matrixDiagonal_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		local_int_t currentRow = b_lev_ind(b_lev_start+i);
		int end = A.graph.row_map(currentRow+1);
		const int diagIdx = matrixDiagonal(currentRow);
		double sum = zv(i);
		for(int j = diagIdx+1; j < end; j++)
			sum -= xv(A.graph.entries(j))*A.values(j);
		xv(i) = sum/A.values(diagIdx);
	}
};
#endif
#ifdef Option_1
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
		xv(i) = sum/currentDiagonal;
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
		xv(i) = sum/currentDiagonal;
	}
};
#endif

#ifdef Option_2
class forwardSweep{
	public:
	local_matrix_type A;
	double_1d_type rv, xv;
	int_1d_type matrixDiagonal;
	
	forwardSweep(const local_matrix_type& A_, const double_1d_type& rv_, double_1d_type& xv_,
		int_1d_type matrixDiagonal_):
		A(A_), rv(rv_), xv(xv_), matrixDiagonal(matrixDiagonal_) {}

	void operator()(const int & i) const{
		int start = A.graph.row_map(i);
		int end = A.graph.row_map(i+1);
		const double currentDiagonal = A.values(matrixDiagonal(i));
		double sum = rv(i);

		for(int j = start; j < end; j++)
			sum -= A.values(j) * xv(A.graph.entries(j));
		sum += xv(i) * currentDiagonal;
		
		xv(i) = sum/currentDiagonal;
	}
	
};

class backSweep{
	public:
	local_matrix_type A;
	double_1d_type rv, xv;
	int_1d_type matrixDiagonal;
	int nrow;
	
	backSweep(const local_matrix_type& A_, const double_1d_type& rv_, double_1d_type& xv_,
		int_1d_type matrixDiagonal_, const local_int_t nrow_):
		A(A_), rv(rv_), xv(xv_), matrixDiagonal(matrixDiagonal_), nrow(nrow_) {}

	void operator()(const int & i) const{
		int start = A.graph.row_map(nrow - i - 1); //Work from the end of the matrix up.
		int end = A.graph.row_map(nrow - i);
		const double currentDiagonal = A.values(matrixDiagonal(i));
		double sum = rv(i);

		for(int j = start; j < end; j++)
			sum -= A.values(j) * xv(A.graph.entries(j));
		sum += xv(i) * currentDiagonal;
		
		xv(i) = sum/currentDiagonal;
	}
	
};
#endif

#ifdef Option_3
class LowerTrisolve{
	public:
	local_matrix_type A;
	const_int_1d_type diag;
	const_double_1d_type r;
	double_1d_type x_new;
	const_double_1d_type x_old;

	LowerTrisolve(const local_matrix_type& A_,const const_int_1d_type& diag_, const const_double_1d_type& r_,
		double_1d_type& x_new_):
		A(A_), diag(diag_), r(r_), x_new(x_new_){
		double_1d_type x_tmp(Kokkos::ViewAllocateWithoutInitializing("x_tmp"), x_new_.dimension_0());
		Kokkos::deep_copy(x_tmp, x_new_);
		x_old = x_tmp;
		}

/* [a_i1/a_ii, a_i2/a_ii, ..., a_ii-1/a_ii, 1, 0, 0, ..., 0] This is our row to dot product with X */

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		double rowDot = 0.0;
		x_new(i) = r(i);//Since our diagonal in this matrix is 1 there should be no need to divide the diagonal.
		x_new(i) += x_old(i);
		for(int k = A.graph.row_map(i); k < diag(i); k++){ // This should start at the beginning of the row and go up to the diagonal.
			rowDot += A.values(k) * x_old(A.graph.entries(k)) / A.values(diag(A.graph.entries(k)));
		}
//		rowDot = rowDot/A.values((local_int_t)diag(i)); // Scale by the diagonal in A.
		rowDot += x_old(i); // Add in 1 * x_old(i) since we skipped the diagonal before.
		x_new(i) -= rowDot;//Since our diagonal in this matrix is 1 there should be no need to divide the diagonal
	}
	
};

class UpperTrisolve{
	public:
	local_matrix_type A;
	const_int_1d_type diag;
	const_double_1d_type r;
	double_1d_type x_new;
	const_double_1d_type x_old;
	local_int_t nrows;

	UpperTrisolve(const local_matrix_type& A_,const const_int_1d_type& diag_, const const_double_1d_type& r_,
		double_1d_type& x_new_):
		A(A_), diag(diag_), r(r_), x_new(x_new_){
		double_1d_type x_tmp(Kokkos::ViewAllocateWithoutInitializing("x_tmp"), x_new_.dimension_0());
		Kokkos::deep_copy(x_tmp, x_new_);
		x_old = x_tmp;
		nrows = x_new_.dimension_0() - 1;
		}
/* This is just the jacobi method only applied from the diagonal to the end of the row
	 to simulate the lower triangular portion being only 0's
*/
	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		double rowDot = 0.0;
		x_new(nrows - i) = r(nrows - i)/A.values(diag(nrows - i));
		x_new(nrows - i) += x_old(nrows - i);
		for(int k = diag(nrows - i); k < A.graph.row_map(nrows - i+1); k++)
			rowDot+=A.values(k) * x_old(A.graph.entries(k));
		x_new(nrows - i) -= rowDot/A.values(diag(nrows - i));
	}
};
#endif

#ifdef Option_4
class LowerTrisolve{
	public:
	local_matrix_type A;
	const_int_1d_type diag;
	const_double_1d_type r;
	double_1d_type z_new;
	double_1d_type z_old;

	LowerTrisolve(const local_matrix_type& A_,const const_int_1d_type& diag_, const const_double_1d_type& r_,
		double_1d_type& z_new_, const double_1d_type& z_old_):
		A(A_), diag(diag_), r(r_), z_new(z_new_), z_old(z_old_){
		Kokkos::deep_copy(z_old, z_new);
		}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i) const{
		double rowDot = 0.0;
		z_new(i) = r(i)/A.values(diag(i));
		z_new(i) += z_old(i);
		for(int k = A.graph.row_map(i); k <= diag(i); k++)
			rowDot += A.values(k) * z_old(A.graph.entries(k));
		z_new(i) -= rowDot/A.values(diag(i));
	}
};

class applyD{
	public:
	local_matrix_type A;
	const_int_1d_type diag;
	double_1d_type z;

	applyD(const local_matrix_type& A_, const const_int_1d_type& diag_, const double_1d_type& z_):
		A(A_), diag(diag_), z(z_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		z(i) = z(i)*A.values(diag(i));
	}
};

class UpperTrisolve{
	public:
	local_matrix_type A;
	const_int_1d_type diag;
	const_double_1d_type w;
	double_1d_type x_new;
	double_1d_type x_old;

	UpperTrisolve(const local_matrix_type& A_,const const_int_1d_type& diag_, const const_double_1d_type& w_,
		double_1d_type& x_new_,const double_1d_type& x_old_):
		A(A_), diag(diag_), w(w_), x_new(x_new_), x_old(x_old_){
		Kokkos::deep_copy(x_old, x_new_);
		}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i) const{
		double rowDot = 0.0;
		x_new(i) = w(i)/A.values(diag(i));
		x_new(i) += x_old(i);
		for(int k = diag(i); k < A.graph.row_map(i+1); k++)
			rowDot += A.values(k) * x_old(A.graph.entries(k));
		x_new(i) -= rowDot/A.values(diag(i));
	}
};
#endif

int ComputeSYMGS_ref(const SparseMatrix & A, const Vector & r, Vector & x){

	assert(x.localLength == A.localNumberOfColumns); // Make sure x contains space for halo values

#ifndef HPCG_NOMPI
	ExchangeHalo(A,x);
#endif
//	for(int i = 0; i < 10; i++) std::cout << "Before SYMGS: " << x.values(i) << "    " << r.values(i) << std::endl;
#ifdef Option_0
/*

// Step 1. Find (D+L)z=r
	host_double_1d_type z("z", xv.dimension_0());
	for(local_int_t i = 0; i < nrow; i++){
		int start = rowMap(i);
		int end = rowMap(i+1);
		const int diagIdx = matrixDiagonal(i);
		double sum = rv(i);
		for(int j = start; j < diagIdx; j++)
			sum -= z(entries(j))*values(j);
		z(i) = sum/values(diagIdx);
	}
// Step 2. Find Dw = z
	host_double_1d_type w("w", xv.dimension_0());
	for(local_int_t i = 0; i < nrow; i++)
		w(i) = z(i)*values(matrixDiagonal(i));
// Step 3. Find (D+U)x = w
	for(local_int_t i = nrow - 1; i >= 0; i--){
		int start = rowMap(i);
		int end = rowMap(i+1);
		const int diagIdx = matrixDiagonal(i);
		double sum = w(i);
		for(int j = diagIdx + 1; j < end; j++)
			sum -= xv(entries(j))*values(j);
		xv(i) = sum/values(diagIdx);
	}
*/

	const int f_numLevels = A.levels.f_numberOfLevels;
	const int b_numLevels = A.levels.b_numberOfLevels;
	double_1d_type z("z", x.values.dimension_0());
	for(int i = 0; i < f_numLevels; i++){
		int start = A.levels.f_lev_map(i);
		int end = A.levels.f_lev_map(i+1);
		Kokkos::parallel_for(end - start, leveledForwardSweep(start, A.levels.f_lev_ind, A.localMatrix, r.values, z, A.matrixDiagonal));
	}
	Kokkos::parallel_for(z.dimension_0(), applyD(A.localMatrix, A.matrixDiagonal, z));
	for(int i = 0; i < b_numLevels; i++){
		int start = A.levels.b_lev_map(i);
		int end = A.levels.b_lev_map(i+1);
		Kokkos::parallel_for(end - start, leveledBackSweep(start, A.levels.b_lev_ind, A.localMatrix, z, x.values, A.matrixDiagonal));
	}
#else
#ifdef Option_1
 // Level Solve Algorithm will go here.
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
	for(int i = 0; i < numColors; i++){
		int currentColor = A.b_colors_order(i);
		int start = A.colors_map(currentColor - 1); // Colors start at 1, i starts at 0
		int end = A.colors_map(currentColor);
		Kokkos::parallel_for(end - start, colouredBackSweep(start, A.colors_ind, A.localMatrix, r.values, x.values, A.matrixDiagonal));
	}
	
#else
#ifdef Option_2
	const local_int_t nrow = A.localNumberOfRows;
	int approxIter = 4; //Since this is 2*number of default threads on faure.
	for(int k = 0; k < approxIter; k++){
		Kokkos::parallel_for(nrow, forwardSweep(A.localMatrix, r.values, x.values, A.matrixDiagonal));
		Kokkos::fence();
	}
	for(int k = 0; k < approxIter; k++){
		Kokkos::parallel_for(nrow, backSweep(A.localMatrix, r.values, x.values, A.matrixDiagonal, nrow));
		Kokkos::fence();
	}
#else
#ifdef Option_3
	const local_int_t localNumberOfRows = A.localNumberOfRows;
	const int iterations = 1;
	double_1d_type z("z", x.values.dimension_0());
// Apply LowerTrisolve
// Solves (I - ED^{-1})z = r
	for(int i = 0; i < iterations; i++){
		Kokkos::parallel_for(localNumberOfRows, LowerTrisolve(A.localMatrix, A.matrixDiagonal, r.values, z));
		Kokkos::fence();
	}
// Apply UpperTrisolve
// Solves (D - F)x = z
	for(int i = 0; i < iterations; i++){
		Kokkos::parallel_for(localNumberOfRows, UpperTrisolve(A.localMatrix, A.matrixDiagonal, z, x.values));
		Kokkos::fence();
	}
#else
#ifdef Option_4
	const local_int_t localNumberOfRows = A.localNumberOfRows;
	const int iterations = 8;
	double_1d_type z("z", x.values.dimension_0());
	for(int i = 0; i < iterations; i++){
		Kokkos::parallel_for(localNumberOfRows, LowerTrisolve(A.localMatrix, A.matrixDiagonal, r.values, z, A.old));
		Kokkos::fence();
	}
	Kokkos::parallel_for(localNumberOfRows, applyD(A.localMatrix, A.matrixDiagonal, z));
	for(int i = 0; i < iterations; i++){
		Kokkos::parallel_for(localNumberOfRows, UpperTrisolve(A.localMatrix, A.matrixDiagonal, z, x.values, A.old));
		Kokkos::fence();
	}
#else
	const local_int_t nrow = A.localNumberOfRows;
	host_int_1d_type matrixDiagonal = create_mirror_view(A.matrixDiagonal); //Host Mirror to A.matrixDiagonal.
	host_const_double_1d_type rv = create_mirror_view(r.values); // Host Mirror to r.values
	const host_double_1d_type xv = create_mirror_view(x.values); // Host Mirror to x.values
	const host_const_char_1d_type nonZerosInRow = create_mirror_view(A.nonzerosInRow); // Host Mirror to A.nonZerosInRow.
	const host_values_type values = create_mirror_view(A.localMatrix.values);
	const host_local_index_type entries = create_mirror_view(A.localMatrix.graph.entries);
	const host_row_map_type rowMap = create_mirror_view(A.localMatrix.graph.row_map);
//	Easier to Mirror it once than mirror in every iteration
//	Deep Copy the values into the mirrors... Because mirrors don't do that...
	deep_copy(matrixDiagonal, A.matrixDiagonal);
	deep_copy(rv, r.values);
	deep_copy(xv, x.values);
	deep_copy(nonZerosInRow, A.nonzerosInRow);
	deep_copy(values, A.localMatrix.values);
	deep_copy(entries, A.localMatrix.graph.entries);
	deep_copy(rowMap, A.localMatrix.graph.row_map);
/*// FIXME DEBUG TESTING
	for(local_int_t i = 0; i < nrow; i++){
		int start = rowMap(i);
		int end = rowMap(i+1);
		const double currentDiagonal = values(matrixDiagonal(i)); //This works only if matrixDiagonal is the indices of the diagonal and not the value.If its the values remove currentValues( ).
		double sum = rv(i);

		for(int j = start; j < end; j++){
			local_int_t curCol = entries(j);
			sum -= values(j) * xv(curCol);
		}
		sum += xv(i)*currentDiagonal;

		xv(i) = sum/currentDiagonal;
	}

	// Now the back sweep.

	for(local_int_t i = nrow-1; i >= 0; i--){
		int start = rowMap(i);
		int end = rowMap(i+1);
		const double currentDiagonal = values(matrixDiagonal(i)); // Same reason as the last loop. Change if needed.
		double sum = rv(i);

		for(int j = start; j < end; j++){
			local_int_t curCol = entries(j);
			sum -= values(j) * xv(curCol);
		}
		sum += xv(i)*currentDiagonal;

		xv(i) = sum/currentDiagonal;
	}
*/

// Step 1. Find (D+L)z=r
	host_double_1d_type z("z", xv.dimension_0());
	for(local_int_t i = 0; i < nrow; i++){
		int start = rowMap(i);
		int end = rowMap(i+1);
		const int diagIdx = matrixDiagonal(i);
		double sum = rv(i);
		for(int j = start; j < diagIdx; j++)
			sum -= z(entries(j))*values(j);
		z(i) = sum/values(diagIdx);
	}
// Step 2. Find Dw = z
	host_double_1d_type w("w", xv.dimension_0());
	for(local_int_t i = 0; i < nrow; i++)
		w(i) = z(i)*values(matrixDiagonal(i));
// Step 3. Find (D+U)x = w
	for(local_int_t i = nrow - 1; i >= 0; i--){
		int start = rowMap(i);
		int end = rowMap(i+1);
		const int diagIdx = matrixDiagonal(i);
		double sum = w(i);
		for(int j = diagIdx + 1; j < end; j++)
			sum -= xv(entries(j))*values(j);
		xv(i) = sum/values(diagIdx);
	}

	deep_copy(x.values, xv); // Copy the updated xv on the host back to x.values.
	// All of the mirrors should go out of scope here and deallocate themselves.
#endif // Option_4
#endif // Option_3
#endif // Option_2
#endif // Option_1
#endif // Option_0
//for(int i = 0; i < 10; i++) std::cout << "After SYMGS: " << x.values(i) << "    " << r.values(i) << std::endl;
	return(0);
}
