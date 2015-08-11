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
		double z_tmp;
		int diag_tmp;
		diag_tmp=A.values(diag(i));
		z_tmp=r(i)/diag_tmp;
		z_tmp += z_old(i);
		for(int k = A.graph.row_map(i); k <= diag(i); k++)
			rowDot += A.values(k) * z_old(A.graph.entries(k));
		z_tmp -=rowDot/diag_tmp;
		z_new(i)=z_tmp;
	}
};

...
Kokkos::parallel_for(localNumberOfRows, LowerTrisolve(A.localMatrix, A.matrixDiagonal, r.values, z, A.old));
..