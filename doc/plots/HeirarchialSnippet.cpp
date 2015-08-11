typedef Kokkos::TeamPolicy<>              team_policy ;
typedef team_policy::member_type team_member ;
int rows_per_team=104;

class LowerTrisolve{
    public:
    local_matrix_type A;
    const_int_1d_type diag;
    const_double_1d_type r;
    double_1d_type z_new;
    double_1d_type z_old;
    int localNumberOfRows;
    int rpt = rows_per_team;

    LowerTrisolve(const local_matrix_type& A_,const const_int_1d_type& diag_, const const_double_1d_type& r_,
            double_1d_type& z_new_, const double_1d_type& z_old_, const int localNumberOfRows_):
            A(A_), diag(diag_), r(r_), z_new(z_new_), z_old(z_old_), localNumberOfRows(localNumberOfRows_){
            Kokkos::deep_copy(z_old, z_new);
            }

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member &  thread) const{
        int row_indx=thread.league_rank()* rpt;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, row_indx, row_indx+rpt), [=] (int& irow){
            double rowDot = 0.0;
            double z_tmp;
            int diag_tmp;
            diag_tmp=A.values(diag(irow));
            z_tmp=r(irow)/diag_tmp;
            z_tmp += z_old(irow);
            const int k_start=A.graph.row_map(irow);
            const int k_end=diag(irow)+1;
            const int vector_range=k_end-k_start;
            Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread, vector_range),
                    KOKKOS_LAMBDA(const int& lk, double& lrowDot){
                    const int k=k_start+lk;
                    lrowDot += A.values(k) * z_old(A.graph.entries(k));
                }, rowDot);
            z_tmp -=rowDot/diag_tmp;
            z_new(irow)=z_tmp;
        });
    }
};