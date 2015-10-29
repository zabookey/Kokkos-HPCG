#ifdef SYMGS_INEXACT
#include "InexactSYMGS.hpp"

#ifdef KOKKOS_TEAM
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
            //   int row_indx=   thread.league_rank()*thread.team_size(); //+thread.team_rank();
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
                //Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread, k_start, k_end),
                //KOKKOS_LAMBDA(const int& k, double& lrowDot){
                Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread, vector_range),
                               KOKKOS_LAMBDA(const int& lk, double& lrowDot){
                           const int k=k_start+lk;
                           lrowDot += A.values(k) * z_old(A.graph.entries(k));
                }, rowDot);
//                for(int k = A.graph.row_map(irow); k <= diag(irow); k++)
//                        rowDot += A.values(k) * z_old(A.graph.entries(k));
//                Kokkos::single(Kokkos::PerThread(thread),[&](){
                z_tmp -=rowDot/diag_tmp;
                z_new(irow)=z_tmp;
//                });
              });
        }
};

class UpperTrisolve{
  public:
  local_matrix_type A;
  const_int_1d_type diag;
  const_double_1d_type w;
  double_1d_type x_new;
  double_1d_type x_old;
  int localNumberOfRows;
  int rpt = rows_per_team;

  UpperTrisolve(const local_matrix_type& A_,const const_int_1d_type& diag_, const const_double_1d_type& w_,
    double_1d_type& x_new_,const double_1d_type& x_old_, const int localNumberOfRows_):
    A(A_), diag(diag_), w(w_), x_new(x_new_), x_old(x_old_), localNumberOfRows(localNumberOfRows_){
    Kokkos::deep_copy(x_old, x_new_);
    }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member & thread) const{
    int row_indx = thread.league_rank()* rpt;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, row_indx, row_indx+rpt), [=] (int& irow){
      double rowDot = 0.0;
      double x_tmp;
      int diag_tmp;
      diag_tmp = A.values(diag(irow));
      x_tmp = w(irow)/diag_tmp;
      x_tmp += x_old(irow);
      const int k_start = diag(irow);
      const int k_end = A.graph.row_map(irow + 1);
      const int vector_range = k_end - k_start;
      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(thread, vector_range),
          KOKKOS_LAMBDA(const int& lk, double& lrowDot){
        const int k = k_start+lk;
        lrowDot += A.values(k) * x_old(A.graph.entries(k));
          }, rowDot);

      x_tmp -= rowDot/diag_tmp;
      x_new(irow)=x_tmp;
    });
  }
};

#else
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
  //  z_new(i) = r(i)/A.values(diag(i));
  //  z_new(i) += z_old(i);
    diag_tmp=A.values(diag(i));
    z_tmp=r(i)/diag_tmp;
                z_tmp += z_old(i);
    for(int k = A.graph.row_map(i); k <= diag(i); k++)
      rowDot += A.values(k) * z_old(A.graph.entries(k));
    //z_new(i) -= rowDot/A.values(diag(i));
    z_tmp -=rowDot/diag_tmp;
                z_new(i)=z_tmp;
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
    double x_tmp;
    int diag_tmp;
    diag_tmp = A.values(diag(i));
    x_tmp = w(i)/diag_tmp;
    x_tmp += x_old(i);
    for(int k = diag(i); k < A.graph.row_map(i+1); k++)
      rowDot += A.values(k) * x_old(A.graph.entries(k));
    x_tmp -= rowDot/diag_tmp;
    x_new(i)=x_tmp;
  }
};
#endif

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
/*
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
*/

int InexactSYMGS( const SparseMatrix & A, const Vector & r, Vector & x){
	
assert(x.localLength == A.localNumberOfColumns); // Make sure x contains space for halo values

#ifndef HPCG_NOMPI
  ExchangeHalo(A,x);
#endif

	  const local_int_t localNumberOfRows = A.localNumberOfRows;
  const int iterations = 20;
  double_1d_type z("z", x.values.dimension_0());
  for(int i = 0; i < iterations; i++){
#ifdef KOKKOS_TEAM
          const int team_size=localNumberOfRows/rows_per_team;
//           const team_policy policy( 512 , team_policy::team_size_max( LowerTrisolve(A.localMatrix, A.matrixDiagonal, r.values, z, A.old, localNumberOfRows) ), 4 );

        const team_policy policy( team_size , team_policy::team_size_max( LowerTrisolve(A.localMatrix, A.matrixDiagonal, r.values, z, A.old, localNumberOfRows) ),1 );
//         const team_policy policy( team_size , team_policy::team_size_max( LowerTrisolve(A.localMatrix, A.matrixDiagonal, r.values, z, A.old, localNumberOfRows) ), 16 );
          Kokkos::parallel_for(policy, LowerTrisolve(A.localMatrix, A.matrixDiagonal, r.values, z, A.old,  localNumberOfRows));
#else
    Kokkos::parallel_for(localNumberOfRows, LowerTrisolve(A.localMatrix, A.matrixDiagonal, r.values, z, A.old));
#endif
    Kokkos::fence();
  }
  Kokkos::parallel_for(localNumberOfRows, applyD(A.localMatrix, A.matrixDiagonal, z));
  for(int i = 0; i < iterations; i++){
#ifdef KOKKOS_TEAM
    const int team_size = localNumberOfRows/rows_per_team;
    const team_policy policy(team_size, team_policy::team_size_max(UpperTrisolve(A.localMatrix, A.matrixDiagonal, z, x.values, A.old, localNumberOfRows)), 16);
    Kokkos::parallel_for(policy, UpperTrisolve(A.localMatrix, A.matrixDiagonal, z, x.values, A.old, localNumberOfRows));
#else
    Kokkos::parallel_for(localNumberOfRows, UpperTrisolve(A.localMatrix, A.matrixDiagonal, z, x.values, A.old));
#endif
    Kokkos::fence();
  }
  return(0);
}
#endif
