
#ifndef HPCG_NOMPI
#include <mpi.h> // If this routine is not compiled with HPCG_NOMPI
#endif
#ifndef HPCG_NOOPENMP
#include <omp.h> // If this routine is not compiled with HPCG_NOOPENMP
#endif

#include "Vector.hpp"

#ifdef HPCG_DETAILED_DEBUG
#include <fstream>
#include "hpcg.hpp"
#endif

#include <cmath>  // needed for fabs
#include <mutex>  // needed for mutex
#include "ComputeResidual.hpp"
#ifdef HPCG_DETAILED_DEBUG
#include <iostream>
#endif

//TODO Try to use Atomics instead of mutex
static std::mutex local_residual_mutex;

class FunctorforKokkos{
  public:
  kokkos_type v1v;
  kokkos_type v2v;
  mutable double local_residual;
  //std::mutex local_residual_mutex;
  mutable double threadlocal_residual = 0.0;
  local_int_t loopcount;

  FunctorforKokkos(kokkos_type v1, kokkos_type v2, double & shared_local_residual, 
                      /*std::mutex & shared_mutex,*/ local_int_t n){
    v1v = v1;
    v2v = v2;
    local_residual = shared_local_residual;
    //local_residual_mutex = shared_mutex;
    loopcount = n-1;
   }

  KOKKOS_INLINE_FUNCTION
  void operator()(local_int_t i)const{
    double diff = std::fabs(v1v(i) - v2v(i));
    if(diff > threadlocal_residual) threadlocal_residual = diff;
    if(i == loopcount){
      local_residual_mutex.lock();
      if(threadlocal_residual > local_residual) local_residual = threadlocal_residual;
      local_residual_mutex.unlock();
    }
  }
};

/*!
  Routine to compute the inf-norm difference between two vectors where:

  @param[in]  n        number of vector elements (local to this processor)
  @param[in]  v1, v2   input vectors
  @param[out] residual pointer to scalar value; on exit, will contain result: inf-norm difference

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeResidual(const local_int_t n, const Vector & v1, const Vector & v2, double & residual) {

  kokkos_type v1v = v1.values;
  kokkos_type v2v = v2.values;
  double local_residual = 0.0;
  //std::mutex local_residual_mutex;
  //double threadlocal_residual = 0.0;
  //int loopcount = n-1;

/*
  Kokkos::parallel_for(n,
  [&v1v, &v2v, &local_residual, &local_residual_mutex, threadlocal_residual, loopcount](const int & i){
    double diff = std::fabs(v1v(i) - v2v(i));
    if(diff > threadlocal_residual) threadlocal_residual = diff;
    if(i == loopcount){ //Replace this if there is a better way to tell if we are on the last iteration.
      local_residual_mutex.lock();
      if(threadlocal_residual>local_residual) local_residual = threadlocal_residual;
      local_residual_mutex.unlock();
    }
  });
*/

  Kokkos::parallel_for(n, FunctorforKokkos(v1v, v2v, local_residual, /*local_residual_mutex,*/ n));

#ifdef HPCG_DETAILED_DEBUG
    HPCG_fout << " Computed, exact, diff = " << v1v(i) << " " << v2v(i) << " " << diff << std::endl;
#endif

#ifndef HPCG_NOMPI
  // Use MPI's reduce function to collect all partial sums
  double global_residual = 0;
  MPI_Allreduce(&local_residual, &global_residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  residual = global_residual;
#else
  residual = local_residual;
#endif

  return(0);
}
