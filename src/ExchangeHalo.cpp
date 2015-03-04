
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ExchangeHalo.cpp

 HPCG routine
 */

#ifndef HPCG_NOMPI  // Compile this routine only if running in parallel
#include <mpi.h>
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include <cstdlib>

//Debugging include
#include "hpcg.hpp"

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
void ExchangeHalo(const SparseMatrix & A, Vector & x) {

  // Extract Matrix pieces

  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  double * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend = A.elementsToSend;

  kokkos_type xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  int MPI_MY_TAG = 99;

  MPI_Request * request = new MPI_Request[num_neighbors];

  //
  // Externals are at end of locals
  //
  // double * x_external = (double *) xv + localNumberOfRows;
   double * x_external = xv.ptr_on_device() + localNumberOfRows;

//	double * x_external = (double *) A.optimizationData;
  // Post receives first
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request+i);
    x_external += n_recv;
  }


  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv(elementsToSend[i]);

//This produces way too much output. Maybe it would be better to decrease the problem size
/*
	for(local_int_t i = 0; i < totalToBeSent; i++){
		if(rank == 0)
			HPCG_fout<<"Rank: 0    Location: " << i << "    Value: " << xv(elementsToSend[i]) << std::endl;
	}
*/
//if(rank == 0) HPCG_fout << xv(elementsToSend[1]) << std::endl;
  //
  // Send to each neighbor
  //

  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_send = sendLength[i];
    MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
    sendBuffer += n_send;
  }

  //
  // Complete the reads issued above
  //

  MPI_Status status;
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    if ( MPI_Wait(request+i, &status) ) {
      std::exit(-1); // TODO: have better error exit
    }
  }
/*
	//Used to add the exchanged values at the end of our vector.
	for(int i = 0; i < num_neighbors; i++){
		xv(x.localLength - num_neighbors + i) = x_external[i];
	}
*/
//Irina commented out
 /*	for(int i = 0; i < num_neighbors; i++){
		xv(localNumberOfRows + i) = x_external[i];
		if(rank == 0) HPCG_fout << xv(localNumberOfRows + i) << std::endl;
	}*/
/*	
	Kokkos::parallel_for(num_neighbors, [=](const int & i){
		xv(localNumberOfRows + i) = x_external[i];
	});
*/
  delete [] request;


  return;
}
#endif // ifndef HPCG_NOMPI
