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

using Kokkos::create_mirror_view;
using Kokkos::deep_copy;
/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
 */
void ExchangeHalo(const SparseMatrix & A, Vector & x) {

  // Extract Matrix pieces

	local_int_t localNumberOfRows = A.localNumberOfRows;
	int num_neighbors = A.numberOfSendNeighbors;
	host_local_int_1d_type receiveLength = create_mirror_view(A.receiveLength);
	deep_copy(receiveLength, A.receiveLength);
	host_local_int_1d_type sendLength = create_mirror_view(A.sendLength);
	deep_copy(sendLength, A.sendLength);
	host_int_1d_type neighbors = create_mirror_view(A.neighbors);
	deep_copy(neighbors, A.neighbors);
	double * sendBuffer = A.sendBuffer;
	local_int_t totalToBeSent = A.totalToBeSent;
	host_local_int_1d_type elementsToSend = create_mirror_view(A.elementsToSend);
	deep_copy(elementsToSend, A.elementsToSend);
	
	host_double_1d_type xv = create_mirror_view(x.values);
	deep_copy(xv, x.values);

	int size, rank; // Number of MPI process, My process ID
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

	double * x_external = xv.ptr_on_device() + localNumberOfRows;

	// Post receives first
	// TODO: Thread this loop
	for(int i = 0; i < num_neighbors; i++){
		local_int_t n_recv = receiveLength(i);
		MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors(i), MPI_MY_TAG, MPI_COMM_WORLD, request+i);
		x_external += n_recv;
	}

	//
	// Fill up send buffer
	//

	// TODO: Thread this loop
	for (local_int_t i = 0; i < totalToBeSent; i++) sendBuffer[i] = xv(elementsToSend(i));

	//
	// Send to each neighbor
	//

	// TODO: Thread this loop
	for (int i = 0; i < num_neighbors; i++) {
		local_int_t n_send = sendLength[i];
		MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors(i), MPI_MY_TAG, MPI_COMM_WORLD);
		sendBuffer += n_send;
	}

	//
	// Complete the reads issued above
	//
	
	MPI_Status status;
	// TODO: Thread this loop
	for(int i = 0; i < num_neighbors; i++){
		if(MPI_Wait(request+i, &status)){
			std::exit(-1); // TODO: have better error exit
		}
	}

	delete [] request;
	
	deep_copy(x.values, xv);

	return;
}
#endif // ifndef HPCG_NOMPI
