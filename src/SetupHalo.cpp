/*!
 @file SetupHalo.cpp

 HPCG routine
 */

#ifndef HPCG_NOMPI
#include <mpi.h>
#include <map>
#include <set>
#endif
/*
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
*/
#ifdef HPCG_DETAILED_DEBUG
#include <fstream>
using std::endl;
#include "hpcg.hpp"
#include <cassert>
#endif

#include "SetupHalo.hpp"
#include "mytimer.hpp"

using Kokkos::create_mirror_view;
using Kokkos::deep_copy;
/*!
  Prepares system matrix data structure and creates data necessary necessary
  for communication of boundary values of this process.

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix

  @see ExchangeHalo
*/

class NoMPIFunctor{
	local_int_2d_type mtxIndL;
	const_global_int_2d_type mtxIndG;
	const_char_1d_type nonzerosInRow;
	
	NoMPIFunctor(local_int_2d_type &mtxIndL_, const_global_int_2d_type &mtxIndG_,
		const_char_1d_type nonzerosInRow_):
		mtxIndL(mtxIndL_), mtxIndG(mtxIndG_), nonzerosInRow(nonzerosInRow_)
		{}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i){
		int cur_nnz = nonzerosInRow(i);
		for(int j = 0; j < cur_nnz; j++) mtxIndL(i, j) = mtxIndG(i, j);
	}
};

void SetupHalo(SparseMatrix & A) {

  // Extract Matrix pieces
	//I'm not sure whether or not I'll need to mirror these...
	local_int_t localNumberOfRows = A.localNumberOfRows;
	char_1d_type nonzerosInRow = A.nonzerosInRow;
	global_int_2d_type mtxIndG = A.mtxIndG;
	local_int_2d_type mtxIndL = A.mtxIndL;
	global_int_1d_type localToGlobalMap = A.localToGlobalMap;

#ifdef HPCG_NOMPI  // In the non-MPI case we simply copy global indices to local index storage

	Kokkos::parallel_for(localNumberOfRows, NoMPIFunctor(mtxIndL, mtxIndG, nonzerosInRow));

#else // Run this section if compiling for MPI

  // Scan global IDs of the nonzeros in the matrix.  Determine if the column ID matches a row ID.  If not:
  // 1) We call the ComputeRankOfMatrixRow function, which tells us the rank of the processor owning the row ID.
  //  We need to receive this value of the x vector during the halo exchange.
  // 2) We record our row ID since we know that the other processor will need this value from us, due to symmetry.

  std::map< int, std::set< global_int_t> > sendList, receiveList;
  typedef std::map< int, std::set< global_int_t> >::iterator map_iter;
  typedef std::set<global_int_t>::iterator set_iter;
  std::map< local_int_t, local_int_t > externalToLocalMap;

  // TODO: With proper critical and atomic regions, this loop could be threaded, but not attempting it at this time
	// Mirror nonzerosInRow and mtxIndG and create a scope so they go away after we deep_copy them back.
{	host_char_1d_type host_nonzerosInRow = create_mirror_view(nonzerosInRow);
	host_global_int_2d_type host_mtxIndG = create_mirror_view(mtxIndG);
	host_global_int_1d_type host_localToGlobalMap = create_mirror_view(localToGlobalMap);
  for (local_int_t i=0; i< localNumberOfRows; i++) {
    global_int_t currentGlobalRow = host_localToGlobalMap(i);
    for (int j=0; j< host_nonzerosInRow(i); j++) {
      global_int_t curIndex = host_mtxIndG(i, j);
      int rankIdOfColumnEntry = ComputeRankOfMatrixRow(*A.geom, curIndex);
#ifdef HPCG_DETAILED_DEBUG
      HPCG_fout << "rank, row , col, globalToLocalMap[col] = " << A.geom.rank << " " << currentGlobalRow << " "
          << curIndex << " " << A.globalToLocalMap[curIndex] << endl;
#endif
      if (A.geom->rank!=rankIdOfColumnEntry) {// If column index is not a row index, then it comes from another processor
        receiveList[rankIdOfColumnEntry].insert(curIndex);
        sendList[rankIdOfColumnEntry].insert(currentGlobalRow); // Matrix symmetry means we know the neighbor process wants my value
      }
    }
  }
	// deep_copy the mirrors back.
	deep_copy(nonzerosInRow, host_nonzerosInRow);
	deep_copy(mtxIndG, host_mtxIndG);}
  // Count number of matrix entries to send and receive
  local_int_t totalToBeSent = 0;
  for (map_iter curNeighbor = sendList.begin(); curNeighbor != sendList.end(); ++curNeighbor) {
    totalToBeSent += (curNeighbor->second).size();
  }
  local_int_t totalToBeReceived = 0;
  for (map_iter curNeighbor = receiveList.begin(); curNeighbor != receiveList.end(); ++curNeighbor) {
    totalToBeReceived += (curNeighbor->second).size();
  }

#ifdef HPCG_DETAILED_DEBUG
  // These are all attributes that should be true, due to symmetry
  HPCG_fout << "totalToBeSent = " << totalToBeSent << " totalToBeReceived = " << totalToBeReceived << endl;
  assert(totalToBeSent==totalToBeReceived); // Number of sent entry should equal number of received
  assert(sendList.size()==receiveList.size()); // Number of send-to neighbors should equal number of receive-from
  // Each receive-from neighbor should be a send-to neighbor, and send the same number of entries
  for (map_iter curNeighbor = receiveList.begin(); curNeighbor != receiveList.end(); ++curNeighbor) {
    assert(sendList.find(curNeighbor->first)!=sendList.end());
    assert(sendList[curNeighbor->first].size()==receiveList[curNeighbor->first].size());
  }
#endif

  // Build the arrays and lists needed by the ExchangeHalo function.
  double * sendBuffer = new double[totalToBeSent];
  local_int_1d_type elementsToSend = local_int_1d_type("Matrix: elementsToSend", totalToBeSent);
  int_1d_type neighbors = int_1d_type("Matrix: neighbors", sendList.size());
  local_int_1d_type receiveLength = local_int_1d_type("Matrix: receiveLength", receiveList.size());
  local_int_1d_type sendLength = local_int_1d_type("Matrix: sendLength", sendList.size());
  int neighborCount = 0;
  local_int_t receiveEntryCount = 0;
  local_int_t sendEntryCount = 0;
	// Mirror the views used below. Create a scope so the mirrors automitically dealloacte after deep_copy
{	host_local_int_1d_type host_elementsToSend = create_mirror_view(elementsToSend);
	host_int_1d_type host_neighbors = create_mirror_view(neighbors);
	host_local_int_1d_type host_receiveLength = create_mirror_view(receiveLength);
	host_local_int_1d_type host_sendLength = create_mirror_view(sendLength);
  for (map_iter curNeighbor = receiveList.begin(); curNeighbor != receiveList.end(); ++curNeighbor, ++neighborCount) {
    int neighborId = curNeighbor->first; // rank of current neighbor we are processing
    host_neighbors(neighborCount) = neighborId; // store rank ID of current neighbor
    host_receiveLength(neighborCount) = receiveList[neighborId].size();
    host_sendLength(neighborCount) = sendList[neighborId].size(); // Get count if sends/receives
    for (set_iter i = receiveList[neighborId].begin(); i != receiveList[neighborId].end(); ++i, ++receiveEntryCount) {
      externalToLocalMap[*i] = localNumberOfRows + receiveEntryCount; // The remote columns are indexed at end of internals
    }
    for (set_iter i = sendList[neighborId].begin(); i != sendList[neighborId].end(); ++i, ++sendEntryCount) {
      //if (geom.rank==1) HPCG_fout << "*i, globalToLocalMap[*i], sendEntryCount = " << *i << " " << A.globalToLocalMap[*i] << " " << sendEntryCount << endl;
      host_elementsToSend(sendEntryCount) = A.globalToLocalMap[*i]; // store local ids of entry to send
    }
  }
	// deep_copy the mirrors back.
	deep_copy(elementsToSend, host_elementsToSend);
	deep_copy(neighbors, host_neighbors);
	deep_copy(receiveLength, host_receiveLength);
	deep_copy(sendLength, host_sendLength);}
  // Convert matrix indices to local IDs
	//FIXME: Lambda capture list needs to be [=] Problem is A.globalToLocalMap (std::map) and externalToLocalMap (std::map)
	Kokkos::parallel_for(localNumberOfRows, [&](const int & i){ // This is going to have some issues due to some parts being on a different device.
		for (int j = 0; j < nonzerosInRow(i); j++) {
			global_int_t curIndex = mtxIndG(i, j);
			int rankIdOfColumnEntry = ComputeRankOfMatrixRow(*A.geom, curIndex);
			if (A.geom->rank == rankIdOfColumnEntry){
				mtxIndL(i, j) = A.globalToLocalMap[curIndex];
			} else {
				mtxIndL(i, j) = externalToLocalMap[curIndex];
			}
		}
	});

  // Store contents in our matrix struct
  A.numberOfExternalValues = externalToLocalMap.size();
  A.localNumberOfColumns = A.localNumberOfRows + A.numberOfExternalValues;
  A.numberOfSendNeighbors = sendList.size();
  A.totalToBeSent = totalToBeSent;
  A.elementsToSend = elementsToSend;
  A.neighbors = neighbors;
  A.receiveLength = receiveLength;
  A.sendLength = sendLength;
  A.sendBuffer = sendBuffer;
	A.optimizationData = (void *) new double[A.numberOfExternalValues];

#ifdef HPCG_DETAILED_DEBUG
  HPCG_fout << " For rank " << A.geom->rank << " of " << A.geom->size << ", number of neighbors = " << A.numberOfSendNeighbors << endl;
  for (int i = 0; i < A.numberOfSendNeighbors; i++) {
    HPCG_fout << "     rank " << A.geom->rank << " neighbor " << neighbors[i] << " send/recv length = " << sendLength[i] << "/" << receiveLength[i] << endl;
    for (local_int_t j = 0; j<sendLength[i]; ++j)
      HPCG_fout << "       rank " << A.geom->rank << " elementsToSend[" << j << "] = " << elementsToSend[j] << endl;
  }
#endif

#endif // ifndef HPCG_NOMPI

  return;
}
