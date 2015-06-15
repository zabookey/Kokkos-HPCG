
#ifndef HPCG_NOMPI
#include <mpi.h>
#endif

/*
#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif
*/
#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)
#include <fstream>
using std::endl;
#include "hpcg.hpp"
#endif
#include <cassert>

#include "GenerateProblem.hpp"

#include <iostream>

using Kokkos::create_mirror_view;
using Kokkos::deep_copy;
using Kokkos::subview;
using Kokkos::ALL;

/*!
  Routine to read a sparse matrix, right hand side, initial guess, and exact
  solution (as computed by a direct solver).

  @param[in]  geom   data structure that stores the parallel run parameters and the factoring of total number of processes into three dimensional grid
  @param[in]  A      The known system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0 non-zero on entry)

  @see GenerateGeometry
*/

void GenerateProblem(SparseMatrix & A, Vector & b, Vector & x, Vector & xexact){

	// Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
  // below may result in global range values.
  global_int_t nx = A.geom->nx;
  global_int_t ny = A.geom->ny;
  global_int_t nz = A.geom->nz;
  global_int_t npx = A.geom->npx;
  global_int_t npy = A.geom->npy;
  global_int_t npz = A.geom->npz;
  global_int_t ipx = A.geom->ipx;
  global_int_t ipy = A.geom->ipy;
  global_int_t ipz = A.geom->ipz;
  global_int_t gnx = nx*npx;
  global_int_t gny = ny*npy;
  global_int_t gnz = nz*npz;

	local_int_t localNumberOfRows = nx*ny*nz; // This is the size of our subblock
	// If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
	assert(localNumberOfRows > 0); // Throws an exception if the number of rows is less than zero (can happen if int overflow)
	local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

	global_int_t totalNumberOfRows = ((global_int_t) localNumberOfRows) * ((global_int_t) A.geom->size); // Total number of grid points in  mesh
	//If this assert fails, it most likely means taht the global_int_t is set to int and should be set to long long
	assert(totalNumberOfRows>0); // Throw an exception if the number of rows is less than zero (can happen if int overflow)


	// Allocate views... These are all allocated with all of their entries to 0.
	char_1d_type nonzerosInRow = char_1d_type("Matrix: nonzerosInRow", localNumberOfRows);
	double_1d_type matrixDiagonal = double_1d_type("Matrix: matrixDiagonal", localNumberOfRows); // This view will hold the indices to the values along the diagonal
	global_int_2d_type mtxIndG = global_int_2d_type("Matrix: mtxIndG", localNumberOfRows, numberOfNonzerosPerRow);
	local_int_2d_type mtxIndL = local_int_2d_type("Matrix: mtxIndL", localNumberOfRows, numberOfNonzerosPerRow);
	double_2d_type matrixValues = double_2d_type("Matrix: matrixValues", localNumberOfRows, numberOfNonzerosPerRow);

	//Initialize Views to be all 0's.
	deep_copy(matrixDiagonal, 0.0);
	deep_copy(mtxIndG, 0.0);
	deep_copy(mtxIndL, 0.0);
	deep_copy(matrixValues, 0.0);

	// Vectors, Becuase vectors won't have address 0 since I'm not using poniters I can ignore the != 0
	InitializeVector(b, localNumberOfRows);
	InitializeVector(x, localNumberOfRows);
	InitializeVector(xexact, localNumberOfRows);
	// The mirrors are only temporary while we run this method in serial.
	host_double_1d_type bv = create_mirror_view(b.values);
	host_double_1d_type xv = create_mirror_view(x.values);
	host_double_1d_type xexactv = create_mirror_view(xexact.values);
	deep_copy(bv, b.values);
	deep_copy(xv, x.values);
	deep_copy(xexactv, xexact.values);
//	A.localToGlobalMap.resize(localNumberOfRows);
	global_int_1d_type localToGlobalMap = global_int_1d_type("Matrix: localToGlobalMap", localNumberOfRows);

	// Now were to the assign values stage...
	local_int_t localNumberOfNonzeros = 0;
	// Since were using Kokkos::Parallel_for I don't need to make mirrors.
	// FIXME: This needs to use [=] capture list. Problem is A.globalToLocalMap (std::map) and potentially A.localToGlobalMap (std::vector)
//We'll just make this serial for now...
//Mirrors for serial.
host_char_1d_type host_nonzerosInRow = create_mirror_view(nonzerosInRow);
host_double_1d_type host_matrixDiagonal = create_mirror_view(matrixDiagonal);
host_global_int_2d_type host_mtxIndG = create_mirror_view(mtxIndG);
host_local_int_2d_type host_mtxIndL = create_mirror_view(mtxIndL);
host_double_2d_type host_matrixValues = create_mirror_view(matrixValues);
host_global_int_1d_type host_localToGlobalMap = create_mirror_view(localToGlobalMap);
for(local_int_t iz = 0; iz < nx; iz++){
/*	Kokkos::parallel_for(nz, [&](const int & iz)
{*/
	global_int_t giz = ipz * nz + iz;
	for(local_int_t iy = 0; iy < ny; iy++){
		global_int_t giy = ipy * ny + iy;
		for(local_int_t ix = 0; ix < nx; ix++){
			global_int_t gix = ipx * nx + ix;
			local_int_t currentLocalRow = iz * nx * ny + iy * nx + ix;
			global_int_t currentGlobalRow = giz * gnx * gny + giy * gnx + gix;
//		/*	A.globalToLocalMap = */Kokkos::atomic_exchange(& A.globalToLocalMap[currentGlobalRow], currentLocalRow); // I want this to be equivalent to A.globalToLocalMap[currentGlobalRow] = currentLocalRow...
			A.globalToLocalMap[currentGlobalRow] = currentLocalRow;
			host_localToGlobalMap(currentLocalRow) = currentGlobalRow;
#ifdef HPCG_DETAILED_DEBUG
			HPCG_fout << " rank, globalRow, localRow = " << A.geom.rank << " " << currentGlobalRow << " " << A.globalToLocalMap[currentGlobalRow] << endl;
#endif
			char numberOfNonzerosInRow = 0;
			auto currentValuePointer = subview(matrixValues, currentLocalRow, ALL()); // should be double_1d_type.
			auto currentIndexPointerG = subview(mtxIndG, currentLocalRow, ALL()); // should be double_1d_type.
			int cvpIndex = 0;
			int cipgIndex = 0;
      for(int sz = -1; sz <= 1; sz++){
        if(giz + sz > -1 && giz + sz < gnz) {
          for(int sy = -1; sy <= 1; sy++) {
            if(giy + sy > -1 && giy + sy < gny){
              for(int sx = -1; sx <= 1; sx++){
                if(gix + sx > -1 && gix + sx < gnx){
                  global_int_t curcol = currentGlobalRow + sz * gnx * gny + sy * gnx + sx;
									if(curcol == currentGlobalRow){
										matrixDiagonal(currentLocalRow) = cvpIndex;
										currentValuePointer(cvpIndex) = 26.0;
										cvpIndex++;
									} else {
										currentValuePointer(cvpIndex) = -1.0;
										cvpIndex++;
									}
									currentIndexPointerG(cipgIndex) = curcol;
									cipgIndex++;
									numberOfNonzerosInRow++;
								} // end x bounds test
							} // end sx loop
						} // end y bounds test
					} // end sy loop
				} // end z bounds test
			} // end sz loop
			nonzerosInRow(currentLocalRow) = numberOfNonzerosInRow;
// This will be an issue due to changing a const when lambda [=]... Maybe wrap a view around it so I can alter the data in parallel and then mirror it back to localNumberOfNonZeros...
//Serial...			Kokkos::atomic_add(&localNumberOfNonzeros, (local_int_t) numberOfNonzerosInRow);
			localNumberOfNonzeros += numberOfNonzerosInRow;
			bv(currentLocalRow) = 26.0 -((double) (numberOfNonzerosInRow -1));
			xv(currentLocalRow) = 0.0;
			xexactv(currentLocalRow) = 1.0;
		}
	}
}//);
//Copy back the temp mirrors.
deep_copy(nonzerosInRow, host_nonzerosInRow);
deep_copy(matrixDiagonal, host_matrixDiagonal);
deep_copy(mtxIndG, host_mtxIndG);
deep_copy(mtxIndL, host_mtxIndL);
deep_copy(matrixValues, host_matrixValues);
deep_copy(localToGlobalMap, host_localToGlobalMap);
deep_copy(x.values, xv);
deep_copy(b.values, bv);
deep_copy(xexact.values, xexactv);
#ifdef HPCG_DETAILED_DEBUG
	HPCG_fout << "Process " << A.geom.rank << " of " << A.geom.size << " has " << localNumberOfRows << "rows." << endl;
		<< "Process " << A.geom.rank << " of " << A.geom.size << " has " << localNumbverOfNonzeros << " nonzeros." << endl;
#endif

	global_int_t totalNumberOfNonzeros = 0;
#ifdef HPCG_NOMPI
	// Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
	MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
	long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit for MPI call
	MPI_ALLreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	toatlNumberOfNonzeros = gnnz; // Copy back
#endif
#else
	totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
	// If this assert fails it most likele means that the global_int_t is set to int and should be set to long long
	// This assert is usuall the first to fail as problem size increases beyond the 32-bit integer range.
	assert(totalNumberOfNonzeros > 0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

	A.totalNumberOfRows = totalNumberOfRows;
	A.totalNumberOfNonzeros = totalNumberOfNonzeros;
	A.localNumberOfRows = localNumberOfRows;
	A.localNumberOfColumns = localNumberOfRows;
	A.localNumberOfNonzeros = localNumberOfNonzeros;
	A.nonzerosInRow = nonzerosInRow;
	A.mtxIndG = mtxIndG;
	A.mtxIndL = mtxIndL;
	A.matrixValues = matrixValues;
	A.matrixDiagonal = matrixDiagonal;
	A.localToGlobalMap = localToGlobalMap;

	return;
}



/*
	This method is here to replace the old way of passing vector pointers and checking if they pointed to 0
	It is identical to the other method aside from getting rid of anything to do with x, b, and xexact.
*/
void GenerateProblem(SparseMatrix & A){
	// Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
  // below may result in global range values.
  global_int_t nx = A.geom->nx;
  global_int_t ny = A.geom->ny;
  global_int_t nz = A.geom->nz;
  global_int_t npx = A.geom->npx;
  global_int_t npy = A.geom->npy;
  global_int_t npz = A.geom->npz;
  global_int_t ipx = A.geom->ipx;
  global_int_t ipy = A.geom->ipy;
  global_int_t ipz = A.geom->ipz;
  global_int_t gnx = nx*npx;
  global_int_t gny = ny*npy;
  global_int_t gnz = nz*npz;

	local_int_t localNumberOfRows = nx*ny*nz; // This is the size of our subblock
	// If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
	assert(localNumberOfRows > 0); // Throws an exception if the number of rows is less than zero (can happen if int overflow)
	local_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

	global_int_t totalNumberOfRows = ((global_int_t) localNumberOfRows) * ((global_int_t) A.geom->size); // Total number of grid points in  mesh
	//If this assert fails, it most likely means taht the global_int_t is set to int and should be set to long long
	assert(totalNumberOfRows>0); // Throw an exception if the number of rows is less than zero (can happen if int overflow)


	// Allocate views... These are all allocated with all of their entries to 0.
	char_1d_type nonzerosInRow = char_1d_type("Matrix: nonzerosInRow", localNumberOfRows);
	double_1d_type matrixDiagonal = double_1d_type("Matrix: matrixDiagonal", localNumberOfRows); // This view will hold the indices to the values along the diagonal
	global_int_2d_type mtxIndG = global_int_2d_type("Matrix: mtxIndG", localNumberOfRows, numberOfNonzerosPerRow);
	local_int_2d_type mtxIndL = local_int_2d_type("Matrix: mtxIndL", localNumberOfRows, numberOfNonzerosPerRow);
	double_2d_type matrixValues = double_2d_type("Matrix: matrixValues", localNumberOfRows, numberOfNonzerosPerRow);

//	Initialize Views to be all 0's
	deep_copy(matrixDiagonal, 0.0);
	deep_copy(mtxIndG, 0.0);
	deep_copy(mtxIndL, 0.0);
	deep_copy(matrixValues, 0.0);
//	A.localToGlobalMap.resize(localNumberOfRows);
	global_int_1d_type localToGlobalMap = global_int_1d_type("Matrix: localToGlobalMap", localNumberOfRows);

	// Now were to the assign values stage...
	local_int_t localNumberOfNonzeros = 0;
	// Since were using Kokkos::Parallel_for I don't need to make mirrors.
	// FIXME: Lambda capture list needs to be [=] Problem due to (std::map) and possibly (std::vector)
//We'll just make this serial for now...
//Mirrors for serial.
host_char_1d_type host_nonzerosInRow = create_mirror_view(nonzerosInRow);
host_double_1d_type host_matrixDiagonal = create_mirror_view(matrixDiagonal);
host_global_int_2d_type host_mtxIndG = create_mirror_view(mtxIndG);
host_local_int_2d_type host_mtxIndL = create_mirror_view(mtxIndL);
host_double_2d_type host_matrixValues = create_mirror_view(matrixValues);
host_global_int_1d_type host_localToGlobalMap = create_mirror_view(localToGlobalMap);
for(local_int_t iz = 0; iz < nx; iz++){
/*	Kokkos::parallel_for(nz, [&](const int & iz)
{*/
	global_int_t giz = ipz * nz + iz;
	for(local_int_t iy = 0; iy < ny; iy++){
		global_int_t giy = ipy * ny + iy;
		for(local_int_t ix = 0; ix < nx; ix++){
			global_int_t gix = ipx * nx + ix;
			local_int_t currentLocalRow = iz * nx * ny + iy * nx + ix;
			global_int_t currentGlobalRow = giz * gnx * gny + giy * gnx + gix;
//			Kokkos::atomic_exchange(& A.globalToLocalMap[currentGlobalRow], currentLocalRow);
			A.globalToLocalMap[currentGlobalRow] = currentLocalRow;
			host_localToGlobalMap(currentLocalRow) = currentGlobalRow;
#ifdef HPCG_DETAILED_DEBUG
			HPCG_fout << " rank, globalRow, localRow = " << A.geom.rank << " " << currentGlobalRow << " " << A.globalToLocalMap[currentGlobalRow] << endl;
#endif
			char numberOfNonzerosInRow = 0;
			auto currentValuePointer = subview(matrixValues, currentLocalRow, ALL()); // should be double_1d_type.
			auto currentIndexPointerG = subview(mtxIndG, currentLocalRow, ALL());
			int cvpIndex = 0;
			int cipgIndex = 0;
      for(int sz = -1; sz <= 1; sz++){
        if(giz + sz > -1 && giz + sz < gnz) {
          for(int sy = -1; sy <= 1; sy++) {
            if(giy + sy > -1 && giy + sy < gny){
              for(int sx = -1; sx <= 1; sx++){
                if(gix + sx > -1 && gix + sx < gnx){
                  global_int_t curcol = currentGlobalRow + sz * gnx * gny + sy * gnx + sx;
									if(curcol == currentGlobalRow){
										matrixDiagonal(currentLocalRow) = cvpIndex;
										currentValuePointer(cvpIndex) = 26.0;
										cvpIndex++;
									} else {
										currentValuePointer(cvpIndex) = -1.0;
										cvpIndex++;
									}
									currentIndexPointerG(cipgIndex) = curcol;
									cipgIndex++;
									numberOfNonzerosInRow++;
								} // end x bounds test
							} // end sx loop
						} // end y bounds test
					} // end sy loop
				} // end z bounds test
			} // end sz loop
			nonzerosInRow(currentLocalRow) = numberOfNonzerosInRow;
			// This will be an issue due to changing a const when lambda [=]... Maybe wrap a view around it so I can alter the data in parallel and then mirror it back to localNumberOfNonZeros...
//			Kokkos::atomic_add(&localNumberOfNonzeros, (local_int_t) numberOfNonzerosInRow);
			localNumberOfNonzeros += numberOfNonzerosInRow;
		}
	}
}//);
//Copy back the temp mirrors.
deep_copy(nonzerosInRow, host_nonzerosInRow);
deep_copy(matrixDiagonal, host_matrixDiagonal);
deep_copy(mtxIndG, host_mtxIndG);
deep_copy(mtxIndL, host_mtxIndL);
deep_copy(matrixValues, host_matrixValues);
deep_copy(localToGlobalMap, host_localToGlobalMap);
#ifdef HPCG_DETAILED_DEBUG
	HPCG_fout << "Process " << A.geom.rank << " of " << A.geom.size << " has " << localNumberOfRows << "rows." << endl;
		<< "Process " << A.geom.rank << " of " << A.geom.size << " has " << localNumbverOfNonzeros << " nonzeros." << endl;
#endif

	global_int_t totalNumberOfNonzeros = 0;
#ifdef HPCG_NOMPI
	// Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
	MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
	long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit for MPI call
	MPI_ALLreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	toatlNumberOfNonzeros = gnnz; // Copy back
#endif
#else
	totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
	// If this assert fails it most likele means that the global_int_t is set to int and should be set to long long
	// This assert is usuall the first to fail as problem size increases beyond the 32-bit integer range.
	assert(totalNumberOfNonzeros > 0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

	A.totalNumberOfRows = totalNumberOfRows;
	A.totalNumberOfNonzeros = totalNumberOfNonzeros;
	A.localNumberOfRows = localNumberOfRows;
	A.localNumberOfColumns = localNumberOfRows;
	A.localNumberOfNonzeros = localNumberOfNonzeros;
	A.nonzerosInRow = nonzerosInRow;
	A.mtxIndG = mtxIndG;
	A.mtxIndL = mtxIndL;
	A.matrixValues = matrixValues;
	A.matrixDiagonal = matrixDiagonal;
	A.localToGlobalMap = localToGlobalMap;

	return;
}
