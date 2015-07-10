
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

class NonzerosFunctor{
	public:
	char_1d_type nonzerosInRow;
	non_const_row_map_type rowMap;
	double_1d_type bv, xv, xexactv;
	//All of these are taken from Geometry...
	global_int_t nx, ny, nz;
	global_int_t npx, npy, npz;
	global_int_t ipx, ipy, ipz;
	global_int_t gnx, gny, gnz;
	
	NonzerosFunctor(char_1d_type & nonzerosInRow_, non_const_row_map_type & rowMap_,
		double_1d_type& bv_, double_1d_type& xv_, double_1d_type& xexactv_, const Geometry geom_):
		nonzerosInRow(nonzerosInRow_), rowMap(rowMap_), bv(bv_), xv(xv_), xexactv(xexactv_){
		init(geom_);}

	void init(Geometry geom){
		nx = geom.nx; ny = geom.ny; nz = geom.nz;
		npx = geom.npx; npy = geom.npy; npz = geom.npz;
		ipx = geom.ipx; ipy = geom.ipy; ipz = geom.ipz;
		gnx = nx*npx; gny = ny*npy; gnz = nz*npz;
	}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int& iz, local_int_t& localNumberOfNonzeros_)const{
		global_int_t giz = ipz * nz + iz;
		for(local_int_t iy = 0; iy < ny; iy++){
			global_int_t giy = ipy * ny + iy;
			for(local_int_t ix = 0; ix < nx; ix++){
				global_int_t gix = ipx * nx + ix;
				local_int_t currentLocalRow = iz * nx * ny + iy * nx + ix;
				char numberOfNonzerosInRow = 0;
				for(int sz = -1; sz <= 1; sz++){
					if(giz + sz > -1 && giz + sz < gnz){
						for(int sy = -1; sy <= 1; sy++){
							if(giy + sy > -1 && giy + sy < gny){
								for(int sx = -1; sx <= 1; sx++){
									if(gix + sx > -1 && gix + sx < gnx){
										numberOfNonzerosInRow++;
									}
								}
							}
						}
					}
				}
				nonzerosInRow(currentLocalRow) = numberOfNonzerosInRow;
				rowMap(currentLocalRow + 1) = (local_int_t) numberOfNonzerosInRow;
				localNumberOfNonzeros_ += numberOfNonzerosInRow;
				bv(currentLocalRow) = 26.0 -((double) (numberOfNonzerosInRow -1));
				xv(currentLocalRow) = 0.0;
				xexactv(currentLocalRow) = 1.0;
			}
		}
	}
};

class ScanFunctor{
	public:
	non_const_row_map_type rowMap;
	
	ScanFunctor(non_const_row_map_type rowMap_):
		rowMap(rowMap_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i, local_int_t & upd, bool final)const{
		upd += rowMap(i);
		if(final)
			rowMap(i) = upd;
	}
};

class SetFunctor{
	public:
	non_const_row_map_type rowMap;
	values_type values;
	global_index_type indexMap;
	int_1d_type matrixDiagonal;
	global_int_1d_type localToGlobalMap;
	map_type globalToLocalMap;
	//All of these are taken from Geometry...
	global_int_t nx, ny, nz;
	global_int_t npx, npy, npz;
	global_int_t ipx, ipy, ipz;
	global_int_t gnx, gny, gnz;

	SetFunctor(non_const_row_map_type& rowMap_, values_type& values_, global_index_type& indexMap_,
		int_1d_type& matrixDiagonal_, global_int_1d_type& localToGlobalMap_,
		map_type globalToLocalMap_, Geometry geom_):
		rowMap(rowMap_), values(values_), indexMap(indexMap_), matrixDiagonal(matrixDiagonal_),
		localToGlobalMap(localToGlobalMap_), globalToLocalMap(globalToLocalMap_){
		init(geom_);}

	void init(Geometry geom){
		nx = geom.nx; ny = geom.ny; nz = geom.nz;
		npx = geom.npx; npy = geom.npy; npz = geom.npz;
		ipx = geom.ipx; ipy = geom.ipy; ipz = geom.ipz;
		gnx = nx*npx; gny = ny*npy; gnz = nz*npz;
	}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & iz)const{
		global_int_t giz = ipz * nz + iz;
		for(local_int_t iy = 0; iy < ny; iy++){
			global_int_t giy = ipy * ny + iy;
			for(local_int_t ix = 0; ix < nx; ix++){
				global_int_t gix = ipx * nx + ix;
				local_int_t currentLocalRow = iz * nx * ny + iy * nx + ix;
				global_int_t currentGlobalRow = giz * gnx * gny + giy * gnx + gix;
				globalToLocalMap.insert(currentGlobalRow, currentLocalRow);
				localToGlobalMap(currentLocalRow) = currentGlobalRow;
#ifdef HPCG_DETAILED_DEBUG
				HPCG_fout << " rank, globalRow, localRow = " << A.geom.rank << " " << currentGlobalRow << " " << A.globalToLocalMap.value_at(A.globalToLocalMap.find(currentGlobalRow)) << endl;
#endif
				int cvpIndex = rowMap(currentLocalRow);
				int cipgIndex = cvpIndex;
				for(int sz = -1; sz <= 1; sz++){
					if(giz + sz > -1 && giz + sz < gnz){
						for(int sy = -1; sy <= 1; sy++){
							if(giy + sy > -1 && giy + sy < gny){
								for(int sx = -1; sx <= 1; sx++){
									if(gix + sx > -1 && gix + sx < gnx){
										global_int_t curcol = currentGlobalRow + sz * gnx * gny + sy * gnx + sx;
										if(curcol == currentGlobalRow){
											matrixDiagonal(currentLocalRow) = cvpIndex;
											values(cvpIndex) = 26.0;
											cvpIndex++;
										} else {
											values(cvpIndex) = -1.0;
											cvpIndex++;
										}
										indexMap(cipgIndex) = (local_int_t) curcol;
										cipgIndex++;
									}
								}
							}
						}
					}
				}
				assert(cvpIndex == rowMap(currentLocalRow+1)); // Make sure we are completely filling our row and nothing more.
			}
		}
	}
};

class CoarseNonzerosFunctor{
	public:
	char_1d_type nonzerosInRow;
	non_const_row_map_type rowMap;
	//All of these are taken from Geometry...
	global_int_t nx, ny, nz;
	global_int_t npx, npy, npz;
	global_int_t ipx, ipy, ipz;
	global_int_t gnx, gny, gnz;
	
	CoarseNonzerosFunctor(char_1d_type & nonzerosInRow_, non_const_row_map_type & rowMap_,
		const Geometry geom_): 
		nonzerosInRow(nonzerosInRow_), rowMap(rowMap_){
		init(geom_);}

	void init(Geometry geom){
		nx = geom.nx; ny = geom.ny; nz = geom.nz;
		npx = geom.npx; npy = geom.npy; npz = geom.npz;
		ipx = geom.ipx; ipy = geom.ipy; ipz = geom.ipz;
		gnx = nx*npx; gny = ny*npy; gnz = nz*npz;
	}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int& iz, local_int_t& localNumberOfNonzeros_)const{
		global_int_t giz = ipz * nz + iz;
		for(local_int_t iy = 0; iy < ny; iy++){
			global_int_t giy = ipy * ny + iy;
			for(local_int_t ix = 0; ix < nx; ix++){
				global_int_t gix = ipx * nx + ix;
				local_int_t currentLocalRow = iz * nx * ny + iy * nx + ix;
				char numberOfNonzerosInRow = 0;
				for(int sz = -1; sz <= 1; sz++){
					if(giz + sz > -1 && giz + sz < gnz){
						for(int sy = -1; sy <= 1; sy++){
							if(giy + sy > -1 && giy + sy < gny){
								for(int sx = -1; sx <= 1; sx++){
									if(gix + sx > -1 && gix + sx < gnx){
										numberOfNonzerosInRow++;
									}
								}
							}
						}
					}
				}
				nonzerosInRow(currentLocalRow) = numberOfNonzerosInRow;
				rowMap(currentLocalRow + 1) = (local_int_t) numberOfNonzerosInRow;
				localNumberOfNonzeros_ += numberOfNonzerosInRow;
			}
		}
	}
};

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
#ifndef HPCG_NOMPI
std::cout<< "RUNNING WITH MPI COMPILED" << std::endl;
#endif

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
	int_1d_type matrixDiagonal = int_1d_type("Matrix: matrixDiagonal", localNumberOfRows); // This view will hold the indices to the values along the diagonal

	//CrsMatrix setup
	// These may be too large for performance. I'll resize them later.
	values_type values("CrsMatrix: Values", localNumberOfRows * numberOfNonzerosPerRow); //Replaces matrixValues
	global_index_type indexMap("CrsMatrix: Global Index Map", localNumberOfRows * numberOfNonzerosPerRow); //Replace mtxIndl
	non_const_row_map_type rowMap("CrsMatrix: Row Map", localNumberOfRows + 1); //Psuedo replace nonzerosInRow

	//Initialize Views to be all 0's.
	deep_copy(matrixDiagonal, 0.0);

	// Vectors, Becuase vectors won't have address 0 since I'm not using poniters I can ignore the != 0
	InitializeVector(b, localNumberOfRows);
	InitializeVector(x, localNumberOfRows);
	InitializeVector(xexact, localNumberOfRows);
	// The mirrors are only temporary while we run this method in serial.
	double_1d_type bv = b.values;
	double_1d_type xv = x.values;
	double_1d_type xexactv = xexact.values;
//	A.localToGlobalMap.resize(localNumberOfRows);
	global_int_1d_type localToGlobalMap = global_int_1d_type("Matrix: localToGlobalMap", localNumberOfRows);
	map_type globalToLocalMap = map_type(localNumberOfRows);
	host_map_type host_globalToLocalMap;
	deep_copy(host_globalToLocalMap, globalToLocalMap);
	// Now were to the assign values stage...
	local_int_t localNumberOfNonzeros = 0;
	// Since were using Kokkos::Parallel_for I don't need to make mirrors.
// I'll take care of the maps in the loop that specifies values since this loop doesn't need currentGlobalRow
//  This parallel loop figures out how many nonzeros are in each row.
Kokkos::parallel_reduce(A.geom->nz, NonzerosFunctor(nonzerosInRow, rowMap, bv, xv, xexactv, *A.geom), localNumberOfNonzeros);
// This takes the values we've already filled in rowMap and updates them to reflect the layout we want rowMap to be in.
//FIXME: This isn't actually doing anything aside from changing rowMap to all 0's...
const size_t n = rowMap.dimension_0();
Kokkos::parallel_scan(n, ScanFunctor(rowMap));
// This will finally store the values we wish to use
Kokkos::parallel_for(A.geom->nz, SetFunctor(rowMap, values, indexMap, matrixDiagonal,
	localToGlobalMap, globalToLocalMap, *A.geom));

#ifdef HPCG_DETAILED_DEBUG
	HPCG_fout << "Process " << A.geom.rank << " of " << A.geom.size << " has " << localNumberOfRows << "rows." << endl;
		<< "Process " << A.geom.rank << " of " << A.geom.size << " has " << localNumbverOfNonzeros << " nonzeros." << endl;
#endif

	global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NOMPI
	// Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
	MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
	long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit for MPI call
	MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	totalNumberOfNonzeros = gnnz; // Copy back
#endif
#else
	totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
	// If this assert fails it most likele means that the global_int_t is set to int and should be set to long long
	// This assert is usuall the first to fail as problem size increases beyond the 32-bit integer range.
	assert(totalNumberOfNonzeros > 0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

	assert(rowMap(localNumberOfRows) == (unsigned)localNumberOfNonzeros); // This is here to make sure our rowMap covers all the nonzero values.

	Kokkos::resize(values, localNumberOfNonzeros); // values may have been to long and used more space than necessary
	Kokkos::resize(indexMap, localNumberOfNonzeros); // Same reason as values.
	global_matrix_type globalMatrix = global_matrix_type("Matrix: globalMatrix", localNumberOfRows, localNumberOfRows,
		localNumberOfNonzeros, values, rowMap, indexMap); //local matrix will be made in setup halo.

	A.globalMatrix = globalMatrix;

	A.totalNumberOfRows = totalNumberOfRows;
	A.totalNumberOfNonzeros = totalNumberOfNonzeros;
	A.localNumberOfRows = localNumberOfRows;
	A.localNumberOfColumns = localNumberOfRows;
	A.localNumberOfNonzeros = localNumberOfNonzeros;
	A.nonzerosInRow = nonzerosInRow;
	A.matrixDiagonal = matrixDiagonal;
	A.localToGlobalMap = localToGlobalMap;
	A.globalToLocalMap= globalToLocalMap;
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
	int_1d_type matrixDiagonal = int_1d_type("Matrix: matrixDiagonal", localNumberOfRows); // This view will hold the indices to the values along the diagonal

	//CrsMatrix setup
	// These may be too large for performance. I'll resize them later.
	values_type values("CrsMatrix: Values", localNumberOfRows * numberOfNonzerosPerRow); //Replaces matrixValues
	global_index_type indexMap("CrsMatrix: Global Index Map", localNumberOfRows * numberOfNonzerosPerRow); //Replace mtxIndl
	non_const_row_map_type rowMap("CrsMatrix: Row Map", localNumberOfRows + 1); //Psuedo replace nonzerosInRow

	//Initialize Views to be all 0's.
	deep_copy(matrixDiagonal, 0.0);

//	A.localToGlobalMap.resize(localNumberOfRows);
	global_int_1d_type localToGlobalMap = global_int_1d_type("Matrix: localToGlobalMap", localNumberOfRows);
	map_type globalToLocalMap = map_type(localNumberOfRows);
	// Now were to the assign values stage...
	local_int_t localNumberOfNonzeros = 0;
	// Since were using Kokkos::Parallel_for I don't need to make mirrors.
// I'll take care of the maps in the loop that specifies values since this loop doesn't need currentGlobalRow
//  This parallel loop figures out how many nonzeros are in each row.
Kokkos::parallel_reduce(A.geom->nz, CoarseNonzerosFunctor(nonzerosInRow, rowMap, *A.geom), localNumberOfNonzeros);
// This takes the values we've already filled in rowMap and updates them to reflect the layout we want rowMap to be in.
//TODO: Exclusive Scan would make more sense but the inclusive scan works.
const size_t n = rowMap.dimension_0();
Kokkos::parallel_scan(n, ScanFunctor(rowMap));
// This will finally store the values we wish to use
Kokkos::parallel_for(A.geom->nz, SetFunctor(rowMap, values, indexMap, matrixDiagonal,
	localToGlobalMap, globalToLocalMap, *A.geom));
#ifdef HPCG_DETAILED_DEBUG
	HPCG_fout << "Process " << A.geom.rank << " of " << A.geom.size << " has " << localNumberOfRows << "rows." << endl;
		<< "Process " << A.geom.rank << " of " << A.geom.size << " has " << localNumberOfNonzeros << " nonzeros." << endl;
#endif

	global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NOMPI
	// Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
	MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
	long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit for MPI call
	MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	totalNumberOfNonzeros = gnnz; // Copy back
#endif
#else
	totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
	// If this assert fails it most likele means that the global_int_t is set to int and should be set to long long
	// This assert is usuall the first to fail as problem size increases beyond the 32-bit integer range.
	assert(totalNumberOfNonzeros > 0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

	assert(rowMap(localNumberOfRows) == (unsigned)localNumberOfNonzeros); // This is here to make sure our rowMap covers all the nonzero values.

	Kokkos::resize(values, localNumberOfNonzeros); // values may have been to long and used more space than necessary
	Kokkos::resize(indexMap, localNumberOfNonzeros); // Same reason as values.
	global_matrix_type globalMatrix = global_matrix_type("Matrix: globalMatrix", localNumberOfRows, localNumberOfRows,
		localNumberOfNonzeros, values, rowMap, indexMap); //local matrix will be made in setup halo.

	A.globalMatrix = globalMatrix;

	A.totalNumberOfRows = totalNumberOfRows;
	A.totalNumberOfNonzeros = totalNumberOfNonzeros;
	A.localNumberOfRows = localNumberOfRows;
	A.localNumberOfColumns = localNumberOfRows;
	A.localNumberOfNonzeros = localNumberOfNonzeros;
	A.nonzerosInRow = nonzerosInRow;
	A.matrixDiagonal = matrixDiagonal;
	A.localToGlobalMap = localToGlobalMap;
	A.globalToLocalMap = globalToLocalMap;
	return;
}
