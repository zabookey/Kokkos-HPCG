#ifndef LEVELS_HPP
#define LEVELS_HPP

#include "KokkosSetup.hpp"

struct LevelScheduler{
// Number of levels needed for the matrix that owns this
	int f_numberOfLevels;
	int b_numberOfLevels;
// Forward sweep data and backward sweep data
// map gives us which indixes in _lev_ind are in each level and _lev_ind contains the row numbers.
	host_local_int_1d_type f_lev_map;
	local_int_1d_type f_lev_ind;
	host_local_int_1d_type b_lev_map;
	local_int_1d_type b_lev_ind;
// Simple view of length number of rows that holds what level each row is in.
	local_int_1d_type f_row_level;
	local_int_1d_type b_row_level;
};
#endif/*

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
