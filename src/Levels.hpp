#ifndef LEVELS_HPP
#define LEVELS_HPP

#include "KokkosSetup.hpp"

struct LevelScheduler{
// Number of levels needed for the matrix that owns this
	int f_numberOfLevels;
	int b_numberOfLevels;
// Forward sweep data and backward sweep data
// map gives us which indixes in _lev_ind are in each level and _lev_ind contains the row numbers.
	local_int_1d_type f_lev_map;
	local_int_1d_type f_lev_ind;
	local_int_1d_type b_lev_map;
	local_int_1d_type b_lev_ind;
// Simple view of length number of rows that holds what level each row is in.
	local_int_1d_type f_row_level;
	local_int_1d_type b_row_level;
};
#endif
