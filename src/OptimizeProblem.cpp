/*!
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
#include <cassert>
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/

#ifdef Option_0

using Kokkos::create_mirror_view;
using Kokkos::deep_copy;

class fillColorsMap{
	public:
	local_int_1d_type colors_map;
	local_int_1d_type colors;

	fillColorsMap(const local_int_1d_type& colors_map_, const local_int_1d_type& colors_):
		colors_map(colors_map_), colors(colors_){}

	// This fills colors_map(i) with the number of indices with color i. Parallel scan will make this the appropriate map.
	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		int color = i+1;
		int total = 0;
		for(unsigned j = 0; j < colors.dimension_0(); j++)
			if(colors(j) == color) total++;
		colors_map(color) = total;
	}
};

class mapScan{
	public:
	local_int_1d_type colors_map;

	mapScan(const local_int_1d_type& colors_map_):
		colors_map(colors_map_){}

// Parallel scan that finishes off setting up colors_map.
	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i, local_int_t & upd, bool final)const{
		upd += colors_map(i);
		if(final)
			colors_map(i) = upd;
	}
};

class fillColorsInd{
	public:
	local_int_1d_type colors_ind;
	local_int_1d_type colors_map;
	local_int_1d_type colors;

	fillColorsInd(local_int_1d_type& colors_ind_, local_int_1d_type& colors_map_,
		local_int_1d_type colors_):
		colors_ind(colors_ind_), colors_map(colors_map_), colors(colors_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		int color = i+1; // Colors start at 1 and i starts at 0.
		int start = colors_map(i);
		for(unsigned j = 0; j < colors.dimension_0(); j++){
			if(colors(j) == color){
				colors_ind(start) = j;
				start++;
			}
		}
		assert(start == colors_map(i+1));//Make sure we only fill up exactly our color. Nothing more nothing less.
	}
};

	void LevelScheduler(SparseMatrix & A){
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

		f_numberOfLevels = 0;
		b_numberOfLevels = 0;
// Grab the parts of the matrix A that we need.
		row_map_type matrixRowMap = A.localMatrix.graph.row_map;
		local_index_type matrixEntries = A.localMatrix.graph.entries;
		int_1d_type matrixDiagonal = A.matrixDiagonal;
		local_int_t nrows = A.localNumberOfRows;
// Allocate the views that need to be allocated right now
		f_row_level = local_int_1d_type("f_row_level", nrows);
		b_row_level = local_int_1d_type("b_row_level", nrows);
// Since this will run in Serial at least for now, create the necessary mirrors for filling row_level
		row_map_type::HostMirror host_matrixRowMap = create_mirror_view(matrixRowMap);
		deep_copy(host_matrixRowMap, matrixRowMap);
		local_index_type::HostMirror host_matrixEntries = create_mirror_view(matrixEntries);
		deep_copy(host_matrixEntries, matrixEntries);
		local_int_1d_type::HostMirror host_f_row_level = create_mirror_view(f_row_level);
		local_int_1d_type::HostMirror host_b_row_level = create_mirror_view(b_row_level);
// Start by taking care of f_row_level
		for(int i = 0; i < nrows; i++){
			int depth = 0;
	// Just doing from beginning of row to the diagonal for the forward.
			int start = host_matrixRowMap(i);
			int end = host_matrixRowMap(i+1);
			for(int j = start; j < end; j++){
				int col = host_matrixEntries(j);
				if((col < i) && (host_f_row_level(col) > depth))
					depth = host_f_row_level(col);
			}
			depth++;
			if(depth > f_numberOfLevels) f_numberOfLevels = depth;
			host_f_row_level(i) = depth;
		}
		deep_copy(f_row_level, host_f_row_level); // Copy the host back to the device. I shouldn't need to modify the host anymore so this is fine
// Take care of b_row_level
		for(int i = nrows - 1; i >= 0; i--){
			int depth = 0;
			int start = host_matrixRowMap(i);
			int end = host_matrixRowMap(i+1);
			for(int j = start; j < end; j++){
				int col = host_matrixEntries(j);
				if((col > i) && (host_b_row_level(col) > depth))
					depth = host_b_row_level(col);
			}
			depth++;
			if(depth > b_numberOfLevels) b_numberOfLevels = depth;
			host_b_row_level(i) = depth;
		}
		std::cout<< "Num f_levels: " << f_numberOfLevels << "   Num b_levels: " << b_numberOfLevels << std::endl;
		deep_copy(b_row_level, host_b_row_level);
// Set up f_lev_map and f_lev_ind
		f_lev_map = local_int_1d_type("f_lev_map", f_numberOfLevels+1);
		f_lev_ind = local_int_1d_type("f_lev_ind", nrows);
// Fill up f_lev_map to prepare for scan
		Kokkos::parallel_for(f_numberOfLevels, fillColorsMap(f_lev_map, f_row_level));
// Do the parallel scan on f_lev_map
		Kokkos::parallel_scan(f_numberOfLevels+1, mapScan(f_lev_map));
// Fill our f_lev_ind now.
		Kokkos::parallel_for(f_numberOfLevels, fillColorsInd(f_lev_ind, f_lev_map, f_row_level));
// Set up b_lev_map and b_lev_ind
		b_lev_map = local_int_1d_type("b_lev_map", b_numberOfLevels+1);
		b_lev_ind = local_int_1d_type("b_lev_ind", nrows);
// Fill up b_lev_map to prepare for scan
		Kokkos::parallel_for(b_numberOfLevels, fillColorsMap(b_lev_map, b_row_level));
// Do the parallel scan on f_lev_map
		Kokkos::parallel_scan(b_numberOfLevels+1, mapScan(b_lev_map));
// Fill our b_lev_ind now.
		Kokkos::parallel_for(b_numberOfLevels, fillColorsInd(b_lev_ind, b_lev_map, b_row_level));

		A.levels.f_numberOfLevels = f_numberOfLevels;
		A.levels.b_numberOfLevels = b_numberOfLevels;
		A.levels.f_lev_map = f_lev_map;
		A.levels.f_lev_ind = f_lev_ind;
		A.levels.b_lev_map = b_lev_map;
		A.levels.b_lev_ind = b_lev_ind;
		A.levels.f_row_level = f_row_level;
		A.levels.b_row_level = b_row_level;
	}
#endif

#ifdef Option_1
class Coloring{
	public:
	typedef local_int_t Ordinal;
	typedef execution_space ExecSpace;
  typedef typename Kokkos::View<Ordinal *, ExecSpace> array_type;
  typedef typename Kokkos::View<Ordinal, ExecSpace> ordinal_type;

  Ordinal _size;
  array_type _idx;
  array_type _adj;
  array_type _colors;
  conflict_type  _conflictType; // Choose at run-time
  array_type _vertexList;
  array_type _recolorList;
  ordinal_type _vertexListLength;  // 0-dim Kokkos::View, so really Ordinal
  ordinal_type _recolorListLength; // 0-dim Kokkos::View, so really Ordinal	

	Coloring(Ordinal nvtx, array_type idx, array_type adj, array_type colors):
		_size(nvtx), _idx(idx), _adj(adj), _colors(colors){
		// vertexList contains all initial vertices
		_vertexList = array_type("vertexList", nvtx);
		_vertexListLength = ordinal_type("vertexListLength");
		_vertexListLength = nvtx;
		// Initialize _vertexList (natural order)
		functorInitList<Ordinal, ExecSpace> init(_vertexList);
		Kokkos::parallel_for(nvtx, init);

		// Vertices to recolor. Will swap with vertexList
		_recolorList = array_type("recolorList", nvtx);
		_recolorListLength = ordinal_type("recolorListLength");
		_recolorListLength() = 0;
	}

	void color(bool useConflictList, bool serialConflictResolution, bool ticToc){
		Ordinal numUncolored = _size; // on host
		double t, total = 0.0;
		Kokkos::Impl::Timer timer;

		if(useConflictList)
			_conflictType = CONFLICT_LIST;

		// While vertices to color, do speculative coloring.
		int iter = 0;
		for(iter = 0; (iter<20) && (numUncolored>0); iter++){
			std::cout<< "Start iteration " << iter << std::endl;

			// First color greedy speculatively, some conflicts expected
			this -> colorGreedy();
			ExecSpace::fence();
			if(ticToc){
				t = timer.seconds();
				total += t;
				std::cout << "Time speculative greedy phase " << iter << " : " << std::endl;
				timer.reset();
			}

#ifdef DEBUG
			// UVM required - will be slow!
			printf("\n 100 first vertices: ");
			for(int i = 0; i < 100; i++){
				printf(" %i", _colors[i]);
			}
			printf("\n");
#endif

			// Check for conflicts (parallel), find vertices to recolor
			numUncolored = this -> findConflicts();

			ExecSpace::fence();
			if(ticToc){
			t = timer.seconds();
			total += t;
			std::cout << "Time conflict detection " << iter << " : " << t << std::endl;
			timer.reset();
			}
			if (serialConflictResolution) break; // Break after first iteration

			if(_conflictType == CONFLICT_LIST){
				array_type temp = _vertexList;
				_vertexList = _recolorList;
				_vertexListLength() = _recolorListLength();
				_recolorList = temp;
				_recolorListLength() = 0;
			}
		}

		std::cout << "Number of coloring iterations: " << iter << std::endl;

		if(numUncolored > 0){
			// Resolve conflicts by recolor in serial
			this -> resolveConflicts();
			ExecSpace::fence();
			if(ticToc){
				t = timer.seconds();
				total += t;
				std::cout << "Time conflict resolution: " << t << std::endl;
				std::cout << "Total time: " << total << std::endl;
			}
		}
	}

	void colorGreedy(){
		Ordinal chunkSize = 8; // Process chunkSize vertices in one chunk
		if(_vertexListLength < 100*chunkSize)
			chunkSize = 1;

		functorGreedyColor<Ordinal, ExecSpace> gc(_idx, _adj, _colors, _vertexList, _vertexListLength, chunkSize);
		Kokkos::parallel_for(_vertexListLength/chunkSize+1, gc);	
	}

	Ordinal findConflicts(){
		functorFindConflicts<Ordinal, ExecSpace> conf(_idx, _adj, _colors, _vertexList, _recolorList, _recolorListLength, _conflictType);
		Ordinal numUncolored;
		Kokkos::parallel_reduce(_vertexListLength(), conf, numUncolored);
		std::cout<< "Number of uncolored vertices: " << numUncolored << std::endl;
#ifdef DEBUG
		if(_conflictType == CONFLICT_LIST)
			std::cout << "findConflicts: recolorListLength = " << _recolorListLength() << std::endl;
#endif
		return numUncolored;
	}

	void resolveConflicts(){
	// This method is in serial so it will need a bit of reworking to be used on Cuda.
		// Compute maxColor.
		const int maxColor = 255; // Guess, since too expensive to loop over nvtx
		
		int forbidden[maxColor+1];
		Ordinal i = 0;
		for(Ordinal k = 0; k < _size; k++){
			if(_conflictType == CONFLICT_LIST){
				if(k == _recolorListLength()) break;
				i = _recolorList[k];
			}
			else {
				// Check for uncolored vertices
				i = k;
				if (_colors[i] > 0) continue;
			}

			// recolor vertex i with smallest available color
			
			// check neighbors
			for(Ordinal j = _idx[i]; j < _idx[i+1]; j++){
				if(_adj[j] == i) continue; // Skip self-loops
				forbidden[_colors[_adj[j]]] = i;
			}
			// color vertex i with smallest available color
			int c=1;
			while((forbidden[c] == i) && c<= maxColor) c++;
			_colors[i] = c;
		}
	}

	Ordinal getNumColors() {
      Ordinal maxColor=0;
      // TODO: parallel_reduce? This produced strange results... So instead we'll use mirrors and only call this method once and store the result
			array_type::HostMirror _colors_host = Kokkos::create_mirror_view(_colors);
			Kokkos::deep_copy(_colors_host, _colors);
			for(int i = 0; i < _size; i++)
				if(_colors_host(i) > maxColor) maxColor = _colors_host(i);
			return maxColor;
    }
};

class fillIdx{
	public:
	Coloring::array_type idx;
	row_map_type row_map;

	fillIdx(Coloring::array_type& idx_, row_map_type& row_map_):
		idx(idx_), row_map(row_map_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		idx(i) = row_map(i);
	}
};

class fillAdj{
	public:
	Coloring::array_type adj;
	local_index_type indices;

	fillAdj(Coloring::array_type& adj_, local_index_type& indices_):
		adj(adj_), indices(indices_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		adj(i) = indices(i);
	}
};

class fillColorsMap{
	public:
	local_int_1d_type colors_map;
	Coloring::array_type colors;

	fillColorsMap(const local_int_1d_type& colors_map_, const Coloring::array_type& colors_):
		colors_map(colors_map_), colors(colors_){}

	// This fills colors_map(i) with the number of indices with color i. Parallel scan will make this the appropriate map.
	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		int color = i+1; // Since i starts at 0 and colors start at 1.
		int total = 0;
		for(int j = 0; j < colors.dimension_0(); j++)
			if(colors(j) == color) total++;
		colors_map(color) = total;
	}
};

class mapScan{
	public:
	local_int_1d_type colors_map;

	mapScan(const local_int_1d_type& colors_map_):
		colors_map(colors_map_){}

// Parallel scan that finishes off setting up colors_map.
	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i, local_int_t & upd, bool final)const{
		upd += colors_map(i);
		if(final)
			colors_map(i) = upd;
	}
};

class fillColorsInd{
	public:
	local_int_1d_type colors_ind;
	local_int_1d_type colors_map;
	Coloring::array_type colors;

	fillColorsInd(local_int_1d_type& colors_ind_, local_int_1d_type& colors_map_,
		Coloring::array_type colors_):
		colors_ind(colors_ind_), colors_map(colors_map_), colors(colors_){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int & i)const{
		int color = i+1; // Colors start at 1 and i starts at 0.
		int start = colors_map(i);
		for(int j = 0; j < colors.dimension_0(); j++){
			if(colors(j) == color){
				colors_ind(start) = j;
				start++;
			}
		}
		assert(start == colors_map(i+1));//Make sure we only fill up exactly our color. Nothing more nothing less.
	}
};
#endif

/*
	The goal of this function will be to color the vertices in our matrix so we can use the
	level solve in SYGMS if so desired. Start by coloring A and then moving onto coloring
	A->Ac. This may get slow during the resolveConflicts phase since that one will run
	in serial and fix any errors that may have occured during greedyColor.

	Credit to Erik Boman since this code is mostly cut/paste from his color work.
*/

int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

// This function can be used to completely transform any part of the data structures.
// Right now it does nothing, so compiling with a check for unused variables results in complaints
#ifdef Option_0
	LevelScheduler(A);
	if(A.Ac != 0) return OptimizeProblem(*A.Ac, data, b, x, xexact);
#else
#ifdef Option_1
	Coloring::array_type colors("colors", A.localNumberOfRows);
	Coloring::array_type idx("idx", A.localMatrix.graph.row_map.dimension_0()); // Should be A.localNumberOfRows+1 length
	Coloring::array_type adj("adj", A.localMatrix.graph.entries.dimension_0()); // Should be A.LocalNumberOfNonzeros.
	Kokkos::parallel_for(A.localMatrix.graph.row_map.dimension_0(), fillIdx(idx, A.localMatrix.graph.row_map));
	Kokkos::parallel_for(A.localMatrix.graph.entries.dimension_0(), fillAdj(adj, A.localMatrix.graph.entries));

	Coloring c(A.localNumberOfRows, idx, adj, colors);
	c.color(false, false, true); // Flags are as follows... Use conflict List, Serial Resolve Conflict, Time and show.
	int numColors = c.getNumColors();
	std::cout<<"Number of colors used: " << numColors << std::endl;
	local_int_1d_type colors_map("Colors Map", numColors + 1);
	local_int_1d_type colors_ind("Colors Idx", A.localNumberOfRows);
// Fill colors_map so that colors_map(i) contains the number of entries with color i
	Kokkos::parallel_for(numColors, fillColorsMap(colors_map, colors));
// Scan colors_map to finish filling out the map.
	Kokkos::parallel_scan(numColors + 1, mapScan(colors_map));
// Use colors_map to fill fill out colors_ind.
	Kokkos::parallel_for(numColors, fillColorsInd(colors_ind, colors_map, colors));
// Prepare to fill out our _colors_order
	Coloring::array_type::HostMirror host_colors = Kokkos::create_mirror_view(colors);
	Kokkos::deep_copy(host_colors, colors);
//	for(int i = 0; i < colors.dimension_0(); i++) std::cout << "Row " << i << " got color: " << colors(i) << std::endl;
	bool marked[numColors];
	int colorsToFill = numColors;
	for(int i = 0; i < numColors; i++) marked[i] = false;
// Fill out f_colors_order.
	local_int_1d_type f_colors_order("f_colors_order", numColors);
	host_local_int_1d_type host_f_colors_order = Kokkos::create_mirror_view(f_colors_order);
	for(local_int_t i = 0; colorsToFill > 0; i++){ // This should be safe assuming that Coloring::getNumColors() works.
		int currentColor = host_colors(i);
		if(!marked[currentColor-1]){ // If our currentColor isnt being used, add it to f_colors_order. -1 since colors start at 1 and i starts at 0
			host_f_colors_order(numColors-colorsToFill) = currentColor;
			marked[currentColor-1] = true;
			colorsToFill--;
			std::cout<< "Used color: " << currentColor << " Colors Remaining: " << colorsToFill << std::endl;
		}
	}
	Kokkos::deep_copy(f_colors_order, host_f_colors_order);
// Reset our markers to fill out b_colors_order
	for(int i = 0; i < numColors; i++) marked[i] = false; // TODO Technically these should all be true so we could just flip if statement in b_colors_order.
	colorsToFill = numColors;
// Fill out b_colors_order
	local_int_1d_type b_colors_order("b_colors_order", numColors);
	host_local_int_1d_type host_b_colors_order = Kokkos::create_mirror_view(b_colors_order);
	for(int i = colors.dimension_0() - 1; colorsToFill > 0; i--){ // This should be safe assuming that Coloring::getNumColors() works.
		int currentColor = host_colors(i);
		if(!marked[currentColor-1]){
			host_b_colors_order(numColors-colorsToFill) = currentColor;
			marked[currentColor-1] = true;
			colorsToFill--;
		}
	}
std::cout<< "here" << std::endl;
	Kokkos::deep_copy(b_colors_order, host_b_colors_order);
	for(int i = 0; i < numColors; i++)
		std::cout<< "F(i): " << host_f_colors_order(i) << " B(i): " << host_b_colors_order(i) << std::endl;
// Assign everything back to A now.
	A.colors_map = colors_map;
	A.colors_ind = colors_ind;
	A.numColors = numColors;
	A.f_colors_order = f_colors_order;
	A.b_colors_order = b_colors_order;
// Make sure we color our coarse matrices as well.
	if(A.Ac != 0) return OptimizeProblem(*A.Ac, data, b, x, xexact);//TODO data, b, x, and xexact are never used but if that changes they may need to be changed.
	else return(0);
#else
#ifdef Option_4
	A.z = double_1d_type("z", A.localNumberOfRows);
	A.old = double_1d_type("old", A.localNumberOfRows);
	if(A.Ac != 0) return OptimizeProblem(*A.Ac, data, b, x, xexact);
	else return(0);
#endif // Option_4
#endif // Option_1
#endif // Option_0
	return(0);
}
