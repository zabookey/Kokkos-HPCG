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
      // TODO: parallel_reduce? This produced strange results...
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

#ifdef Option_1
	Coloring::array_type colors("colors", A.localNumberOfRows);
	Coloring::array_type idx("idx", A.localMatrix.graph.row_map.dimension_0()); // Should be A.localNumberOfRows+1 length
	Coloring::array_type adj("adj", A.localMatrix.graph.entries.dimension_0()); // Should be A.LocalNumberOfNonzeros.
	Kokkos::parallel_for(A.localMatrix.graph.row_map.dimension_0(), fillIdx(idx, A.localMatrix.graph.row_map));
	Kokkos::parallel_for(A.localMatrix.graph.entries.dimension_0(), fillAdj(adj, A.localMatrix.graph.entries));

	Coloring c(A.localNumberOfRows, idx, adj, colors);
	c.color(false, false, false); // Flags are as follows... Use conflict List, Serial Resolve Conflict, Time and show.
	int numColors = c.getNumColors();
	local_int_1d_type colors_map("Colors Map", numColors + 1);
	local_int_1d_type colors_ind("Colors Idx", A.localNumberOfRows);
	Kokkos::parallel_for(numColors, fillColorsMap(colors_map, colors));
	Kokkos::parallel_scan(numColors + 1, mapScan(colors_map));
	Kokkos::parallel_for(numColors, fillColorsInd(colors_ind, colors_map, colors));
	A.colors_map = colors_map;
	A.colors_ind = colors_ind;
	A.numColors = numColors;
// Make sure we color our coarse matrices as well.
	if(A.Ac != 0) return OptimizeProblem(*A.Ac, data, b, x, xexact);//TODO data, b, x, and xexact are never used but if that changes they may need to be changed.
	else return(0);
#endif
	return(0);
}
