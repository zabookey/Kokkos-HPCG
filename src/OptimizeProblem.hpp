#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "KokkosSetup.hpp"
#include "Kokkos_Atomic.hpp"
#include <impl/Kokkos_Timer.hpp>

#ifdef Option_1 // Coloring Option.
#ifndef FORBIDDEN_SIZE
#define FORBIDDEN_SIZE 64 // GPU: Hopefully this fits in fast (shared) memory
#endif

enum conflict_type { CONFLICT_IMPLICIT, CONFLICT_LIST };

template <class Ordinal, class ExecSpace>
struct functorGreedyColor {
  typedef Kokkos::View<Ordinal *, ExecSpace> array_type; 
  typedef ExecSpace execution_space;

    functorGreedyColor(
                const array_type idx, 
                const array_type adj,
                array_type colors,
                array_type vertexList,
                Ordinal vertexListLength,
                Ordinal chunkSize
               ) : 
      _idx(idx), _adj(adj), _colors(colors), _vertexList(vertexList), _vertexListLength(vertexListLength), _chunkSize(chunkSize){
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Ordinal ii) const {
      // Color vertex i with smallest available color.
      //
      // Each thread colors a chunk of vertices to prevent all
      // vertices getting the same color.
      //
      // This version uses a bool array of size FORBIDDEN_SIZE. 
      // TODO: With chunks, the forbidden array should be char/int
      //       and reused for all vertices in the chunk.
      //
      Ordinal i = 0;
      for (Ordinal ichunk=0; ichunk<_chunkSize; ichunk++){
      if (ii*_chunkSize +ichunk < _vertexListLength)
        i = _vertexList[ii*_chunkSize +ichunk];
      else
        continue;

      if (_colors[i] > 0) continue; // Already colored this vertex

#ifdef DEBUG
#ifdef KOKKOS_HAVE_OPENMP
      int tid = omp_get_thread_num();
      printf("  Thread %i, color vertex %i\n", tid, i);
#else
      printf("  color vertex %i\n", i);
#endif
#endif

      bool foundColor = false; // Have we found a valid color?
      // Use forbidden array to find available color.
      // This array should be small enough to fit in fast memory (use Kokkos memoryspace?)
      bool forbidden[FORBIDDEN_SIZE]; // Forbidden colors 
      // Do multiple passes if array is too small.
      Ordinal degree = _idx[i+1]-_idx[i]; // My degree
      for (Ordinal offset = 0; (offset <= degree) && (!foundColor); offset += FORBIDDEN_SIZE){
        // initialize
        for (int j=0; j< FORBIDDEN_SIZE; j++)
          forbidden[j] = false;
        if (offset == 0)
          forbidden[0] = true; // by convention, start at 1
  
        // Check nbors, fill forbidden array.
        for (Ordinal j=_idx[i]; j<_idx[i+1]; j++){
          if (_adj[j] == i) continue; // Skip self-loops
          int c= _colors[_adj[j]];
          // Removed option to leave potentially conflicted vertices uncolored.
          //if (c== -1){ // Nbor is being colored at same time
          //  _colors[i] = 0; // Neutral color, skip and recolor later
          //  foundColor = true;
          //  return;
          //}
          if ((c>= offset) && (c-offset < FORBIDDEN_SIZE))
            forbidden[c-offset] = true;
        }
        // color vertex i with smallest available color (FirstFit)
        // TODO: Add options for other color choices (Random, LeastUsed)
        for (int c=0; c< FORBIDDEN_SIZE; c++){
          if (!forbidden[c]){
            _colors[i] = offset+c;
            //_colors[i] += (i&1); // RandX strategy to reduce conflicts
#ifdef DEBUG
            printf("Debug: Colors[%i] = %i\n", i, _colors[i]); 
#endif
            foundColor = true;
            break;
          }
        }
      }

#if 0
      if (!foundColor) {
        // Don't use forbidden array as no small color was found.
        // Use simple O(d^2) method to find a valid color.
        //std::cout << "No valid color yet for vertex " << i << ", switching to linear search." << std::endl;
        Ordinal degree = _idx[i+1]-_idx[i]; // My degree
        for (int c=FORBIDDEN_SIZE; c<= degree+1; c++){
          if (c==0) continue; // Skip color 0
          // Check if c is valid color.
          bool valid = true;
          for (int j=_idx[i]; (j<_idx[i+1]) && valid; j++){
            if (_adj[j] == i) continue; // Skip self-loops
            if (_colors[_adj[j]] == c){
              valid = false;
            }
          }
          if (valid){
            foundColor = true;
            _colors[i] = c;
            break;
          }
        }
      }
#endif

      //if (!foundColor) std::cerr << "ERROR: No valid color found for vertex " << i << ". This should never happen." << std::endl;
    }
    }

    array_type _idx;
    array_type _adj;
    array_type _colors;
    array_type _vertexList;
    Ordinal _vertexListLength;
    Ordinal _chunkSize;
};

template <class Ordinal, class ExecSpace>
struct functorCheckColoring {
  typedef struct dummy_struct_type { Ordinal numConflicts; Ordinal invalidColors; } value_type;
  typedef Kokkos::View<Ordinal *, ExecSpace> array_type; 
  typedef ExecSpace execution_space;

    functorCheckColoring(
                const array_type idx, 
                const array_type adj,
                const array_type colors
               ) : 
      _idx(idx), _adj(adj), _colors(colors) {
    }

    KOKKOS_INLINE_FUNCTION void
    init (value_type& status) const
    {
      status.numConflicts = 0;
      status.invalidColors = 0; 
    }

    KOKKOS_INLINE_FUNCTION void
    join (volatile value_type& dst ,
          const volatile value_type& src) const
    {
      // Join src and dst, return result in dst.
      dst.numConflicts += src.numConflicts;
      dst.invalidColors += src.invalidColors;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Ordinal i, value_type &status) const {
      // Reduction operator: Check valid colors and no conflicts.
      // The join() method will join results from all threads.
 
      // Check if vertex i has been colored.
      if (_colors[i] <= 0)
        (status.invalidColors)++; 

      // Check for conflicts at vertex i.
      for (Ordinal j=_idx[i]; j<_idx[i+1]; j++){
        if (_adj[j] == i) continue; // Skip self-loops
        if (_colors[_adj[j]] == _colors[i]) {
          // Note: every conflict will be counted twice.
          (status.numConflicts)++;
        }
      }
    }

    array_type _idx;
    array_type _adj;
    array_type _colors;
};

// Functor to be used in parallel_reduce, returns numUncolored vertices
template <class Ordinal, class ExecSpace>
struct functorFindConflicts {
  typedef ExecSpace execution_space;
  typedef Kokkos::View<Ordinal *, ExecSpace> array_type; 
  typedef Kokkos::View<Ordinal , ExecSpace> ordinal_type; 
  typedef Ordinal value_type; // for numUncolored vertices

    functorFindConflicts(
                const array_type idx, 
                const array_type adj,
                array_type colors,
                array_type vertexList,
                array_type recolorList,
                ordinal_type recolorListLength,
                const conflict_type conflictType
               ) : 
      _idx(idx), _adj(adj), _colors(colors), 
      _vertexList(vertexList), 
      _recolorList(recolorList), 
      _recolorListLength(recolorListLength),
      _conflictType(conflictType) 
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const Ordinal ii, value_type &numConflicts) const {
      Ordinal i = _vertexList[ii];
#ifdef DEBUG
#ifdef KOKKOS_HAVE_OPENMP
      int tid = omp_get_thread_num();
      printf("  Thread %i, findConflicts vertex %i\n", tid, i);
#else
      printf("  findConflicts vertex %i\n", i);
#endif
#endif

      // check vertex i conflicts
      for (Ordinal j=_idx[i]; j<_idx[i+1]; j++){
        if (_adj[j] == i) continue; // Skip self-loops
        if (_colors[_adj[j]] == _colors[i]) {
#ifdef DEBUG
          printf("Debug: found conflict for vertex %i, nbor= %i\n", i, _adj[j]) ; 
#endif
          // Only mark one of (i, adj[j]) as conflict
          if (i < _adj[j]){
            // Uncolor i and insert it in the conflict set
            // TODO: Pick lowest degree vertex instead.
            _colors[i] = 0; // Uncolor vertex i
            if (_conflictType == CONFLICT_LIST){
              // Atomically add vertex i to recolorList
              const Ordinal k = Kokkos::atomic_fetch_add( &_recolorListLength(), 1);
              _recolorList[k] = i;
#ifdef DEBUG
              printf("Adding vertex %i to conflict list; index= %i\n", i, k);
#endif
            }
            numConflicts++;
            break; // Once i is uncolored and marked conflict
          }
        }
      }
    }

    array_type _idx;
    array_type _adj;
    array_type _colors;
    array_type _vertexList;
    array_type _recolorList;
    ordinal_type _recolorListLength;
    conflict_type _conflictType;
};

    template<class Ordinal, class ExecSpace>
    struct functorInitList{
      typedef ExecSpace execution_space;
      typedef Kokkos::View<Ordinal *, ExecSpace> array_type;

      functorInitList (
        array_type vertexList 
      ) : 
        _vertexList(vertexList)
      {
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(const Ordinal i) const {
        // Natural order
        _vertexList[i] = i;
      }

      array_type _vertexList;
    };
#endif

int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact);

#endif //OPTIMIZEPROBLEM_HPP
