// $Id$
// Copyright Guido Kanschat, 1999

#include <lac/sparse_vanka.h>
#include <lac/full_matrix.h>
#include <lac/sparse_matrix.h>
#include <lac/vector.h>

#ifdef DEAL_II_USE_MT
#  include <base/thread_manager.h>
#  include <algorithm>
#endif

#include <map>



template<typename number>
SparseVanka<number>::SparseVanka(const SparseMatrix<number> &M,
				 const vector<bool>         &selected,
				 const bool                  conserve_mem,
				 const unsigned int          n_threads)
		:
		matrix (&M),
		conserve_mem (conserve_mem),
		selected (selected),
		n_threads (n_threads),
		inverses (M.m(), 0)
{
  Assert (M.m() == M.n(), ExcMatrixNotSquare ());
  Assert (M.m() == selected.size(), ExcInvalidVectorSize(M.m(), selected.size()));
  
  if (conserve_mem == false)
    compute_inverses ();
}


template<typename number>
SparseVanka<number>::~SparseVanka()
{
  vector<SmartPointer<FullMatrix<float> > >::iterator i;
  for(i=inverses.begin(); i!=inverses.end(); ++i)
    {
      FullMatrix<float> *p = *i;
      *i = 0;
      if (p != 0) delete p;
    }
}



template <typename number>
void
SparseVanka<number>::compute_inverses () 
{
#ifdef DEAL_II_USE_MT
  const unsigned int n_inverses = count (selected.begin(),
					 selected.end(),
					 true);

  const unsigned int n_inverses_per_thread = max(n_inverses / n_threads,
						 1U);
  
				   // set up start and end index for
				   // each of the threads. note that
				   // we have to work somewhat to get
				   // this appropriate, since the
				   // indices for which inverses have
				   // to be computed may not be evenly
				   // distributed in the vector. as an
				   // extreme example consider
				   // numbering of DoFs by component,
				   // then all indices for which we
				   // have to do work will be
				   // consecutive, with other
				   // consecutive regions where we do
				   // not have to do something
  typedef ThreadManager::Mem_Fun_Data2<SparseVanka<number>,unsigned int,unsigned int> MemFunData;
  vector<MemFunData> mem_fun_data (n_threads,
				   MemFunData(this,
					      0, 0,
					      &compute_inverses));

  unsigned int c       = 0;
  unsigned int thread  = 0;
  mem_fun_data[0].arg1 = 0;
  
  for (unsigned int i=0; (i<matrix->m()) && (thread+1<n_threads); ++i)
    {
      if (selected[i] == true)
	++c;
      if (c == n_inverses_per_thread)
	{
	  mem_fun_data[thread].arg2   = i;
	  mem_fun_data[thread+1].arg1 = i;
	  ++thread;

	  c = 0;
	};
    };
  mem_fun_data[n_threads-1].arg2 = matrix->m();

				   // Now spawn the threads
  ThreadManager thread_manager;
  for (unsigned int i=0; i<n_threads; ++i)
    thread_manager.spawn (&mem_fun_data[i], THR_SCOPE_SYSTEM | THR_DETACHED);

  thread_manager.wait ();
  
#else
  compute_inverses (0, matrix->m());
#endif
};



template <typename number>
void
SparseVanka<number>::compute_inverses (const unsigned int begin,
				       const unsigned int end) 
{
				   // set-up the map that will be used
				   // by the functions which we call
				   // below.
  map<unsigned int, unsigned int> local_index;

				   // traverse all rows of the matrix
				   // which are selected
  for (unsigned int row=begin; row<end; ++row)
    if (selected[row] == true)
      compute_inverse (row, local_index);
};



template <typename number>
void
SparseVanka<number>::compute_inverse (const unsigned int               row,
				      map<unsigned int, unsigned int> &local_index) 
{
				   // first define an alias to the sparsity
				   // pattern of the matrix, since this
				   // will be used quite often
  const SparsityPattern &structure
    = matrix->get_sparsity_pattern();

  const unsigned int row_length = structure.row_length(row);
  inverses[row] = new FullMatrix<float> (row_length,
					 row_length); 
				   // mapping between:
				   // 1 column number of all
				   //   entries in this row, and
				   // 2 the position within this
				   //   row (as stored in the
				   //   SparsityPattern object
				   //
				   // since we do not explicitely
				   // consider nonsysmmetric sparsity
				   // patterns, the first element
				   // of each entry simply denotes
				   // all degrees of freedom that
				   // couple with #row#.
  local_index.clear ();
  for (unsigned int i=0; i<row_length; ++i)
    local_index.insert(pair<unsigned int, unsigned int>
		       (structure.column_number(row, i), i));
  
				   // Build local matrix and rhs
  for (map<unsigned int, unsigned int>::const_iterator is=local_index.begin();
       is!=local_index.end(); ++is)
    {
				       // irow loops over all DoFs that
				       // couple with the present DoF
      const unsigned int irow = is->first;
				       // index of DoF irow in the matrix
				       // row corresponding to DoF #row#.
				       // runs between 0 and row_length
      const unsigned int i = is->second;
				       // number of DoFs coupling to
				       // irow (including irow itself)
      const unsigned int irow_length = structure.row_length(irow);
      
				       // for all the DoFs that irow
				       // couples with
      for (unsigned int j=0; j<irow_length; ++j)
	{
					   // col is the number of
					   // this dof
	  const unsigned int col = structure.column_number(irow, j);
					   // find out whether this DoF
					   // (that couples with #irow#,
					   // which itself couples with
					   // #row#) also couples with
					   // #row#.
	  const map<unsigned int, unsigned int>::const_iterator js
	    = local_index.find(col);
	  
	  if (js != local_index.end())
	    (*inverses[row])(i,js->second) = matrix->raw_entry(irow,j);
	};
    };
  
				   // Compute new values
  inverses[row]->gauss_jordan();
};




template<typename number>
template<typename number2>
void
SparseVanka<number>::operator ()(Vector<number2>       &dst,
				 const Vector<number2> &src) const
{
				   // first set output vector to zero
  dst.clear ();
				   // then pass on to the function
				   // that actually does the work
  apply_preconditioner (dst, src, 0, matrix->m());
};




template<typename number>
template<typename number2>
void
SparseVanka<number>::apply_preconditioner (Vector<number2>       &dst,
					   const Vector<number2> &src,
					   const unsigned int     begin,
					   const unsigned int     end) const
{
  Assert (begin < end, ExcInvalidRange(begin, end));
  Assert (dst.size() == src.size(),
	  ExcInvalidVectorSize(dst.size(), src.size()));
  Assert (dst.size() == matrix->m(),
	  ExcInvalidVectorSize(dst.size(), src.size()));
  
				   // first define an alias to the sparsity
				   // pattern of the matrix, since this
				   // will be used quite often
  const SparsityPattern &structure
    = matrix->get_sparsity_pattern();


				   // store whether we shall work on
				   // the whole matrix, or only on
				   // blocks. this variable is used to
				   // optimize access to vectors a
				   // little bit.
  const bool range_is_restricted = ((begin != 0) || (end != matrix->m()));
  
				   // space to be used for local
				   // systems. allocate as much memory
				   // as is the maximum. this
				   // eliminates the need to
				   // re-allocate memory inside the
				   // loop.
  FullMatrix<float> local_matrix (structure.max_entries_per_row(),
				  structure.max_entries_per_row());
  Vector<float> b (structure.max_entries_per_row());
  Vector<float> x (structure.max_entries_per_row());
  
  map<unsigned int, unsigned int> local_index;

				   // traverse all rows of the matrix
				   // which are selected
  for (unsigned int row=begin; row<end; ++row)
    if (selected[row] == true)
      {
	const unsigned int row_length = structure.row_length(row);
	
					 // if we don't store the
					 // inverse matrices, then alias
					 // the entry in the global
					 // vector to the local matrix
					 // to be used
	if (conserve_mem == true)
	  {
	    inverses[row] = &local_matrix;
	    inverses[row]->reinit (row_length, row_length);
	  };
	
	b.reinit (row_length);
	x.reinit (row_length);
					 // mapping between:
					 // 1 column number of all
					 //   entries in this row, and
					 // 2 the position within this
					 //   row (as stored in the
					 //   SparsityPattern object
					 //
					 // since we do not explicitely
					 // consider nonsysmmetric sparsity
					 // patterns, the first element
					 // of each entry simply denotes
					 // all degrees of freedom that
					 // couple with #row#.
	local_index.clear ();
	for (unsigned int i=0; i<row_length; ++i)
	  local_index.insert(pair<unsigned int, unsigned int>
			     (structure.column_number(row, i), i));
	
					 // Build local matrix and rhs
	for (map<unsigned int, unsigned int>::const_iterator is=local_index.begin();
	     is!=local_index.end(); ++is)
	  {
					     // irow loops over all DoFs that
					     // couple with the present DoF
	    const unsigned int irow = is->first;
					     // index of DoF irow in the matrix
					     // row corresponding to DoF #row#.
					     // runs between 0 and row_length
	    const unsigned int i = is->second;
					     // number of DoFs coupling to
					     // irow (including irow itself)
	    const unsigned int irow_length = structure.row_length(irow);
	    
					     // copy rhs
	    b(i) = src(irow);
	    
					     // for all the DoFs that irow
					     // couples with
	    for (unsigned int j=0; j<irow_length; ++j)
	      {
						 // col is the number of
						 // this dof
		const unsigned int col = structure.column_number(irow, j);
						 // find out whether this DoF
						 // (that couples with #irow#,
						 // which itself couples with
						 // #row#) also couples with
						 // #row#.
		const map<unsigned int, unsigned int>::const_iterator js
		  = local_index.find(col);
						 // if not, then still use
						 // this dof to modify the rhs
						 //
						 // note that if so, we already
						 // have copied the entry above
		if (js == local_index.end())
		  {
		    if (!range_is_restricted ||
			((begin <= col) && (col < end)))
		      b(i) -= matrix->raw_entry(irow,j) * dst(col);
		  }
		else
						     // if so, then build the
						     // matrix out of it
		  if (conserve_mem == true)
		    (*inverses[row])(i,js->second) = matrix->raw_entry(irow,j);
	      };
	  };
	
					 // Compute new values
	if (conserve_mem == true)
	  inverses[row]->gauss_jordan();

					 // apply preconditioner
	inverses[row]->vmult(x,b);
	
					 // Distribute new values
	for (map<unsigned int, unsigned int>::const_iterator is=local_index.begin();
	     is!=local_index.end(); ++is)
	  {
	    const unsigned int irow = is->first;
	    const unsigned int i = is->second;

	    if (!range_is_restricted ||
		((begin <= irow) && (irow < end)))
	      dst(irow) = x(i);
					       // do nothing if not in
					       // the range
	  };
	
					 // if we don't store the
					 // inverses, then unalias the
					 // local matrix
	if (conserve_mem == true)
	  inverses[row] = 0;
      };
};




template <typename number>
SparseBlockVanka<number>::SparseBlockVanka (const SparseMatrix<number> &M,
					    const vector<bool>         &selected,
					    const bool                  conserve_memory,
					    const unsigned int          n_threads,
					    const unsigned int          n_blocks)
		:
		SparseVanka<number> (M, selected, conserve_memory, n_threads),
                n_blocks (n_blocks)
{
  Assert (n_blocks > 0, ExcInternalError());

				   // precompute the splitting points
  intervals.resize (n_blocks);
  
  const unsigned int n_inverses = count (selected.begin(),
					 selected.end(),
					 true);
  
  const unsigned int n_inverses_per_block = max(n_inverses / n_blocks,
						1U);
  
				   // set up start and end index for
				   // each of the blocks. note that
				   // we have to work somewhat to get
				   // this appropriate, since the
				   // indices for which inverses have
				   // to be computed may not be evenly
				   // distributed in the vector. as an
				   // extreme example consider
				   // numbering of DoFs by component,
				   // then all indices for which we
				   // have to do work will be
				   // consecutive, with other
				   // consecutive regions where we do
				   // not have to do something
  unsigned int c       = 0;
  unsigned int block   = 0;
  intervals[0].first   = 0;
  
  for (unsigned int i=0; (i<M.m()) && (block+1<n_blocks); ++i)
    {
      if (selected[i] == true)
	++c;
      if (c == n_inverses_per_block)
	{
	  intervals[block].second  = i;
	  intervals[block+1].first = i;
	  ++block;
	  
	  c = 0;
	};
    };
  intervals[n_blocks-1].second = M.m();
};




template <typename number>
template <typename number2>
void SparseBlockVanka<number>::operator() (Vector<number2>       &dst,
					   const Vector<number2> &src) const
{
  dst.clear ();

				   // if no blocking is required, pass
				   // down to the underlying class
  if (n_blocks == 1)
    apply_preconditioner (dst, src, 0, dst.size());
  else
				     // otherwise: blocking requested
    {
#ifdef DEAL_II_USE_MT
      typedef ThreadManager::Mem_Fun_Data4
	<const SparseVanka<number>, Vector<number2>&,
	const Vector<number2> &, unsigned int, unsigned int> MemFunData;
      vector<MemFunData> mem_fun_data
	(n_blocks,
	 MemFunData (this,
		     dst, src, 0, 0,
		     &SparseVanka<number>::template apply_preconditioner<number2>));

      ThreadManager thread_manager;
      for (unsigned int block=0; block<n_blocks; ++block)
	{
	  mem_fun_data[block].arg3 = intervals[block].first;
	  mem_fun_data[block].arg4 = intervals[block].second;

	  thread_manager.spawn (&mem_fun_data[block],
				THR_SCOPE_SYSTEM | THR_DETACHED);
	};

      thread_manager.wait ();
#else
      for (unsigned int block=0; block<n_blocks; ++block)
	apply_preconditioner (dst, src,
			      intervals[block].first,
			      intervals[block].second);
#endif
    };
};
