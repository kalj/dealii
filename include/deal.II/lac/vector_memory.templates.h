// ---------------------------------------------------------------------
//
// Copyright (C) 2007 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii_vector_memory_templates_h
#define dealii_vector_memory_templates_h


#include <deal.II/lac/vector_memory.h>

DEAL_II_NAMESPACE_OPEN


template <typename VectorType>
typename GrowingVectorMemory<VectorType>::Pool GrowingVectorMemory<VectorType>::pool;

template <typename VectorType>
Threads::Mutex GrowingVectorMemory<VectorType>::mutex;

template <typename VectorType>
inline
GrowingVectorMemory<VectorType>::Pool::Pool()
  :
  data(nullptr)
{}



template <typename VectorType>
inline
GrowingVectorMemory<VectorType>::Pool::~Pool()
{
  // Nothing to do if memory was unused.
  if (data == nullptr) return;

  // First, delete all remaining
  // vectors. Actually, there should
  // be none, if there is no memory
  // leak
  for (typename std::vector<entry_type>::iterator i=data->begin();
       i != data->end();
       ++i)
    {
      delete i->second;
    }
  delete data;
}


template <typename VectorType>
inline
void
GrowingVectorMemory<VectorType>::Pool::initialize(const size_type size)
{
  if (data == nullptr)
    {
      data = new std::vector<entry_type>(size);

      for (typename std::vector<entry_type>::iterator i= data->begin();
           i != data->end();
           ++i)
        {
          i->first = false;
          i->second = new VectorType;
        }
    }
}


template <typename VectorType>
inline
GrowingVectorMemory<VectorType>::GrowingVectorMemory (const size_type initial_size,
                                                      const bool log_statistics)

  :
  total_alloc(0),
  current_alloc(0),
  log_statistics(log_statistics)
{
  Threads::Mutex::ScopedLock lock(mutex);
  pool.initialize(initial_size);
}


template <typename VectorType>
inline
GrowingVectorMemory<VectorType>::~GrowingVectorMemory()
{
  AssertNothrow(current_alloc == 0,
                StandardExceptions::ExcMemoryLeak(current_alloc));
  if (log_statistics)
    {
      deallog << "GrowingVectorMemory:Overall allocated vectors: "
              << total_alloc << std::endl;
      deallog << "GrowingVectorMemory:Maximum allocated vectors: "
              << pool.data->size() << std::endl;
    }
}



template <typename VectorType>
inline
VectorType *
GrowingVectorMemory<VectorType>::alloc ()
{
  Threads::Mutex::ScopedLock lock(mutex);
  ++total_alloc;
  ++current_alloc;
  // see if there is a free vector
  // available in our list
  for (typename std::vector<entry_type>::iterator i=pool.data->begin();
       i != pool.data->end(); ++i)
    {
      if (i->first == false)
        {
          i->first = true;
          return (i->second);
        }
    }

  // no free vector found, so let's
  // just allocate a new one
  const entry_type t (true, new VectorType);
  pool.data->push_back(t);

  return t.second;
}



template <typename VectorType>
inline
void
GrowingVectorMemory<VectorType>::free(const VectorType *const v)
{
  Threads::Mutex::ScopedLock lock(mutex);
  for (typename std::vector<entry_type>::iterator i=pool.data->begin();
       i != pool.data->end(); ++i)
    {
      if (v == (i->second))
        {
          i->first = false;
          --current_alloc;
          return;
        }
    }
  Assert(false, typename VectorMemory<VectorType>::ExcNotAllocatedHere());
}



template <typename VectorType>
inline
void
GrowingVectorMemory<VectorType>::release_unused_memory ()
{
  Threads::Mutex::ScopedLock lock(mutex);

  std::vector<entry_type> new_data;

  if (pool.data != nullptr)
    {
      const typename std::vector<entry_type>::const_iterator
      end = pool.data->end();
      for (typename std::vector<entry_type>::const_iterator
           i = pool.data->begin(); i != end ; ++i)
        if (i->first == false)
          delete i->second;
        else
          new_data.push_back (*i);

      *pool.data = new_data;
    }
}



template <typename VectorType>
inline
std::size_t
GrowingVectorMemory<VectorType>::memory_consumption () const
{
  Threads::Mutex::ScopedLock lock(mutex);

  std::size_t result = sizeof (*this);
  const typename std::vector<entry_type>::const_iterator
  end = pool.data->end();
  for (typename std::vector<entry_type>::const_iterator
       i = pool.data->begin(); i != end ; ++i)
    result += sizeof (*i) + i->second->memory_consumption();

  return result;
}


DEAL_II_NAMESPACE_CLOSE

#endif
