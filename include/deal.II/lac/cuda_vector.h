// ---------------------------------------------------------------------
//
// Copyright (C) 2016 by the deal.II authors
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

#ifndef dealii__cuda_vector_h
#define dealii__cuda_vector_h

#include <deal.II/base/config.h>
#include <deal.II/lac/vector_space_vector.h>

// Using DEAL_II_NAMESPACE_OPEN creates a strange error:
// "This declaration has no storage class or type specifier"
namespace dealii
{

  class CommunicationPatternBase;
  class IndexSet;
  template <typename Number> class ReadWriteVector;

  namespace LinearAlgebra
  {
    namespace CUDAWrappers
    {
      /**
       * This class implement vector using CUDA for use on Nvidia GPUs. This
       * class is derived from the LinearAlgebra::VectorSpaceVector class.
       *
       * @ingroup CUDAWrappers
       * @ingroup Vectors
       * @author Karl Ljungkvist, Bruno Turcksin, 2016
       */
      template <typename Number>
      class Vector: public VectorSpaceVector<Number>
      {
      public:
        typedef typename VectorSpaceVector<Number>::size_type size_type;
        typedef typename VectorSpaceVector<Number>::real_type real_type;

        /**
         * Constructor. Create a vector of dimension zero.
         */
        Vector();

        /**
         * Copy constructor.
         */
        Vector(const Vector<Number> &V);

        /**
         * Constructor. Set dimension to @p n and initialize all elements with
         * zero.
         *
         * The constructor is made explicit to avoid accident like this:
         * <tt>v=0;</tt>. Presumably, the user wants to set every elements of
         * the vector to zero, but instead, what happens is this call:
         * <tt>v=Vector@<Number@>(0);</tt>, i.e. the vector is replaced by one
         * of length zero.
         */
        explicit Vector(const size_type n);

        /**
         * Destructor.
         */
        ~Vector();

        /**
         * Reinit functionality. The flag <tt>omit_zeroing_entries</tt>
         * determines wheter the vector should be filled with zero (false) or
         * left untouched (true).
         */
        void reinit(const size_type n,
                    const bool      omit_zeroing_entries = false);

        /**
         * Import all the element from the input vector @p V.
         * VectorOperation::values @p operation is used to decide if the
         * elements int @p V should be added to the current vector or replace
         * the current elements. The last parameter is not used. It is only used
         * for distributed vectors. This is the function that should be used to
         * copy a vector to the GPU.
         */
        virtual void import(const ReadWriteVector<Number> &V,
                            VectorOperation::values operation,
                            std_cxx11::shared_ptr<const CommunicationPatternBase>);

        /**
         * Multiply the entive vector by a fixed factor.
         */
        virtual Vector<Number> &operator*= (const Number factor);

        /**
         * Divide the entire vector by a fixed factor.
         */
        virtual Vector<Number> &operator/= (const Number factor);

        /**
         * Add the vector @p V to the present one.
         */
        virtual Vector<Number> &operator+= (const VectorSpaceVector<Number> &V);

        /**
         * Subtract the vector @p V from the present one.
         */
        virtual Vector<Number> &operator-= (const VectorSpaceVector<Number> &V);

        /**
         * Return the scalar product of two vectors.
         */
        virtual Number operator* (const VectorSpaceVector<Number> &V);

        /**
         * Add @p to all components. Not that @p a is a scalar not a vector.
         */
        virtual void add(const Number a);

        /**
         * Simple addition of a multiple of a vector, i.e. <tt>*thos += a*V</tt>.
         */
        virtual void add(const Number a, const VectorSpaceVector<Number> &V);

        /**
         * Multiple addition of scaled vectors, i.e. <tt>*this += a*V</tt>.
         */
        virtual void add(const Number a, const VectorSpaceVector<Number> &V,
                         const Number b, const VectorSpaceVector<Number> &W);

        /**
         * Scaling and simple addition of a multiple of a vector, i.e. <tt>*this
         * = s*(*this)+a*V</tt>
         */
        virtual void sadd(const Number s, const Number a,
                          const VectorSpaceVector<Number> &V);

        /**
         * Scale each element of this vector by the corresponding element in the
         * argument. This function is mostly meant to simulation multiplication
         * (and immediate re-assignment) by a diagonal scaling matrix.
         */
        virtual void scale(const VectorSpaceVector<Number> &scaling_factors);

        /**
         * Assignement <tt>*this = a*V</tt>.
         */
        virtual void equ(const Number a, const VectorSpaceVector<Number> &V);

        /**
         * Return the l<sub>1</sub> norm of the vector (i.e., the sum of the
         * absolute values of all entries among all processors).
         */
        virtual real_type l1_norm() const;

        /**
         * Return the l<sub>2</sub> norm of the vector (i.e., the square root of
         * the sum of the square of all entries among all processors).
         */
        virtual real_type l2_norm() const;

        /**
         * Return the maximum norm of the vector (i.e., the maximum absolute
         * value among all entries and among all processors).
         */
        virtual real_type linfty_norm() const;

        /**
         * Perform a combined operation of a vector addition and a subsequent
         * inner product, returning the value of the inner product. In other
         * words, the result of this function is the same as if the user called
         * @code
         * this->add(a, V);
         * return_value = *this * W;
         * @endcode
         *
         * The reason this function exists is that this operation involves less
         * memory transfer than calling the two functions separately. This
         * method only needs to load three vectors, @p this, @p V, @p W, whereas
         * calling separate methods means to load the calling vector @p this
         * twice. Since most vector operations are memory transfer limited, this
         * reduces the time by 25\% (or 50\% if @p W equals @p this).
         */
        virtual Number add_and_dot(const Number a,
                                   const VectorSpaceVector<Number> &V,
                                   const VectorSpaceVector<Number> &W);

        /**
         * Return the size of the vector.
         */
        virtual size_type size();

        /**
         * Return an index set that describe which elements of this vector are
         * owned by the current processor, i.e. [0, size).
         */
        virtual dealii::IndexSet locally_owned_elements();

        /**
         * Print the vector to the output stream @p out.
         */
        virtual void print(std::ostream &out,
                           const unsigned int precision=2,
                           const bool scientific=true,
                           const bool across=true) const;

        /**
         * Return the memory consumption of this class in bytes.
         */
        virtual std::size_t memory_consumption() const;

      private:
        /**
         * Pointer to the array of elements of this vector.
         */
        Number *val;

        /**
         * Number of elements in the vector.
         */
        size_type n_elements;
      };
    }
  }

}
//DEAL_II_NAME_CLOSE

#endif
