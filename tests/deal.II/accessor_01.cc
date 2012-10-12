//----------------------------------------------------------------------
//    $Id$
//
//    Copyright (C) 2012 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------------------------------------------------

// We used to forget to instantiate a few static const member variables.
// Verify that this is now fixed.


#include "../tests.h"
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <fstream>


template <class ACCESSOR>
LogStream&
operator << (LogStream& log, const TriaIterator<ACCESSOR>& i)
{
  log << ACCESSOR::dimension << ' '
      << ACCESSOR::structure_dimension << ' '
      << ACCESSOR::space_dimension << ' ';
  i.print(log);
  return log;
}


template <class DH>
void test_in_dim(const DH& d1, const DH& d2)
{
  typename DH::active_cell_iterator a = d1.begin_active();
  typename DH::cell_iterator l = d1.begin(d1.get_tria().n_levels()-1);

  deallog << "a " << a << std::endl
	  << "l " << l << std::endl;
}


template<int dim>
void init_tria (Triangulation<dim>& tr)
{
  GridGenerator::hyper_cube(tr);
  tr.refine_global(4-dim);
}


template<int dim>
void init_dofs (DoFHandler<dim>& dof,
		const Triangulation<dim>& tr,
		const FiniteElement<dim>& fe)
{
  dof.initialize(tr, fe);
}


int main ()
{
  initlog(__FILE__, true);
  deallog.depth_console(0);

  Triangulation<2> t2;
  init_tria (t2);

  FE_Q<2> q21(1);
  DoFHandler<2> d21;
  init_dofs(d21, t2, q21);

  FE_Q<2> q22(2);
  DoFHandler<2> d22;
  init_dofs(d22, t2, q22);

  test_in_dim(d21,d22);
}
