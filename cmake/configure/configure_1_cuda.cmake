## ---------------------------------------------------------------------
##
## Copyright (C) 2012 - 2015 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE at
## the top level of the deal.II distribution.
##
## ---------------------------------------------------------------------

#
# Configuration for cuda support:
#

MACRO(FEATURE_CUDA_FIND_EXTERNAL var)
  FIND_PACKAGE(CUDA)

  IF(CUDA_FOUND)
    SET(${var} TRUE)

  ENDIF()
ENDMACRO()

MACRO(FEATURE_CUDA_ERROR_MESSAGE)
  MESSAGE(FATAL_ERROR "\n"
    "Could not find any suitable cuda library!\n"
    # ${CUDA_ADDITIONAL_ERROR_STRING}
    # "\nPlease ensure that a cuda library is installed on your computer\n"
    # "and set CMAKE_CXX_COMPILER to the appropriate mpi wrappers:\n"
    # "    $ CXX=\".../mpicxx\" cmake <...>\n"
    # "    $ cmake -DCMAKE_CXX_COMPILER=\".../mpicxx\" <...>\n"
    # "Or with additional C and Fortran wrappers (recommended!):\n"
    # "    $ CC=\".../mpicc\" CXX=\".../mpicxx\" F90=\".../mpif90\" cmake <...>\n"
    # "    $ cmake -DCMAKE_C_COMPILER=\".../mpicc\"\\\n"
    # "            -DCMAKE_CXX_COMPILER=\".../mpicxx\"\\\n"
    # "            -DCMAKE_Fortran_COMPILER=\".../mpif90\"\\\n"
    # "            <...>\n"
    )
ENDMACRO()


CONFIGURE_FEATURE(CUDA)
