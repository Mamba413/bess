%module cbess

%{
#include "bess_lm.h"
#include "List.h"
#define SWIG_FILE_WITH_INIT
%}

%include "numpy.i"

%init %{
import_array();
%}

void pywrap_bess_lm(double* IN_ARRAY2, int DIM1, int DIM2, double* IN_ARRAY1, int DIM1, int T0, int max_steps, double* IN_ARRAY1, int DIM1, double* IN_ARRAY1, int DIM1, double* OUTPUT, double* ARGOUT_ARRAY1, int DIM1, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, int* ARGOUT_ARRAY1, int DIM1, bool normal);

