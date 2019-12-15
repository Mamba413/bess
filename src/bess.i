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
void pywrap_bess_lms(double* IN_ARRAY2, int DIM1, int DIM2, double* IN_ARRAY1, int DIM1, int* IN_ARRAY1, int DIM1, int max_steps, double* IN_ARRAY1, int DIM1, double* IN_ARRAY1, int DIM1, double* OUTPUT, double* ARGOUT_ARRAY1, int DIM1, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, bool warm_start = false, bool normal = true);
\\void pywrap_bess_lm_gs(double* IN_ARRAY2, int DIM1, int DIM2, double* IN_ARRAY1, int DIM1, int s_min, int s_max, int K_max, int max_steps, double epsilon, double* IN_ARRAY1, int DIM1, double* IN_ARRAY1, int DIM1, double* OUTPUT, double* ARGOUT_ARRAY1, int DIM1, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, double* OUTPUT, bool warm_start = false, bool normal = true);

