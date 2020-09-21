#include <stdlib.h>
#include "sub_sdfg.h"

int main(int argc, char** argv) {
  long long * __restrict__ array_concat_result = (long long*) calloc(3, sizeof(long long));
  long long * __restrict__ array_inputs__0 = (long long*) calloc(1, sizeof(long long));
  long long * __restrict__ array_inputs__1 = (long long*) calloc(1, sizeof(long long));
  long long * __restrict__ array_inputs__2 = (long long*) calloc(1, sizeof(long long));

  __dace_init_sub_sdfg(array_concat_result, array_inputs__0, array_inputs__1, array_inputs__2);
  __program_sub_sdfg(array_concat_result, array_inputs__0, array_inputs__1, array_inputs__2);
  __dace_exit_sub_sdfg(array_concat_result, array_inputs__0, array_inputs__1, array_inputs__2);

  free(array_concat_result);
  free(array_inputs__0);
  free(array_inputs__1);
  free(array_inputs__2);
  return 0;
}
