/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../include/dace_onnx.h"
#include "onnxruntime_c_api.h"
#include "cpu_provider_factory.h"
#include "cuda_provider_factory.h"


// Start global ORT setup
const OrtApi* __ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

// helper function to check for status
void __ort_check_status(OrtStatus* status)
{
    if (status != NULL) {
        const char* msg = __ort_api->GetErrorMessage(status);
        fprintf(stderr, "%s\n", msg);
        __ort_api->ReleaseStatus(status);
        exit(1);
    }
}
OrtEnv* __ort_env;
OrtKernelSession* __ort_session;
OrtSessionOptions* __ort_session_options;

OrtMemoryInfo* __ort_cpu_mem_info;
// End global ORT setup
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Concat_53_0_0_0;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Concat_53_0_0_0;
void __program_sub_sdfg_internal(long long * __restrict__ array_concat_result, long long * __restrict__ array_inputs__0, long long * __restrict__ array_inputs__1, long long * __restrict__ array_inputs__2)
{

    {
        
        
        {
            long long* inputs__2 = &array_inputs__2[0];
            long long* inputs__0 = &array_inputs__0[0];
            long long* inputs__1 = &array_inputs__1[0];
            long long* concat_result = array_concat_result;

            ///////////////////

            int64_t input_inputs__0_dims[1] = {1};

            OrtValue* ort_value_input_inputs__0;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (inputs__0)),
            1 * sizeof(long long),
            input_inputs__0_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &ort_value_input_inputs__0
            ));

            int64_t input_inputs__1_dims[1] = {1};

            OrtValue* ort_value_input_inputs__1;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (inputs__1)),
            1 * sizeof(long long),
            input_inputs__1_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &ort_value_input_inputs__1
            ));

            int64_t input_inputs__2_dims[1] = {1};

            OrtValue* ort_value_input_inputs__2;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (inputs__2)),
            1 * sizeof(long long),
            input_inputs__2_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &ort_value_input_inputs__2
            ));

            int64_t output_concat_result_dims[1] = {3};

            OrtValue* ort_value_output_concat_result;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (concat_result)),
            3 * sizeof(long long),
            output_concat_result_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &ort_value_output_concat_result
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Concat_53_0_0_0, 0, ort_value_input_inputs__0));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Concat_53_0_0_0, 1, ort_value_input_inputs__1));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Concat_53_0_0_0, 2, ort_value_input_inputs__2));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Concat_53_0_0_0, 0, ort_value_output_concat_result));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Concat_53_0_0_0));
            __ort_api->ReleaseValue(ort_value_input_inputs__0);
            __ort_api->ReleaseValue(ort_value_input_inputs__1);
            __ort_api->ReleaseValue(ort_value_input_inputs__2);
            __ort_api->ReleaseValue(ort_value_output_concat_result);
            ///////////////////

        }
    }
}

DACE_EXPORTED void __program_sub_sdfg(long long * __restrict__ array_concat_result, long long * __restrict__ array_inputs__0, long long * __restrict__ array_inputs__1, long long * __restrict__ array_inputs__2)
{
    __program_sub_sdfg_internal(array_concat_result, array_inputs__0, array_inputs__1, array_inputs__2);
}

DACE_EXPORTED int __dace_init_sub_sdfg(long long * __restrict__ array_concat_result, long long * __restrict__ array_inputs__0, long long * __restrict__ array_inputs__1, long long * __restrict__ array_inputs__2)
{
    int __result = 0;

    __ort_check_status(__ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &__ort_cpu_mem_info));
    __ort_check_status(__ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dace_graph", &__ort_env));
    __ort_check_status(__ort_api->CreateSessionOptions(&__ort_session_options));
    __ort_check_status(OrtSessionOptionsAppendExecutionProvider_CPU(__ort_session_options, /*use_arena=*/0));

    __ort_check_status(__ort_api->CreateKernelSession(__ort_session_options, &__ort_session, 12));

    {
        // Setup for ONNX_ONNX_Concat_53_0_0_0
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Concat_53_0_0_0", "Concat", &__ort_context_ONNX_ONNX_Concat_53_0_0_0));

        // Add parameter inputs__0
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Concat_53_0_0_0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        // Add parameter inputs__1
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Concat_53_0_0_0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        // Add parameter inputs__2
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Concat_53_0_0_0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        // Add parameter concat_result
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Concat_53_0_0_0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        // Setup attributes
        {
            // Setup attribute axis

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInt(__ort_context_ONNX_ONNX_Concat_53_0_0_0, "axis", 0));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Concat_53_0_0_0, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Concat_53_0_0_0));
    } // end setup for context_ONNX_ONNX_Concat_53_0_0_0

    return __result;
}

DACE_EXPORTED void __dace_exit_sub_sdfg(long long * __restrict__ array_concat_result, long long * __restrict__ array_inputs__0, long long * __restrict__ array_inputs__1, long long * __restrict__ array_inputs__2)
{
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Concat_53_0_0_0);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Concat_53_0_0_0);

    __ort_api->ReleaseMemoryInfo(__ort_cpu_mem_info);
    __ort_api->ReleaseKernelSession(__ort_session);
    __ort_api->ReleaseSessionOptions(__ort_session_options);
    __ort_api->ReleaseEnv(__ort_env);

}

