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
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_MatMul_0_0_0_0;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_MatMul_0_0_0_0;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_1_0_0_3;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_1_0_0_3;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_MatMul_2_0_0_5;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_MatMul_2_0_0_5;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_3_0_0_7;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_3_0_0_7;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_MatMul_4_0_0_9;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_MatMul_4_0_0_9;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_5_0_0_11;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_5_0_0_11;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Reshape_15_0_0_13;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Reshape_15_0_0_13;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Transpose_16_0_0_14;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Transpose_16_0_0_14;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Reshape_26_0_0_15;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Reshape_26_0_0_15;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Reshape_36_0_0_16;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Reshape_36_0_0_16;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Transpose_37_0_0_17;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Transpose_37_0_0_17;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Transpose_38_0_0_18;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Transpose_38_0_0_18;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_MatMul_39_0_0_19;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_MatMul_39_0_0_19;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Div_41_0_0_20;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Div_41_0_0_20;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Softmax_42_0_0_22;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Softmax_42_0_0_22;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_MatMul_43_0_0_23;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_MatMul_43_0_0_23;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Transpose_44_0_0_24;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Transpose_44_0_0_24;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Reshape_54_0_0_25;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Reshape_54_0_0_25;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_MatMul_55_0_0_26;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_MatMul_55_0_0_26;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_56_0_0_28;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_56_0_0_28;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_57_0_0_30;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_57_0_0_30;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_ReduceMean_58_0_0_31;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_ReduceMean_58_0_0_31;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Sub_59_0_0_32;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Sub_59_0_0_32;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Cast_60_0_0_33;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Cast_60_0_0_33;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Pow_61_0_0_34;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Pow_61_0_0_34;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_ReduceMean_62_0_0_36;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_ReduceMean_62_0_0_36;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_64_0_0_37;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_64_0_0_37;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Sqrt_65_0_0_39;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Sqrt_65_0_0_39;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Div_66_0_0_40;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Div_66_0_0_40;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Mul_67_0_0_41;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Mul_67_0_0_41;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_68_0_0_43;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_68_0_0_43;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_MatMul_69_0_0_45;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_MatMul_69_0_0_45;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_70_0_0_47;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_70_0_0_47;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Div_72_0_0_49;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Div_72_0_0_49;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Erf_73_0_0_51;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Erf_73_0_0_51;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_75_0_0_52;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_75_0_0_52;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Mul_76_0_0_54;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Mul_76_0_0_54;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Mul_78_0_0_55;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Mul_78_0_0_55;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_MatMul_79_0_0_57;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_MatMul_79_0_0_57;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_80_0_0_59;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_80_0_0_59;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_81_0_0_61;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_81_0_0_61;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_ReduceMean_82_0_0_62;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_ReduceMean_82_0_0_62;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Sub_83_0_0_63;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Sub_83_0_0_63;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Cast_84_0_0_64;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Cast_84_0_0_64;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Pow_85_0_0_65;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Pow_85_0_0_65;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_ReduceMean_86_0_0_67;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_ReduceMean_86_0_0_67;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_88_0_0_68;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_88_0_0_68;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Sqrt_89_0_0_70;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Sqrt_89_0_0_70;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Div_90_0_0_71;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Div_90_0_0_71;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Mul_91_0_0_72;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Mul_91_0_0_72;
OrtExecutableKernel *__ort_kernel_ONNX_ONNX_Add_92_0_0_74;
OrtExecutableKernelContext *__ort_context_ONNX_ONNX_Add_92_0_0_74;
void __program_dace_model_internal(float * __restrict__ ONNX_133, float * __restrict__ ONNX_134, float * __restrict__ ONNX_135, float * __restrict__ ONNX_136, long long * __restrict__ ONNX_137, long long * __restrict__ ONNX_138, long long * __restrict__ ONNX_139, long long * __restrict__ ONNX_140, long long * __restrict__ ONNX_141, long long * __restrict__ ONNX_142, long long * __restrict__ ONNX_143, float * __restrict__ ONNX_144, float * __restrict__ ONNX_146, float * __restrict__ ONNX_147, long long * __restrict__ ONNX___tmp0, long long * __restrict__ ONNX___tmp1, long long * __restrict__ ONNX___tmp16, long long * __restrict__ ONNX___tmp17, long long * __restrict__ ONNX___tmp18, long long * __restrict__ ONNX___tmp19, long long * __restrict__ ONNX___tmp2, long long * __restrict__ ONNX___tmp20, long long * __restrict__ ONNX___tmp21, long long * __restrict__ ONNX___tmp22, long long * __restrict__ ONNX___tmp23, long long * __restrict__ ONNX___tmp24, long long * __restrict__ ONNX___tmp25, long long * __restrict__ ONNX___tmp26, long long * __restrict__ ONNX___tmp27, long long * __restrict__ ONNX___tmp3, long long * __restrict__ ONNX___tmp4, long long * __restrict__ ONNX___tmp5, long long * __restrict__ ONNX___tmp6, long long * __restrict__ ONNX___tmp7, float * __restrict__ ONNX_attentionDOToutputDOTLayerNormDOTbias, float * __restrict__ ONNX_attentionDOToutputDOTLayerNormDOTweight, float * __restrict__ ONNX_attentionDOToutputDOTdenseDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTkeyDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTqueryDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTvalueDOTbias, float * __restrict__ ONNX_inputDOT1, float * __restrict__ ONNX_intermediateDOTdenseDOTbias, float * __restrict__ ONNX_outputDOTLayerNormDOTbias, float * __restrict__ ONNX_outputDOTLayerNormDOTweight, float * __restrict__ ONNX_outputDOTdenseDOTbias, float ONNX_100, float ONNX_109, float ONNX_112, float ONNX_115, float ONNX_128, float ONNX_145, float ONNX_148, long long ONNX_27, long long ONNX_30, long long ONNX_42, long long ONNX_45, long long ONNX_56, long long ONNX_59, float ONNX_72, long long ONNX_78, long long ONNX_81, long long ONNX___tmp10, long long ONNX___tmp11, long long ONNX___tmp12, long long ONNX___tmp13, long long ONNX___tmp14, long long ONNX___tmp15, long long ONNX___tmp8, long long ONNX___tmp9)
{

    {
        float *ONNX_18 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_19 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_21 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_22 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_24 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_25 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_39 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_40 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_54 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_68 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_69 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_70 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_71 = new float DACE_ALIGN(64)[25165824];
        float *ONNX_73 = new float DACE_ALIGN(64)[25165824];
        float *ONNX_74 = new float DACE_ALIGN(64)[25165824];
        float *ONNX_75 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_76 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_88 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_90 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_91 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_92 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_94 = new float DACE_ALIGN(64)[4096];
        float *ONNX_95 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_96 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_98 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_99 = new float DACE_ALIGN(64)[4096];
        float *ONNX_101 = new float DACE_ALIGN(64)[4096];
        float *ONNX_102 = new float DACE_ALIGN(64)[4096];
        float *ONNX_103 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_104 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_105 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_107 = new float DACE_ALIGN(64)[12582912];
        float *ONNX_108 = new float DACE_ALIGN(64)[12582912];
        float *ONNX_110 = new float DACE_ALIGN(64)[12582912];
        float *ONNX_111 = new float DACE_ALIGN(64)[12582912];
        float *ONNX_113 = new float DACE_ALIGN(64)[12582912];
        float *ONNX_114 = new float DACE_ALIGN(64)[12582912];
        float *ONNX_116 = new float DACE_ALIGN(64)[12582912];
        float *ONNX_118 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_119 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_120 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_122 = new float DACE_ALIGN(64)[4096];
        float *ONNX_123 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_124 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_126 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_127 = new float DACE_ALIGN(64)[4096];
        float *ONNX_129 = new float DACE_ALIGN(64)[4096];
        float *ONNX_130 = new float DACE_ALIGN(64)[4096];
        float *ONNX_131 = new float DACE_ALIGN(64)[3145728];
        float *ONNX_132 = new float DACE_ALIGN(64)[3145728];
        
        
        {
            float* A = &ONNX_inputDOT1[0];
            float* B = &ONNX_134[0];
            float* Y = ONNX_18;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[2] = {768, 768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            589824 * sizeof(float),
            input_B_dims,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_Y_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            3145728 * sizeof(float),
            output_Y_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_0_0_0_0, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_0_0_0_0, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_MatMul_0_0_0_0, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_MatMul_0_0_0_0));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_18[0];
            float* B = &ONNX_attentionDOTselfDOTqueryDOTbias[0];
            float* C = ONNX_19;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_1_0_0_3, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_1_0_0_3, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_1_0_0_3, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_1_0_0_3));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_inputDOT1[0];
            float* B = &ONNX_135[0];
            float* Y = ONNX_21;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[2] = {768, 768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            589824 * sizeof(float),
            input_B_dims,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_Y_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            3145728 * sizeof(float),
            output_Y_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_2_0_0_5, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_2_0_0_5, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_MatMul_2_0_0_5, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_MatMul_2_0_0_5));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_21[0];
            float* B = &ONNX_attentionDOTselfDOTkeyDOTbias[0];
            float* C = ONNX_22;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_3_0_0_7, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_3_0_0_7, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_3_0_0_7, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_3_0_0_7));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_inputDOT1[0];
            float* B = &ONNX_136[0];
            float* Y = ONNX_24;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[2] = {768, 768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            589824 * sizeof(float),
            input_B_dims,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_Y_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            3145728 * sizeof(float),
            output_Y_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_4_0_0_9, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_4_0_0_9, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_MatMul_4_0_0_9, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_MatMul_4_0_0_9));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_24[0];
            float* B = &ONNX_attentionDOTselfDOTvalueDOTbias[0];
            float* C = ONNX_25;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_5_0_0_11, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_5_0_0_11, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_5_0_0_11, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_5_0_0_11));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* data = &ONNX_19[0];
            long long* shape = &ONNX___tmp24[0];
            float* reshaped = ONNX_39;

            ///////////////////

            int64_t input_data_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t input_shape_dims[1] = {4};

            OrtValue* ort_value_input_shape;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (shape)),
            4 * sizeof(long long),
            input_shape_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &ort_value_input_shape
            ));

            int64_t output_reshaped_dims[4] = {8, 512, 12, 64};

            OrtValue* ort_value_output_reshaped;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (reshaped)),
            3145728 * sizeof(float),
            output_reshaped_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_reshaped
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Reshape_15_0_0_13, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Reshape_15_0_0_13, 1, ort_value_input_shape));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Reshape_15_0_0_13, 0, ort_value_output_reshaped));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Reshape_15_0_0_13));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_input_shape);
            __ort_api->ReleaseValue(ort_value_output_reshaped);
            ///////////////////

        }
        {
            float* data = &ONNX_39[0];
            float* transposed = ONNX_40;

            ///////////////////

            int64_t input_data_dims[4] = {8, 512, 12, 64};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t output_transposed_dims[4] = {8, 12, 512, 64};

            OrtValue* ort_value_output_transposed;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (transposed)),
            3145728 * sizeof(float),
            output_transposed_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_transposed
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Transpose_16_0_0_14, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Transpose_16_0_0_14, 0, ort_value_output_transposed));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Transpose_16_0_0_14));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_output_transposed);
            ///////////////////

        }
        {
            float* data = &ONNX_22[0];
            long long* shape = &ONNX___tmp25[0];
            float* reshaped = ONNX_54;

            ///////////////////

            int64_t input_data_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t input_shape_dims[1] = {4};

            OrtValue* ort_value_input_shape;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (shape)),
            4 * sizeof(long long),
            input_shape_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &ort_value_input_shape
            ));

            int64_t output_reshaped_dims[4] = {8, 512, 12, 64};

            OrtValue* ort_value_output_reshaped;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (reshaped)),
            3145728 * sizeof(float),
            output_reshaped_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_reshaped
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Reshape_26_0_0_15, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Reshape_26_0_0_15, 1, ort_value_input_shape));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Reshape_26_0_0_15, 0, ort_value_output_reshaped));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Reshape_26_0_0_15));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_input_shape);
            __ort_api->ReleaseValue(ort_value_output_reshaped);
            ///////////////////

        }
        {
            float* data = &ONNX_54[0];
            float* transposed = ONNX_70;

            ///////////////////

            int64_t input_data_dims[4] = {8, 512, 12, 64};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t output_transposed_dims[4] = {8, 12, 64, 512};

            OrtValue* ort_value_output_transposed;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (transposed)),
            3145728 * sizeof(float),
            output_transposed_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_transposed
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Transpose_38_0_0_18, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Transpose_38_0_0_18, 0, ort_value_output_transposed));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Transpose_38_0_0_18));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_output_transposed);
            ///////////////////

        }
        {
            float* A = &ONNX_40[0];
            float* B = &ONNX_70[0];
            float* Y = ONNX_71;

            ///////////////////

            int64_t input_A_dims[4] = {8, 12, 512, 64};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[4] = {8, 12, 64, 512};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            3145728 * sizeof(float),
            input_B_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_Y_dims[4] = {8, 12, 512, 512};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            25165824 * sizeof(float),
            output_Y_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_39_0_0_19, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_39_0_0_19, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_MatMul_39_0_0_19, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_MatMul_39_0_0_19));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_71[0];
            float B = ONNX_72;
            float* C = ONNX_73;

            ///////////////////

            int64_t input_A_dims[4] = {8, 12, 512, 512};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            25165824 * sizeof(float),
            input_A_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            &B,
            1 * sizeof(float),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[4] = {8, 12, 512, 512};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            25165824 * sizeof(float),
            output_C_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Div_41_0_0_20, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Div_41_0_0_20, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Div_41_0_0_20, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Div_41_0_0_20));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* input = &ONNX_73[0];
            float* output = ONNX_74;

            ///////////////////

            int64_t input_input_dims[4] = {8, 12, 512, 512};

            OrtValue* ort_value_input_input;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (input)),
            25165824 * sizeof(float),
            input_input_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_input
            ));

            int64_t output_output_dims[4] = {8, 12, 512, 512};

            OrtValue* ort_value_output_output;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (output)),
            25165824 * sizeof(float),
            output_output_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_output
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Softmax_42_0_0_22, 0, ort_value_input_input));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Softmax_42_0_0_22, 0, ort_value_output_output));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Softmax_42_0_0_22));
            __ort_api->ReleaseValue(ort_value_input_input);
            __ort_api->ReleaseValue(ort_value_output_output);
            ///////////////////

        }
        {
            float* data = &ONNX_25[0];
            long long* shape = &ONNX___tmp26[0];
            float* reshaped = ONNX_68;

            ///////////////////

            int64_t input_data_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t input_shape_dims[1] = {4};

            OrtValue* ort_value_input_shape;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (shape)),
            4 * sizeof(long long),
            input_shape_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &ort_value_input_shape
            ));

            int64_t output_reshaped_dims[4] = {8, 512, 12, 64};

            OrtValue* ort_value_output_reshaped;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (reshaped)),
            3145728 * sizeof(float),
            output_reshaped_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_reshaped
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Reshape_36_0_0_16, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Reshape_36_0_0_16, 1, ort_value_input_shape));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Reshape_36_0_0_16, 0, ort_value_output_reshaped));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Reshape_36_0_0_16));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_input_shape);
            __ort_api->ReleaseValue(ort_value_output_reshaped);
            ///////////////////

        }
        {
            float* data = &ONNX_68[0];
            float* transposed = ONNX_69;

            ///////////////////

            int64_t input_data_dims[4] = {8, 512, 12, 64};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t output_transposed_dims[4] = {8, 12, 512, 64};

            OrtValue* ort_value_output_transposed;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (transposed)),
            3145728 * sizeof(float),
            output_transposed_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_transposed
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Transpose_37_0_0_17, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Transpose_37_0_0_17, 0, ort_value_output_transposed));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Transpose_37_0_0_17));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_output_transposed);
            ///////////////////

        }
        {
            float* A = &ONNX_74[0];
            float* B = &ONNX_69[0];
            float* Y = ONNX_75;

            ///////////////////

            int64_t input_A_dims[4] = {8, 12, 512, 512};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            25165824 * sizeof(float),
            input_A_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[4] = {8, 12, 512, 64};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            3145728 * sizeof(float),
            input_B_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_Y_dims[4] = {8, 12, 512, 64};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            3145728 * sizeof(float),
            output_Y_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_43_0_0_23, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_43_0_0_23, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_MatMul_43_0_0_23, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_MatMul_43_0_0_23));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* data = &ONNX_75[0];
            float* transposed = ONNX_76;

            ///////////////////

            int64_t input_data_dims[4] = {8, 12, 512, 64};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t output_transposed_dims[4] = {8, 512, 12, 64};

            OrtValue* ort_value_output_transposed;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (transposed)),
            3145728 * sizeof(float),
            output_transposed_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_transposed
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Transpose_44_0_0_24, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Transpose_44_0_0_24, 0, ort_value_output_transposed));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Transpose_44_0_0_24));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_output_transposed);
            ///////////////////

        }
        {
            float* data = &ONNX_76[0];
            long long* shape = &ONNX___tmp27[0];
            float* reshaped = ONNX_88;

            ///////////////////

            int64_t input_data_dims[4] = {8, 512, 12, 64};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t input_shape_dims[1] = {3};

            OrtValue* ort_value_input_shape;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (shape)),
            3 * sizeof(long long),
            input_shape_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &ort_value_input_shape
            ));

            int64_t output_reshaped_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_reshaped;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (reshaped)),
            3145728 * sizeof(float),
            output_reshaped_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_reshaped
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Reshape_54_0_0_25, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Reshape_54_0_0_25, 1, ort_value_input_shape));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Reshape_54_0_0_25, 0, ort_value_output_reshaped));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Reshape_54_0_0_25));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_input_shape);
            __ort_api->ReleaseValue(ort_value_output_reshaped);
            ///////////////////

        }
        {
            float* A = &ONNX_88[0];
            float* B = &ONNX_144[0];
            float* Y = ONNX_90;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[2] = {768, 768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            589824 * sizeof(float),
            input_B_dims,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_Y_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            3145728 * sizeof(float),
            output_Y_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_55_0_0_26, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_55_0_0_26, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_MatMul_55_0_0_26, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_MatMul_55_0_0_26));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_90[0];
            float* B = &ONNX_attentionDOToutputDOTdenseDOTbias[0];
            float* C = ONNX_91;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_56_0_0_28, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_56_0_0_28, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_56_0_0_28, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_56_0_0_28));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_91[0];
            float* B = &ONNX_inputDOT1[0];
            float* C = ONNX_92;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            3145728 * sizeof(float),
            input_B_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_57_0_0_30, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_57_0_0_30, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_57_0_0_30, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_57_0_0_30));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* data = &ONNX_92[0];
            float* reduced = ONNX_94;

            ///////////////////

            int64_t input_data_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t output_reduced_dims[3] = {8, 512, 1};

            OrtValue* ort_value_output_reduced;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (reduced)),
            4096 * sizeof(float),
            output_reduced_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_reduced
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_ReduceMean_58_0_0_31, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_ReduceMean_58_0_0_31, 0, ort_value_output_reduced));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_ReduceMean_58_0_0_31));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_output_reduced);
            ///////////////////

        }
        {
            float* A = &ONNX_92[0];
            float* B = &ONNX_94[0];
            float* C = ONNX_95;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[3] = {8, 512, 1};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            4096 * sizeof(float),
            input_B_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Sub_59_0_0_32, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Sub_59_0_0_32, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Sub_59_0_0_32, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Sub_59_0_0_32));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* input = &ONNX_95[0];
            float* output = ONNX_96;

            ///////////////////

            int64_t input_input_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_input;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (input)),
            3145728 * sizeof(float),
            input_input_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_input
            ));

            int64_t output_output_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_output;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (output)),
            3145728 * sizeof(float),
            output_output_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_output
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Cast_60_0_0_33, 0, ort_value_input_input));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Cast_60_0_0_33, 0, ort_value_output_output));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Cast_60_0_0_33));
            __ort_api->ReleaseValue(ort_value_input_input);
            __ort_api->ReleaseValue(ort_value_output_output);
            ///////////////////

        }
        {
            float* X = &ONNX_96[0];
            float Y = ONNX_145;
            float* Z = ONNX_98;

            ///////////////////

            int64_t input_X_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_X;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (X)),
            3145728 * sizeof(float),
            input_X_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_X
            ));

            OrtValue* ort_value_input_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            &Y,
            1 * sizeof(float),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_Y
            ));

            int64_t output_Z_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_Z;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Z)),
            3145728 * sizeof(float),
            output_Z_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Z
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Pow_61_0_0_34, 0, ort_value_input_X));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Pow_61_0_0_34, 1, ort_value_input_Y));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Pow_61_0_0_34, 0, ort_value_output_Z));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Pow_61_0_0_34));
            __ort_api->ReleaseValue(ort_value_input_X);
            __ort_api->ReleaseValue(ort_value_input_Y);
            __ort_api->ReleaseValue(ort_value_output_Z);
            ///////////////////

        }
        {
            float* data = &ONNX_98[0];
            float* reduced = ONNX_99;

            ///////////////////

            int64_t input_data_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t output_reduced_dims[3] = {8, 512, 1};

            OrtValue* ort_value_output_reduced;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (reduced)),
            4096 * sizeof(float),
            output_reduced_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_reduced
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_ReduceMean_62_0_0_36, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_ReduceMean_62_0_0_36, 0, ort_value_output_reduced));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_ReduceMean_62_0_0_36));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_output_reduced);
            ///////////////////

        }
        {
            float* A = &ONNX_99[0];
            float B = ONNX_100;
            float* C = ONNX_101;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 1};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            4096 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            &B,
            1 * sizeof(float),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 1};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            4096 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_64_0_0_37, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_64_0_0_37, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_64_0_0_37, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_64_0_0_37));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* X = &ONNX_101[0];
            float* Y = ONNX_102;

            ///////////////////

            int64_t input_X_dims[3] = {8, 512, 1};

            OrtValue* ort_value_input_X;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (X)),
            4096 * sizeof(float),
            input_X_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_X
            ));

            int64_t output_Y_dims[3] = {8, 512, 1};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            4096 * sizeof(float),
            output_Y_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Sqrt_65_0_0_39, 0, ort_value_input_X));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Sqrt_65_0_0_39, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Sqrt_65_0_0_39));
            __ort_api->ReleaseValue(ort_value_input_X);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_95[0];
            float* B = &ONNX_102[0];
            float* C = ONNX_103;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[3] = {8, 512, 1};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            4096 * sizeof(float),
            input_B_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Div_66_0_0_40, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Div_66_0_0_40, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Div_66_0_0_40, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Div_66_0_0_40));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_103[0];
            float* B = &ONNX_attentionDOToutputDOTLayerNormDOTweight[0];
            float* C = ONNX_104;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Mul_67_0_0_41, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Mul_67_0_0_41, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Mul_67_0_0_41, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Mul_67_0_0_41));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_104[0];
            float* B = &ONNX_attentionDOToutputDOTLayerNormDOTbias[0];
            float* C = ONNX_105;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_68_0_0_43, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_68_0_0_43, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_68_0_0_43, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_68_0_0_43));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_105[0];
            float* B = &ONNX_146[0];
            float* Y = ONNX_107;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[2] = {768, 3072};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            2359296 * sizeof(float),
            input_B_dims,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_Y_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            12582912 * sizeof(float),
            output_Y_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_69_0_0_45, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_69_0_0_45, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_MatMul_69_0_0_45, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_MatMul_69_0_0_45));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_107[0];
            float* B = &ONNX_intermediateDOTdenseDOTbias[0];
            float* C = ONNX_108;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            12582912 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {3072};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            3072 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            12582912 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_70_0_0_47, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_70_0_0_47, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_70_0_0_47, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_70_0_0_47));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_108[0];
            float B = ONNX_109;
            float* C = ONNX_110;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            12582912 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            &B,
            1 * sizeof(float),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            12582912 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Div_72_0_0_49, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Div_72_0_0_49, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Div_72_0_0_49, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Div_72_0_0_49));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* input = &ONNX_110[0];
            float* output = ONNX_111;

            ///////////////////

            int64_t input_input_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_input_input;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (input)),
            12582912 * sizeof(float),
            input_input_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_input
            ));

            int64_t output_output_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_output_output;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (output)),
            12582912 * sizeof(float),
            output_output_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_output
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Erf_73_0_0_51, 0, ort_value_input_input));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Erf_73_0_0_51, 0, ort_value_output_output));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Erf_73_0_0_51));
            __ort_api->ReleaseValue(ort_value_input_input);
            __ort_api->ReleaseValue(ort_value_output_output);
            ///////////////////

        }
        {
            float* A = &ONNX_111[0];
            float B = ONNX_112;
            float* C = ONNX_113;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            12582912 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            &B,
            1 * sizeof(float),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            12582912 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_75_0_0_52, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_75_0_0_52, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_75_0_0_52, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_75_0_0_52));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_108[0];
            float* B = &ONNX_113[0];
            float* C = ONNX_114;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            12582912 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            12582912 * sizeof(float),
            input_B_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            12582912 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Mul_76_0_0_54, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Mul_76_0_0_54, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Mul_76_0_0_54, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Mul_76_0_0_54));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_114[0];
            float B = ONNX_115;
            float* C = ONNX_116;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            12582912 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            &B,
            1 * sizeof(float),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            12582912 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Mul_78_0_0_55, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Mul_78_0_0_55, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Mul_78_0_0_55, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Mul_78_0_0_55));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_116[0];
            float* B = &ONNX_147[0];
            float* Y = ONNX_118;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 3072};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            12582912 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[2] = {3072, 768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            2359296 * sizeof(float),
            input_B_dims,
            2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_Y_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            3145728 * sizeof(float),
            output_Y_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_79_0_0_57, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_MatMul_79_0_0_57, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_MatMul_79_0_0_57, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_MatMul_79_0_0_57));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_118[0];
            float* B = &ONNX_outputDOTdenseDOTbias[0];
            float* C = ONNX_119;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_80_0_0_59, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_80_0_0_59, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_80_0_0_59, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_80_0_0_59));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_119[0];
            float* B = &ONNX_105[0];
            float* C = ONNX_120;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            3145728 * sizeof(float),
            input_B_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_81_0_0_61, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_81_0_0_61, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_81_0_0_61, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_81_0_0_61));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* data = &ONNX_120[0];
            float* reduced = ONNX_122;

            ///////////////////

            int64_t input_data_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t output_reduced_dims[3] = {8, 512, 1};

            OrtValue* ort_value_output_reduced;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (reduced)),
            4096 * sizeof(float),
            output_reduced_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_reduced
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_ReduceMean_82_0_0_62, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_ReduceMean_82_0_0_62, 0, ort_value_output_reduced));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_ReduceMean_82_0_0_62));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_output_reduced);
            ///////////////////

        }
        {
            float* A = &ONNX_120[0];
            float* B = &ONNX_122[0];
            float* C = ONNX_123;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[3] = {8, 512, 1};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            4096 * sizeof(float),
            input_B_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Sub_83_0_0_63, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Sub_83_0_0_63, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Sub_83_0_0_63, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Sub_83_0_0_63));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* input = &ONNX_123[0];
            float* output = ONNX_124;

            ///////////////////

            int64_t input_input_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_input;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (input)),
            3145728 * sizeof(float),
            input_input_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_input
            ));

            int64_t output_output_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_output;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (output)),
            3145728 * sizeof(float),
            output_output_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_output
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Cast_84_0_0_64, 0, ort_value_input_input));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Cast_84_0_0_64, 0, ort_value_output_output));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Cast_84_0_0_64));
            __ort_api->ReleaseValue(ort_value_input_input);
            __ort_api->ReleaseValue(ort_value_output_output);
            ///////////////////

        }
        {
            float* X = &ONNX_124[0];
            float Y = ONNX_148;
            float* Z = ONNX_126;

            ///////////////////

            int64_t input_X_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_X;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (X)),
            3145728 * sizeof(float),
            input_X_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_X
            ));

            OrtValue* ort_value_input_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            &Y,
            1 * sizeof(float),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_Y
            ));

            int64_t output_Z_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_Z;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Z)),
            3145728 * sizeof(float),
            output_Z_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Z
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Pow_85_0_0_65, 0, ort_value_input_X));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Pow_85_0_0_65, 1, ort_value_input_Y));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Pow_85_0_0_65, 0, ort_value_output_Z));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Pow_85_0_0_65));
            __ort_api->ReleaseValue(ort_value_input_X);
            __ort_api->ReleaseValue(ort_value_input_Y);
            __ort_api->ReleaseValue(ort_value_output_Z);
            ///////////////////

        }
        {
            float* data = &ONNX_126[0];
            float* reduced = ONNX_127;

            ///////////////////

            int64_t input_data_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_data;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (data)),
            3145728 * sizeof(float),
            input_data_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_data
            ));

            int64_t output_reduced_dims[3] = {8, 512, 1};

            OrtValue* ort_value_output_reduced;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (reduced)),
            4096 * sizeof(float),
            output_reduced_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_reduced
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_ReduceMean_86_0_0_67, 0, ort_value_input_data));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_ReduceMean_86_0_0_67, 0, ort_value_output_reduced));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_ReduceMean_86_0_0_67));
            __ort_api->ReleaseValue(ort_value_input_data);
            __ort_api->ReleaseValue(ort_value_output_reduced);
            ///////////////////

        }
        {
            float* A = &ONNX_127[0];
            float B = ONNX_128;
            float* C = ONNX_129;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 1};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            4096 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            &B,
            1 * sizeof(float),
            nullptr,
            0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 1};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            4096 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_88_0_0_68, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_88_0_0_68, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_88_0_0_68, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_88_0_0_68));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* X = &ONNX_129[0];
            float* Y = ONNX_130;

            ///////////////////

            int64_t input_X_dims[3] = {8, 512, 1};

            OrtValue* ort_value_input_X;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (X)),
            4096 * sizeof(float),
            input_X_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_X
            ));

            int64_t output_Y_dims[3] = {8, 512, 1};

            OrtValue* ort_value_output_Y;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (Y)),
            4096 * sizeof(float),
            output_Y_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_Y
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Sqrt_89_0_0_70, 0, ort_value_input_X));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Sqrt_89_0_0_70, 0, ort_value_output_Y));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Sqrt_89_0_0_70));
            __ort_api->ReleaseValue(ort_value_input_X);
            __ort_api->ReleaseValue(ort_value_output_Y);
            ///////////////////

        }
        {
            float* A = &ONNX_123[0];
            float* B = &ONNX_130[0];
            float* C = ONNX_131;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[3] = {8, 512, 1};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            4096 * sizeof(float),
            input_B_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Div_90_0_0_71, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Div_90_0_0_71, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Div_90_0_0_71, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Div_90_0_0_71));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_131[0];
            float* B = &ONNX_outputDOTLayerNormDOTweight[0];
            float* C = ONNX_132;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Mul_91_0_0_72, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Mul_91_0_0_72, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Mul_91_0_0_72, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Mul_91_0_0_72));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        {
            float* A = &ONNX_132[0];
            float* B = &ONNX_outputDOTLayerNormDOTbias[0];
            float* C = ONNX_133;

            ///////////////////

            int64_t input_A_dims[3] = {8, 512, 768};

            OrtValue* ort_value_input_A;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (A)),
            3145728 * sizeof(float),
            input_A_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_A
            ));

            int64_t input_B_dims[1] = {768};

            OrtValue* ort_value_input_B;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (B)),
            768 * sizeof(float),
            input_B_dims,
            1,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_input_B
            ));

            int64_t output_C_dims[3] = {8, 512, 768};

            OrtValue* ort_value_output_C;
            __ort_check_status(__ort_api->CreateTensorWithDataAsOrtValue(
            __ort_cpu_mem_info,
            const_cast < void * > (reinterpret_cast < const void * > (C)),
            3145728 * sizeof(float),
            output_C_dims,
            3,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &ort_value_output_C
            ));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_92_0_0_74, 0, ort_value_input_A));
            __ort_check_status(__ort_api->ExecutableKernel_SetInput(__ort_kernel_ONNX_ONNX_Add_92_0_0_74, 1, ort_value_input_B));
            __ort_check_status(__ort_api->ExecutableKernel_SetOutput(__ort_kernel_ONNX_ONNX_Add_92_0_0_74, 0, ort_value_output_C));
            __ort_check_status(__ort_api->ExecutableKernel_Compute(__ort_kernel_ONNX_ONNX_Add_92_0_0_74));
            __ort_api->ReleaseValue(ort_value_input_A);
            __ort_api->ReleaseValue(ort_value_input_B);
            __ort_api->ReleaseValue(ort_value_output_C);
            ///////////////////

        }
        delete[] ONNX_18;
        delete[] ONNX_19;
        delete[] ONNX_21;
        delete[] ONNX_22;
        delete[] ONNX_24;
        delete[] ONNX_25;
        delete[] ONNX_39;
        delete[] ONNX_40;
        delete[] ONNX_54;
        delete[] ONNX_68;
        delete[] ONNX_69;
        delete[] ONNX_70;
        delete[] ONNX_71;
        delete[] ONNX_73;
        delete[] ONNX_74;
        delete[] ONNX_75;
        delete[] ONNX_76;
        delete[] ONNX_88;
        delete[] ONNX_90;
        delete[] ONNX_91;
        delete[] ONNX_92;
        delete[] ONNX_94;
        delete[] ONNX_95;
        delete[] ONNX_96;
        delete[] ONNX_98;
        delete[] ONNX_99;
        delete[] ONNX_101;
        delete[] ONNX_102;
        delete[] ONNX_103;
        delete[] ONNX_104;
        delete[] ONNX_105;
        delete[] ONNX_107;
        delete[] ONNX_108;
        delete[] ONNX_110;
        delete[] ONNX_111;
        delete[] ONNX_113;
        delete[] ONNX_114;
        delete[] ONNX_116;
        delete[] ONNX_118;
        delete[] ONNX_119;
        delete[] ONNX_120;
        delete[] ONNX_122;
        delete[] ONNX_123;
        delete[] ONNX_124;
        delete[] ONNX_126;
        delete[] ONNX_127;
        delete[] ONNX_129;
        delete[] ONNX_130;
        delete[] ONNX_131;
        delete[] ONNX_132;
    }
}

DACE_EXPORTED void __program_dace_model(float * __restrict__ ONNX_133, float * __restrict__ ONNX_134, float * __restrict__ ONNX_135, float * __restrict__ ONNX_136, long long * __restrict__ ONNX_137, long long * __restrict__ ONNX_138, long long * __restrict__ ONNX_139, long long * __restrict__ ONNX_140, long long * __restrict__ ONNX_141, long long * __restrict__ ONNX_142, long long * __restrict__ ONNX_143, float * __restrict__ ONNX_144, float * __restrict__ ONNX_146, float * __restrict__ ONNX_147, long long * __restrict__ ONNX___tmp0, long long * __restrict__ ONNX___tmp1, long long * __restrict__ ONNX___tmp16, long long * __restrict__ ONNX___tmp17, long long * __restrict__ ONNX___tmp18, long long * __restrict__ ONNX___tmp19, long long * __restrict__ ONNX___tmp2, long long * __restrict__ ONNX___tmp20, long long * __restrict__ ONNX___tmp21, long long * __restrict__ ONNX___tmp22, long long * __restrict__ ONNX___tmp23, long long * __restrict__ ONNX___tmp24, long long * __restrict__ ONNX___tmp25, long long * __restrict__ ONNX___tmp26, long long * __restrict__ ONNX___tmp27, long long * __restrict__ ONNX___tmp3, long long * __restrict__ ONNX___tmp4, long long * __restrict__ ONNX___tmp5, long long * __restrict__ ONNX___tmp6, long long * __restrict__ ONNX___tmp7, float * __restrict__ ONNX_attentionDOToutputDOTLayerNormDOTbias, float * __restrict__ ONNX_attentionDOToutputDOTLayerNormDOTweight, float * __restrict__ ONNX_attentionDOToutputDOTdenseDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTkeyDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTqueryDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTvalueDOTbias, float * __restrict__ ONNX_inputDOT1, float * __restrict__ ONNX_intermediateDOTdenseDOTbias, float * __restrict__ ONNX_outputDOTLayerNormDOTbias, float * __restrict__ ONNX_outputDOTLayerNormDOTweight, float * __restrict__ ONNX_outputDOTdenseDOTbias, float ONNX_100, float ONNX_109, float ONNX_112, float ONNX_115, float ONNX_128, float ONNX_145, float ONNX_148, long long ONNX_27, long long ONNX_30, long long ONNX_42, long long ONNX_45, long long ONNX_56, long long ONNX_59, float ONNX_72, long long ONNX_78, long long ONNX_81, long long ONNX___tmp10, long long ONNX___tmp11, long long ONNX___tmp12, long long ONNX___tmp13, long long ONNX___tmp14, long long ONNX___tmp15, long long ONNX___tmp8, long long ONNX___tmp9)
{
    __program_dace_model_internal(ONNX_133, ONNX_134, ONNX_135, ONNX_136, ONNX_137, ONNX_138, ONNX_139, ONNX_140, ONNX_141, ONNX_142, ONNX_143, ONNX_144, ONNX_146, ONNX_147, ONNX___tmp0, ONNX___tmp1, ONNX___tmp16, ONNX___tmp17, ONNX___tmp18, ONNX___tmp19, ONNX___tmp2, ONNX___tmp20, ONNX___tmp21, ONNX___tmp22, ONNX___tmp23, ONNX___tmp24, ONNX___tmp25, ONNX___tmp26, ONNX___tmp27, ONNX___tmp3, ONNX___tmp4, ONNX___tmp5, ONNX___tmp6, ONNX___tmp7, ONNX_attentionDOToutputDOTLayerNormDOTbias, ONNX_attentionDOToutputDOTLayerNormDOTweight, ONNX_attentionDOToutputDOTdenseDOTbias, ONNX_attentionDOTselfDOTkeyDOTbias, ONNX_attentionDOTselfDOTqueryDOTbias, ONNX_attentionDOTselfDOTvalueDOTbias, ONNX_inputDOT1, ONNX_intermediateDOTdenseDOTbias, ONNX_outputDOTLayerNormDOTbias, ONNX_outputDOTLayerNormDOTweight, ONNX_outputDOTdenseDOTbias, ONNX_100, ONNX_109, ONNX_112, ONNX_115, ONNX_128, ONNX_145, ONNX_148, ONNX_27, ONNX_30, ONNX_42, ONNX_45, ONNX_56, ONNX_59, ONNX_72, ONNX_78, ONNX_81, ONNX___tmp10, ONNX___tmp11, ONNX___tmp12, ONNX___tmp13, ONNX___tmp14, ONNX___tmp15, ONNX___tmp8, ONNX___tmp9);
}

DACE_EXPORTED int __dace_init_dace_model(float * __restrict__ ONNX_133, float * __restrict__ ONNX_134, float * __restrict__ ONNX_135, float * __restrict__ ONNX_136, long long * __restrict__ ONNX_137, long long * __restrict__ ONNX_138, long long * __restrict__ ONNX_139, long long * __restrict__ ONNX_140, long long * __restrict__ ONNX_141, long long * __restrict__ ONNX_142, long long * __restrict__ ONNX_143, float * __restrict__ ONNX_144, float * __restrict__ ONNX_146, float * __restrict__ ONNX_147, long long * __restrict__ ONNX___tmp0, long long * __restrict__ ONNX___tmp1, long long * __restrict__ ONNX___tmp16, long long * __restrict__ ONNX___tmp17, long long * __restrict__ ONNX___tmp18, long long * __restrict__ ONNX___tmp19, long long * __restrict__ ONNX___tmp2, long long * __restrict__ ONNX___tmp20, long long * __restrict__ ONNX___tmp21, long long * __restrict__ ONNX___tmp22, long long * __restrict__ ONNX___tmp23, long long * __restrict__ ONNX___tmp24, long long * __restrict__ ONNX___tmp25, long long * __restrict__ ONNX___tmp26, long long * __restrict__ ONNX___tmp27, long long * __restrict__ ONNX___tmp3, long long * __restrict__ ONNX___tmp4, long long * __restrict__ ONNX___tmp5, long long * __restrict__ ONNX___tmp6, long long * __restrict__ ONNX___tmp7, float * __restrict__ ONNX_attentionDOToutputDOTLayerNormDOTbias, float * __restrict__ ONNX_attentionDOToutputDOTLayerNormDOTweight, float * __restrict__ ONNX_attentionDOToutputDOTdenseDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTkeyDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTqueryDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTvalueDOTbias, float * __restrict__ ONNX_inputDOT1, float * __restrict__ ONNX_intermediateDOTdenseDOTbias, float * __restrict__ ONNX_outputDOTLayerNormDOTbias, float * __restrict__ ONNX_outputDOTLayerNormDOTweight, float * __restrict__ ONNX_outputDOTdenseDOTbias, float ONNX_100, float ONNX_109, float ONNX_112, float ONNX_115, float ONNX_128, float ONNX_145, float ONNX_148, long long ONNX_27, long long ONNX_30, long long ONNX_42, long long ONNX_45, long long ONNX_56, long long ONNX_59, float ONNX_72, long long ONNX_78, long long ONNX_81, long long ONNX___tmp10, long long ONNX___tmp11, long long ONNX___tmp12, long long ONNX___tmp13, long long ONNX___tmp14, long long ONNX___tmp15, long long ONNX___tmp8, long long ONNX___tmp9)
{
    int __result = 0;

    __ort_check_status(__ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &__ort_cpu_mem_info));
    __ort_check_status(__ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dace_graph", &__ort_env));
    __ort_check_status(__ort_api->CreateSessionOptions(&__ort_session_options));
    __ort_check_status(OrtSessionOptionsAppendExecutionProvider_CPU(__ort_session_options, /*use_arena=*/0));

    __ort_check_status(__ort_api->CreateKernelSession(__ort_session_options, &__ort_session, 12));

    {
        // Setup for ONNX_ONNX_MatMul_0_0_0_0
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_MatMul_0_0_0_0", "MatMul", &__ort_context_ONNX_ONNX_MatMul_0_0_0_0));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_0_0_0_0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_0_0_0_0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_MatMul_0_0_0_0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_MatMul_0_0_0_0, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_MatMul_0_0_0_0));
    } // end setup for context_ONNX_ONNX_MatMul_0_0_0_0
    {
        // Setup for ONNX_ONNX_Add_1_0_0_3
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_1_0_0_3", "Add", &__ort_context_ONNX_ONNX_Add_1_0_0_3));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_1_0_0_3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_1_0_0_3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_1_0_0_3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_1_0_0_3, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_1_0_0_3));
    } // end setup for context_ONNX_ONNX_Add_1_0_0_3
    {
        // Setup for ONNX_ONNX_MatMul_2_0_0_5
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_MatMul_2_0_0_5", "MatMul", &__ort_context_ONNX_ONNX_MatMul_2_0_0_5));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_2_0_0_5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_2_0_0_5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_MatMul_2_0_0_5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_MatMul_2_0_0_5, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_MatMul_2_0_0_5));
    } // end setup for context_ONNX_ONNX_MatMul_2_0_0_5
    {
        // Setup for ONNX_ONNX_Add_3_0_0_7
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_3_0_0_7", "Add", &__ort_context_ONNX_ONNX_Add_3_0_0_7));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_3_0_0_7, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_3_0_0_7, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_3_0_0_7, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_3_0_0_7, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_3_0_0_7));
    } // end setup for context_ONNX_ONNX_Add_3_0_0_7
    {
        // Setup for ONNX_ONNX_MatMul_4_0_0_9
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_MatMul_4_0_0_9", "MatMul", &__ort_context_ONNX_ONNX_MatMul_4_0_0_9));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_4_0_0_9, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_4_0_0_9, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_MatMul_4_0_0_9, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_MatMul_4_0_0_9, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_MatMul_4_0_0_9));
    } // end setup for context_ONNX_ONNX_MatMul_4_0_0_9
    {
        // Setup for ONNX_ONNX_Add_5_0_0_11
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_5_0_0_11", "Add", &__ort_context_ONNX_ONNX_Add_5_0_0_11));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_5_0_0_11, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_5_0_0_11, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_5_0_0_11, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_5_0_0_11, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_5_0_0_11));
    } // end setup for context_ONNX_ONNX_Add_5_0_0_11
    {
        // Setup for ONNX_ONNX_Reshape_15_0_0_13
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Reshape_15_0_0_13", "Reshape", &__ort_context_ONNX_ONNX_Reshape_15_0_0_13));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Reshape_15_0_0_13, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter shape
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Reshape_15_0_0_13, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        // Add parameter reshaped
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Reshape_15_0_0_13, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Reshape_15_0_0_13, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Reshape_15_0_0_13));
    } // end setup for context_ONNX_ONNX_Reshape_15_0_0_13
    {
        // Setup for ONNX_ONNX_Transpose_16_0_0_14
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Transpose_16_0_0_14", "Transpose", &__ort_context_ONNX_ONNX_Transpose_16_0_0_14));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Transpose_16_0_0_14, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter transposed
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Transpose_16_0_0_14, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute perm
            int64_t values[4];
            values[0] = 0;
            values[1] = 2;
            values[2] = 1;
            values[3] = 3;

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInts(__ort_context_ONNX_ONNX_Transpose_16_0_0_14, "perm", values, 4));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Transpose_16_0_0_14, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Transpose_16_0_0_14));
    } // end setup for context_ONNX_ONNX_Transpose_16_0_0_14
    {
        // Setup for ONNX_ONNX_Reshape_26_0_0_15
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Reshape_26_0_0_15", "Reshape", &__ort_context_ONNX_ONNX_Reshape_26_0_0_15));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Reshape_26_0_0_15, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter shape
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Reshape_26_0_0_15, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        // Add parameter reshaped
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Reshape_26_0_0_15, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Reshape_26_0_0_15, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Reshape_26_0_0_15));
    } // end setup for context_ONNX_ONNX_Reshape_26_0_0_15
    {
        // Setup for ONNX_ONNX_Reshape_36_0_0_16
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Reshape_36_0_0_16", "Reshape", &__ort_context_ONNX_ONNX_Reshape_36_0_0_16));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Reshape_36_0_0_16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter shape
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Reshape_36_0_0_16, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        // Add parameter reshaped
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Reshape_36_0_0_16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Reshape_36_0_0_16, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Reshape_36_0_0_16));
    } // end setup for context_ONNX_ONNX_Reshape_36_0_0_16
    {
        // Setup for ONNX_ONNX_Transpose_37_0_0_17
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Transpose_37_0_0_17", "Transpose", &__ort_context_ONNX_ONNX_Transpose_37_0_0_17));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Transpose_37_0_0_17, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter transposed
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Transpose_37_0_0_17, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute perm
            int64_t values[4];
            values[0] = 0;
            values[1] = 2;
            values[2] = 1;
            values[3] = 3;

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInts(__ort_context_ONNX_ONNX_Transpose_37_0_0_17, "perm", values, 4));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Transpose_37_0_0_17, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Transpose_37_0_0_17));
    } // end setup for context_ONNX_ONNX_Transpose_37_0_0_17
    {
        // Setup for ONNX_ONNX_Transpose_38_0_0_18
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Transpose_38_0_0_18", "Transpose", &__ort_context_ONNX_ONNX_Transpose_38_0_0_18));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Transpose_38_0_0_18, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter transposed
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Transpose_38_0_0_18, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute perm
            int64_t values[4];
            values[0] = 0;
            values[1] = 2;
            values[2] = 3;
            values[3] = 1;

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInts(__ort_context_ONNX_ONNX_Transpose_38_0_0_18, "perm", values, 4));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Transpose_38_0_0_18, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Transpose_38_0_0_18));
    } // end setup for context_ONNX_ONNX_Transpose_38_0_0_18
    {
        // Setup for ONNX_ONNX_MatMul_39_0_0_19
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_MatMul_39_0_0_19", "MatMul", &__ort_context_ONNX_ONNX_MatMul_39_0_0_19));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_39_0_0_19, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_39_0_0_19, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_MatMul_39_0_0_19, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_MatMul_39_0_0_19, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_MatMul_39_0_0_19));
    } // end setup for context_ONNX_ONNX_MatMul_39_0_0_19
    {
        // Setup for ONNX_ONNX_Div_41_0_0_20
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Div_41_0_0_20", "Div", &__ort_context_ONNX_ONNX_Div_41_0_0_20));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Div_41_0_0_20, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Div_41_0_0_20, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Div_41_0_0_20, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Div_41_0_0_20, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Div_41_0_0_20));
    } // end setup for context_ONNX_ONNX_Div_41_0_0_20
    {
        // Setup for ONNX_ONNX_Softmax_42_0_0_22
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Softmax_42_0_0_22", "Softmax", &__ort_context_ONNX_ONNX_Softmax_42_0_0_22));

        // Add parameter input
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Softmax_42_0_0_22, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter output
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Softmax_42_0_0_22, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute axis

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInt(__ort_context_ONNX_ONNX_Softmax_42_0_0_22, "axis", 3));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Softmax_42_0_0_22, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Softmax_42_0_0_22));
    } // end setup for context_ONNX_ONNX_Softmax_42_0_0_22
    {
        // Setup for ONNX_ONNX_MatMul_43_0_0_23
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_MatMul_43_0_0_23", "MatMul", &__ort_context_ONNX_ONNX_MatMul_43_0_0_23));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_43_0_0_23, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_43_0_0_23, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_MatMul_43_0_0_23, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_MatMul_43_0_0_23, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_MatMul_43_0_0_23));
    } // end setup for context_ONNX_ONNX_MatMul_43_0_0_23
    {
        // Setup for ONNX_ONNX_Transpose_44_0_0_24
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Transpose_44_0_0_24", "Transpose", &__ort_context_ONNX_ONNX_Transpose_44_0_0_24));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Transpose_44_0_0_24, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter transposed
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Transpose_44_0_0_24, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute perm
            int64_t values[4];
            values[0] = 0;
            values[1] = 2;
            values[2] = 1;
            values[3] = 3;

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInts(__ort_context_ONNX_ONNX_Transpose_44_0_0_24, "perm", values, 4));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Transpose_44_0_0_24, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Transpose_44_0_0_24));
    } // end setup for context_ONNX_ONNX_Transpose_44_0_0_24
    {
        // Setup for ONNX_ONNX_Reshape_54_0_0_25
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Reshape_54_0_0_25", "Reshape", &__ort_context_ONNX_ONNX_Reshape_54_0_0_25));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Reshape_54_0_0_25, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter shape
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Reshape_54_0_0_25, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));

        // Add parameter reshaped
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Reshape_54_0_0_25, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Reshape_54_0_0_25, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Reshape_54_0_0_25));
    } // end setup for context_ONNX_ONNX_Reshape_54_0_0_25
    {
        // Setup for ONNX_ONNX_MatMul_55_0_0_26
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_MatMul_55_0_0_26", "MatMul", &__ort_context_ONNX_ONNX_MatMul_55_0_0_26));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_55_0_0_26, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_55_0_0_26, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_MatMul_55_0_0_26, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_MatMul_55_0_0_26, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_MatMul_55_0_0_26));
    } // end setup for context_ONNX_ONNX_MatMul_55_0_0_26
    {
        // Setup for ONNX_ONNX_Add_56_0_0_28
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_56_0_0_28", "Add", &__ort_context_ONNX_ONNX_Add_56_0_0_28));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_56_0_0_28, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_56_0_0_28, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_56_0_0_28, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_56_0_0_28, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_56_0_0_28));
    } // end setup for context_ONNX_ONNX_Add_56_0_0_28
    {
        // Setup for ONNX_ONNX_Add_57_0_0_30
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_57_0_0_30", "Add", &__ort_context_ONNX_ONNX_Add_57_0_0_30));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_57_0_0_30, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_57_0_0_30, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_57_0_0_30, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_57_0_0_30, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_57_0_0_30));
    } // end setup for context_ONNX_ONNX_Add_57_0_0_30
    {
        // Setup for ONNX_ONNX_ReduceMean_58_0_0_31
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_ReduceMean_58_0_0_31", "ReduceMean", &__ort_context_ONNX_ONNX_ReduceMean_58_0_0_31));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_ReduceMean_58_0_0_31, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter reduced
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_ReduceMean_58_0_0_31, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute axes
            int64_t values[1];
            values[0] = -1;

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInts(__ort_context_ONNX_ONNX_ReduceMean_58_0_0_31, "axes", values, 1));
        }
        {
            // Setup attribute keepdims

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInt(__ort_context_ONNX_ONNX_ReduceMean_58_0_0_31, "keepdims", 1));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_ReduceMean_58_0_0_31, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_ReduceMean_58_0_0_31));
    } // end setup for context_ONNX_ONNX_ReduceMean_58_0_0_31
    {
        // Setup for ONNX_ONNX_Sub_59_0_0_32
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Sub_59_0_0_32", "Sub", &__ort_context_ONNX_ONNX_Sub_59_0_0_32));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Sub_59_0_0_32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Sub_59_0_0_32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Sub_59_0_0_32, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Sub_59_0_0_32, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Sub_59_0_0_32));
    } // end setup for context_ONNX_ONNX_Sub_59_0_0_32
    {
        // Setup for ONNX_ONNX_Cast_60_0_0_33
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Cast_60_0_0_33", "Cast", &__ort_context_ONNX_ONNX_Cast_60_0_0_33));

        // Add parameter input
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Cast_60_0_0_33, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter output
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Cast_60_0_0_33, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute to

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInt(__ort_context_ONNX_ONNX_Cast_60_0_0_33, "to", 1));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Cast_60_0_0_33, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Cast_60_0_0_33));
    } // end setup for context_ONNX_ONNX_Cast_60_0_0_33
    {
        // Setup for ONNX_ONNX_Pow_61_0_0_34
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Pow_61_0_0_34", "Pow", &__ort_context_ONNX_ONNX_Pow_61_0_0_34));

        // Add parameter X
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Pow_61_0_0_34, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Pow_61_0_0_34, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Z
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Pow_61_0_0_34, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Pow_61_0_0_34, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Pow_61_0_0_34));
    } // end setup for context_ONNX_ONNX_Pow_61_0_0_34
    {
        // Setup for ONNX_ONNX_ReduceMean_62_0_0_36
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_ReduceMean_62_0_0_36", "ReduceMean", &__ort_context_ONNX_ONNX_ReduceMean_62_0_0_36));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_ReduceMean_62_0_0_36, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter reduced
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_ReduceMean_62_0_0_36, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute axes
            int64_t values[1];
            values[0] = -1;

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInts(__ort_context_ONNX_ONNX_ReduceMean_62_0_0_36, "axes", values, 1));
        }
        {
            // Setup attribute keepdims

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInt(__ort_context_ONNX_ONNX_ReduceMean_62_0_0_36, "keepdims", 1));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_ReduceMean_62_0_0_36, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_ReduceMean_62_0_0_36));
    } // end setup for context_ONNX_ONNX_ReduceMean_62_0_0_36
    {
        // Setup for ONNX_ONNX_Add_64_0_0_37
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_64_0_0_37", "Add", &__ort_context_ONNX_ONNX_Add_64_0_0_37));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_64_0_0_37, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_64_0_0_37, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_64_0_0_37, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_64_0_0_37, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_64_0_0_37));
    } // end setup for context_ONNX_ONNX_Add_64_0_0_37
    {
        // Setup for ONNX_ONNX_Sqrt_65_0_0_39
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Sqrt_65_0_0_39", "Sqrt", &__ort_context_ONNX_ONNX_Sqrt_65_0_0_39));

        // Add parameter X
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Sqrt_65_0_0_39, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Sqrt_65_0_0_39, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Sqrt_65_0_0_39, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Sqrt_65_0_0_39));
    } // end setup for context_ONNX_ONNX_Sqrt_65_0_0_39
    {
        // Setup for ONNX_ONNX_Div_66_0_0_40
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Div_66_0_0_40", "Div", &__ort_context_ONNX_ONNX_Div_66_0_0_40));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Div_66_0_0_40, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Div_66_0_0_40, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Div_66_0_0_40, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Div_66_0_0_40, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Div_66_0_0_40));
    } // end setup for context_ONNX_ONNX_Div_66_0_0_40
    {
        // Setup for ONNX_ONNX_Mul_67_0_0_41
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Mul_67_0_0_41", "Mul", &__ort_context_ONNX_ONNX_Mul_67_0_0_41));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Mul_67_0_0_41, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Mul_67_0_0_41, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Mul_67_0_0_41, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Mul_67_0_0_41, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Mul_67_0_0_41));
    } // end setup for context_ONNX_ONNX_Mul_67_0_0_41
    {
        // Setup for ONNX_ONNX_Add_68_0_0_43
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_68_0_0_43", "Add", &__ort_context_ONNX_ONNX_Add_68_0_0_43));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_68_0_0_43, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_68_0_0_43, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_68_0_0_43, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_68_0_0_43, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_68_0_0_43));
    } // end setup for context_ONNX_ONNX_Add_68_0_0_43
    {
        // Setup for ONNX_ONNX_MatMul_69_0_0_45
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_MatMul_69_0_0_45", "MatMul", &__ort_context_ONNX_ONNX_MatMul_69_0_0_45));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_69_0_0_45, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_69_0_0_45, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_MatMul_69_0_0_45, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_MatMul_69_0_0_45, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_MatMul_69_0_0_45));
    } // end setup for context_ONNX_ONNX_MatMul_69_0_0_45
    {
        // Setup for ONNX_ONNX_Add_70_0_0_47
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_70_0_0_47", "Add", &__ort_context_ONNX_ONNX_Add_70_0_0_47));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_70_0_0_47, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_70_0_0_47, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_70_0_0_47, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_70_0_0_47, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_70_0_0_47));
    } // end setup for context_ONNX_ONNX_Add_70_0_0_47
    {
        // Setup for ONNX_ONNX_Div_72_0_0_49
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Div_72_0_0_49", "Div", &__ort_context_ONNX_ONNX_Div_72_0_0_49));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Div_72_0_0_49, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Div_72_0_0_49, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Div_72_0_0_49, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Div_72_0_0_49, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Div_72_0_0_49));
    } // end setup for context_ONNX_ONNX_Div_72_0_0_49
    {
        // Setup for ONNX_ONNX_Erf_73_0_0_51
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Erf_73_0_0_51", "Erf", &__ort_context_ONNX_ONNX_Erf_73_0_0_51));

        // Add parameter input
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Erf_73_0_0_51, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter output
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Erf_73_0_0_51, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Erf_73_0_0_51, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Erf_73_0_0_51));
    } // end setup for context_ONNX_ONNX_Erf_73_0_0_51
    {
        // Setup for ONNX_ONNX_Add_75_0_0_52
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_75_0_0_52", "Add", &__ort_context_ONNX_ONNX_Add_75_0_0_52));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_75_0_0_52, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_75_0_0_52, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_75_0_0_52, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_75_0_0_52, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_75_0_0_52));
    } // end setup for context_ONNX_ONNX_Add_75_0_0_52
    {
        // Setup for ONNX_ONNX_Mul_76_0_0_54
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Mul_76_0_0_54", "Mul", &__ort_context_ONNX_ONNX_Mul_76_0_0_54));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Mul_76_0_0_54, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Mul_76_0_0_54, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Mul_76_0_0_54, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Mul_76_0_0_54, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Mul_76_0_0_54));
    } // end setup for context_ONNX_ONNX_Mul_76_0_0_54
    {
        // Setup for ONNX_ONNX_Mul_78_0_0_55
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Mul_78_0_0_55", "Mul", &__ort_context_ONNX_ONNX_Mul_78_0_0_55));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Mul_78_0_0_55, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Mul_78_0_0_55, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Mul_78_0_0_55, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Mul_78_0_0_55, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Mul_78_0_0_55));
    } // end setup for context_ONNX_ONNX_Mul_78_0_0_55
    {
        // Setup for ONNX_ONNX_MatMul_79_0_0_57
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_MatMul_79_0_0_57", "MatMul", &__ort_context_ONNX_ONNX_MatMul_79_0_0_57));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_79_0_0_57, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_MatMul_79_0_0_57, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_MatMul_79_0_0_57, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_MatMul_79_0_0_57, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_MatMul_79_0_0_57));
    } // end setup for context_ONNX_ONNX_MatMul_79_0_0_57
    {
        // Setup for ONNX_ONNX_Add_80_0_0_59
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_80_0_0_59", "Add", &__ort_context_ONNX_ONNX_Add_80_0_0_59));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_80_0_0_59, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_80_0_0_59, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_80_0_0_59, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_80_0_0_59, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_80_0_0_59));
    } // end setup for context_ONNX_ONNX_Add_80_0_0_59
    {
        // Setup for ONNX_ONNX_Add_81_0_0_61
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_81_0_0_61", "Add", &__ort_context_ONNX_ONNX_Add_81_0_0_61));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_81_0_0_61, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_81_0_0_61, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_81_0_0_61, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_81_0_0_61, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_81_0_0_61));
    } // end setup for context_ONNX_ONNX_Add_81_0_0_61
    {
        // Setup for ONNX_ONNX_ReduceMean_82_0_0_62
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_ReduceMean_82_0_0_62", "ReduceMean", &__ort_context_ONNX_ONNX_ReduceMean_82_0_0_62));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_ReduceMean_82_0_0_62, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter reduced
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_ReduceMean_82_0_0_62, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute axes
            int64_t values[1];
            values[0] = -1;

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInts(__ort_context_ONNX_ONNX_ReduceMean_82_0_0_62, "axes", values, 1));
        }
        {
            // Setup attribute keepdims

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInt(__ort_context_ONNX_ONNX_ReduceMean_82_0_0_62, "keepdims", 1));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_ReduceMean_82_0_0_62, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_ReduceMean_82_0_0_62));
    } // end setup for context_ONNX_ONNX_ReduceMean_82_0_0_62
    {
        // Setup for ONNX_ONNX_Sub_83_0_0_63
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Sub_83_0_0_63", "Sub", &__ort_context_ONNX_ONNX_Sub_83_0_0_63));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Sub_83_0_0_63, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Sub_83_0_0_63, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Sub_83_0_0_63, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Sub_83_0_0_63, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Sub_83_0_0_63));
    } // end setup for context_ONNX_ONNX_Sub_83_0_0_63
    {
        // Setup for ONNX_ONNX_Cast_84_0_0_64
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Cast_84_0_0_64", "Cast", &__ort_context_ONNX_ONNX_Cast_84_0_0_64));

        // Add parameter input
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Cast_84_0_0_64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter output
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Cast_84_0_0_64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute to

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInt(__ort_context_ONNX_ONNX_Cast_84_0_0_64, "to", 1));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Cast_84_0_0_64, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Cast_84_0_0_64));
    } // end setup for context_ONNX_ONNX_Cast_84_0_0_64
    {
        // Setup for ONNX_ONNX_Pow_85_0_0_65
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Pow_85_0_0_65", "Pow", &__ort_context_ONNX_ONNX_Pow_85_0_0_65));

        // Add parameter X
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Pow_85_0_0_65, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Pow_85_0_0_65, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Z
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Pow_85_0_0_65, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Pow_85_0_0_65, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Pow_85_0_0_65));
    } // end setup for context_ONNX_ONNX_Pow_85_0_0_65
    {
        // Setup for ONNX_ONNX_ReduceMean_86_0_0_67
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_ReduceMean_86_0_0_67", "ReduceMean", &__ort_context_ONNX_ONNX_ReduceMean_86_0_0_67));

        // Add parameter data
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_ReduceMean_86_0_0_67, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter reduced
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_ReduceMean_86_0_0_67, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        {
            // Setup attribute axes
            int64_t values[1];
            values[0] = -1;

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInts(__ort_context_ONNX_ONNX_ReduceMean_86_0_0_67, "axes", values, 1));
        }
        {
            // Setup attribute keepdims

            __ort_check_status(__ort_api->ExecutableKernelContext_AddAttributeInt(__ort_context_ONNX_ONNX_ReduceMean_86_0_0_67, "keepdims", 1));
        }
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_ReduceMean_86_0_0_67, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_ReduceMean_86_0_0_67));
    } // end setup for context_ONNX_ONNX_ReduceMean_86_0_0_67
    {
        // Setup for ONNX_ONNX_Add_88_0_0_68
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_88_0_0_68", "Add", &__ort_context_ONNX_ONNX_Add_88_0_0_68));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_88_0_0_68, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_88_0_0_68, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_88_0_0_68, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_88_0_0_68, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_88_0_0_68));
    } // end setup for context_ONNX_ONNX_Add_88_0_0_68
    {
        // Setup for ONNX_ONNX_Sqrt_89_0_0_70
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Sqrt_89_0_0_70", "Sqrt", &__ort_context_ONNX_ONNX_Sqrt_89_0_0_70));

        // Add parameter X
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Sqrt_89_0_0_70, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter Y
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Sqrt_89_0_0_70, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Sqrt_89_0_0_70, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Sqrt_89_0_0_70));
    } // end setup for context_ONNX_ONNX_Sqrt_89_0_0_70
    {
        // Setup for ONNX_ONNX_Div_90_0_0_71
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Div_90_0_0_71", "Div", &__ort_context_ONNX_ONNX_Div_90_0_0_71));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Div_90_0_0_71, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Div_90_0_0_71, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Div_90_0_0_71, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Div_90_0_0_71, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Div_90_0_0_71));
    } // end setup for context_ONNX_ONNX_Div_90_0_0_71
    {
        // Setup for ONNX_ONNX_Mul_91_0_0_72
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Mul_91_0_0_72", "Mul", &__ort_context_ONNX_ONNX_Mul_91_0_0_72));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Mul_91_0_0_72, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Mul_91_0_0_72, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Mul_91_0_0_72, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Mul_91_0_0_72, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Mul_91_0_0_72));
    } // end setup for context_ONNX_ONNX_Mul_91_0_0_72
    {
        // Setup for ONNX_ONNX_Add_92_0_0_74
        __ort_check_status(__ort_api->CreateExecutableKernelContext("ONNX_ONNX_Add_92_0_0_74", "Add", &__ort_context_ONNX_ONNX_Add_92_0_0_74));

        // Add parameter A
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_92_0_0_74, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter B
        __ort_check_status(__ort_api->ExecutableKernelContext_AddInput(__ort_context_ONNX_ONNX_Add_92_0_0_74, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

        // Add parameter C
        __ort_check_status(__ort_api->ExecutableKernelContext_AddOutput(__ort_context_ONNX_ONNX_Add_92_0_0_74, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        // Setup attributes
        __ort_check_status(__ort_api->CreateExecutableKernel(__ort_session, __ort_context_ONNX_ONNX_Add_92_0_0_74, /*provider_index=*/0, &__ort_kernel_ONNX_ONNX_Add_92_0_0_74));
    } // end setup for context_ONNX_ONNX_Add_92_0_0_74

    return __result;
}

DACE_EXPORTED void __dace_exit_dace_model(float * __restrict__ ONNX_133, float * __restrict__ ONNX_134, float * __restrict__ ONNX_135, float * __restrict__ ONNX_136, long long * __restrict__ ONNX_137, long long * __restrict__ ONNX_138, long long * __restrict__ ONNX_139, long long * __restrict__ ONNX_140, long long * __restrict__ ONNX_141, long long * __restrict__ ONNX_142, long long * __restrict__ ONNX_143, float * __restrict__ ONNX_144, float * __restrict__ ONNX_146, float * __restrict__ ONNX_147, long long * __restrict__ ONNX___tmp0, long long * __restrict__ ONNX___tmp1, long long * __restrict__ ONNX___tmp16, long long * __restrict__ ONNX___tmp17, long long * __restrict__ ONNX___tmp18, long long * __restrict__ ONNX___tmp19, long long * __restrict__ ONNX___tmp2, long long * __restrict__ ONNX___tmp20, long long * __restrict__ ONNX___tmp21, long long * __restrict__ ONNX___tmp22, long long * __restrict__ ONNX___tmp23, long long * __restrict__ ONNX___tmp24, long long * __restrict__ ONNX___tmp25, long long * __restrict__ ONNX___tmp26, long long * __restrict__ ONNX___tmp27, long long * __restrict__ ONNX___tmp3, long long * __restrict__ ONNX___tmp4, long long * __restrict__ ONNX___tmp5, long long * __restrict__ ONNX___tmp6, long long * __restrict__ ONNX___tmp7, float * __restrict__ ONNX_attentionDOToutputDOTLayerNormDOTbias, float * __restrict__ ONNX_attentionDOToutputDOTLayerNormDOTweight, float * __restrict__ ONNX_attentionDOToutputDOTdenseDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTkeyDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTqueryDOTbias, float * __restrict__ ONNX_attentionDOTselfDOTvalueDOTbias, float * __restrict__ ONNX_inputDOT1, float * __restrict__ ONNX_intermediateDOTdenseDOTbias, float * __restrict__ ONNX_outputDOTLayerNormDOTbias, float * __restrict__ ONNX_outputDOTLayerNormDOTweight, float * __restrict__ ONNX_outputDOTdenseDOTbias, float ONNX_100, float ONNX_109, float ONNX_112, float ONNX_115, float ONNX_128, float ONNX_145, float ONNX_148, long long ONNX_27, long long ONNX_30, long long ONNX_42, long long ONNX_45, long long ONNX_56, long long ONNX_59, float ONNX_72, long long ONNX_78, long long ONNX_81, long long ONNX___tmp10, long long ONNX___tmp11, long long ONNX___tmp12, long long ONNX___tmp13, long long ONNX___tmp14, long long ONNX___tmp15, long long ONNX___tmp8, long long ONNX___tmp9)
{
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_92_0_0_74);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_92_0_0_74);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Mul_91_0_0_72);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Mul_91_0_0_72);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Div_90_0_0_71);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Div_90_0_0_71);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Sqrt_89_0_0_70);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Sqrt_89_0_0_70);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_88_0_0_68);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_88_0_0_68);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_ReduceMean_86_0_0_67);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_ReduceMean_86_0_0_67);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Pow_85_0_0_65);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Pow_85_0_0_65);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Cast_84_0_0_64);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Cast_84_0_0_64);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Sub_83_0_0_63);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Sub_83_0_0_63);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_ReduceMean_82_0_0_62);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_ReduceMean_82_0_0_62);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_81_0_0_61);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_81_0_0_61);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_80_0_0_59);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_80_0_0_59);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_MatMul_79_0_0_57);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_MatMul_79_0_0_57);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Mul_78_0_0_55);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Mul_78_0_0_55);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Mul_76_0_0_54);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Mul_76_0_0_54);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_75_0_0_52);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_75_0_0_52);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Erf_73_0_0_51);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Erf_73_0_0_51);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Div_72_0_0_49);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Div_72_0_0_49);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_70_0_0_47);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_70_0_0_47);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_MatMul_69_0_0_45);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_MatMul_69_0_0_45);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_68_0_0_43);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_68_0_0_43);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Mul_67_0_0_41);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Mul_67_0_0_41);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Div_66_0_0_40);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Div_66_0_0_40);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Sqrt_65_0_0_39);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Sqrt_65_0_0_39);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_64_0_0_37);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_64_0_0_37);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_ReduceMean_62_0_0_36);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_ReduceMean_62_0_0_36);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Pow_61_0_0_34);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Pow_61_0_0_34);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Cast_60_0_0_33);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Cast_60_0_0_33);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Sub_59_0_0_32);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Sub_59_0_0_32);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_ReduceMean_58_0_0_31);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_ReduceMean_58_0_0_31);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_57_0_0_30);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_57_0_0_30);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_56_0_0_28);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_56_0_0_28);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_MatMul_55_0_0_26);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_MatMul_55_0_0_26);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Reshape_54_0_0_25);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Reshape_54_0_0_25);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Transpose_44_0_0_24);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Transpose_44_0_0_24);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_MatMul_43_0_0_23);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_MatMul_43_0_0_23);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Softmax_42_0_0_22);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Softmax_42_0_0_22);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Div_41_0_0_20);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Div_41_0_0_20);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_MatMul_39_0_0_19);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_MatMul_39_0_0_19);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Transpose_38_0_0_18);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Transpose_38_0_0_18);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Transpose_37_0_0_17);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Transpose_37_0_0_17);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Reshape_36_0_0_16);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Reshape_36_0_0_16);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Reshape_26_0_0_15);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Reshape_26_0_0_15);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Transpose_16_0_0_14);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Transpose_16_0_0_14);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Reshape_15_0_0_13);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Reshape_15_0_0_13);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_5_0_0_11);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_5_0_0_11);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_MatMul_4_0_0_9);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_MatMul_4_0_0_9);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_3_0_0_7);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_3_0_0_7);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_MatMul_2_0_0_5);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_MatMul_2_0_0_5);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_Add_1_0_0_3);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_Add_1_0_0_3);
    __ort_api->ReleaseExecutableKernel(__ort_kernel_ONNX_ONNX_MatMul_0_0_0_0);
    __ort_api->ReleaseExecutableKernelContext(__ort_context_ONNX_ONNX_MatMul_0_0_0_0);

    __ort_api->ReleaseMemoryInfo(__ort_cpu_mem_info);
    __ort_api->ReleaseKernelSession(__ort_session);
    __ort_api->ReleaseSessionOptions(__ort_session_options);
    __ort_api->ReleaseEnv(__ort_env);

}

