import os
import logging

import dace.library
from dace.config import Config

log = logging.getLogger(__name__)

if 'ORT_ROOT' not in os.environ and 'ORT_RELEASE' not in os.environ:
    raise ValueError("This environment expects the environment variable "
                     "ORT_ROOT or ORT_RELEASE to be set (see README.md)")

if Config.get("compiler", "cuda", "max_concurrent_streams") != -1:
    log.info("Setting compiler.cuda.max_concurrent_streams to -1")
    Config.set("compiler", "cuda", "max_concurrent_streams", value=-1)


def _get_src_includes():
    """
    Get the includes and dll path when ORT is built from source
    """
    ort_path = os.path.abspath(os.environ['ORT_ROOT'])
    cand_path = os.path.join(ort_path, "build", "Linux",
                             dace.Config.get("compiler", "build_type"))

    if os.path.isdir(cand_path):
        ort_build_path = cand_path
    else:
        ort_build_path = os.path.join(ort_path, "build", "Linux", "Release")

    ort_dll_path = os.path.join(ort_build_path, "libonnxruntime.so")
    includes = [
        os.path.join(ort_path, "include", "onnxruntime", "core", "session"),
        os.path.join(ort_path, "include", "onnxruntime", "core", "providers",
                     "cpu"),
        os.path.join(ort_path, "include", "onnxruntime", "core", "providers",
                     "cuda")
    ]
    return includes, ort_dll_path


def _get_dist_includes():
    """
    Get the includes and dll path when ORT is used from the distribution package
    """
    ort_path = os.path.abspath(os.environ['ORT_RELEASE'])
    includes = [os.path.join(ort_path, 'include')]
    ort_dll_path = os.path.join(ort_path, 'lib', 'libonnxruntime.so')
    return includes, ort_dll_path


if 'ORT_RELEASE' in os.environ:
    log.debug("Using ORT_RELEASE")
    INCLUDES, ORT_DLL_PATH = _get_dist_includes()
else:
    log.debug("Using ORT_ROOT")
    INCLUDES, ORT_DLL_PATH = _get_src_includes()


@dace.library.environment
class ONNXRuntime:
    """ Environment used to run ONNX operator nodes using ONNX Runtime.
        See :ref:`ort-installation` for installation instructions.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = INCLUDES
    cmake_libraries = [ORT_DLL_PATH]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []
    state_fields = [
        "const OrtApi* ort_api;", "OrtEnv* ort_env;",
        "OrtKernelSession* ort_session;",
        "OrtSessionOptions* ort_session_options;",
        "OrtMemoryInfo* ort_cpu_mem_info;"
    ]
    dependencies = []

    headers = [
        "../include/dace_onnx.h",
        "onnxruntime_c_api.h",
        "cpu_provider_factory.h",
        "cuda_provider_factory.h",
    ]
    init_code = """
    __state->ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    __ort_check_status(__state->ort_api, __state->ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, /*type=*/OrtMemTypeDefault, &__state->ort_cpu_mem_info));
    __ort_check_status(__state->ort_api, __state->ort_api->CreateEnv(/*default_logging_level=*/ORT_LOGGING_LEVEL_WARNING, /*logid=*/"dace_graph", &__state->ort_env));
    __ort_check_status(__state->ort_api, __state->ort_api->CreateSessionOptions(&__state->ort_session_options));
    __ort_check_status(__state->ort_api, OrtSessionOptionsAppendExecutionProvider_CPU(__state->ort_session_options, /*use_arena=*/0));
    __ort_check_status(__state->ort_api, __state->ort_api->CreateKernelSession(__state->ort_session_options, &__state->ort_session, /*opset_version=*/12));
    """
    finalize_code = """
    __state->ort_api->ReleaseMemoryInfo(__state->ort_cpu_mem_info);
    __state->ort_api->ReleaseKernelSession(__state->ort_session);
    __state->ort_api->ReleaseSessionOptions(__state->ort_session_options);
    __state->ort_api->ReleaseEnv(__state->ort_env);
    """


@dace.library.environment
class ONNXRuntimeCUDA:
    """ Environment used to run ONNX operator nodes using ONNX Runtime, with the CUDA execution provider.
        See :ref:`ort-installation` for installation instructions.
    """

    cmake_minimum_version = None
    cmake_packages = []
    cmake_variables = {}
    cmake_includes = INCLUDES
    cmake_libraries = [ORT_DLL_PATH]
    cmake_compile_flags = []
    cmake_link_flags = []
    cmake_files = []
    state_fields = [
        "OrtMemoryInfo* ort_cuda_mem_info;",
        "OrtMemoryInfo* ort_cuda_pinned_mem_info;"
    ]
    dependencies = [ONNXRuntime]

    headers = []
    init_code = """
    __ort_check_status(__state->ort_api, __state->ort_api->CreateMemoryInfo("Cuda", /*allocator_type=*/OrtDeviceAllocator, /*device=*/0, /*mem_type=*/OrtMemTypeDefault, &__state->ort_cuda_mem_info));
    __ort_check_status(__state->ort_api, __state->ort_api->CreateMemoryInfo("CudaPinned", /*allocator_type=*/OrtDeviceAllocator, /*device=*/0, /*mem_type=*/OrtMemTypeCPU, &__state->ort_cuda_pinned_mem_info));
    
    OrtCUDAProviderOptions options = {
        .device_id = 0,
        .has_user_compute_stream = 1,
        .user_compute_stream = nullptr
    };
    __ort_check_status(__state->ort_api, __state->ort_api->SessionOptionsAppendExecutionProvider_CUDA(__state->ort_session_options, &options));
    
    // overwrite the CPU ORT session with the CUDA session
    
    __state->ort_api->ReleaseKernelSession(__state->ort_session);
    __ort_check_status(__state->ort_api, __state->ort_api->CreateKernelSession(__state->ort_session_options, &__state->ort_session, /*opset_version=*/12));
    """

    finalize_code = """
    __state->ort_api->ReleaseMemoryInfo(__state->ort_cuda_mem_info);
    __state->ort_api->ReleaseMemoryInfo(__state->ort_cuda_pinned_mem_info);
    """
