import torch
from dace import dtypes, nodes as nd
from dace.transformation.dataflow import RedundantSecondArray, MapFusion, Vectorization
from efficientnet_pytorch.model import MBConvBlock, get_model_params

import daceml.onnx as donnx
from daceml.onnx.op_implementations.cudnn_implementations import CudnnConvolution
from daceml.pytorch import DaceModule
from daceml.testing.utils import torch_tensors_close
from daceml.transformation import ConstantFolding, ConstantDeviceCopyElimination, parameter_to_transient, TaskletFusion
from daceml.transformation.pad_conv_fusion import PadConvFusion

donnx.default_implementation = "pure"
donnx.ONNXConv.default_implementation = "cuDNN"
donnx.ONNXBatchNormalization.default_implementation = "cuDNN"
inputs = torch.rand(8, 32, 224, 224).cuda()
dy = torch.rand(8, 16, 224, 224).cuda()

block_params, global_params = get_model_params("efficientnet-b0", {})

pt_model = MBConvBlock(block_params[0], global_params).cuda()
pt_model.set_swish(memory_efficient=False)
dace_model = MBConvBlock(block_params[0], global_params).cuda()
dace_model.set_swish(memory_efficient=False)

dace_model.load_state_dict(pt_model.state_dict())

dace_model = DaceModule(dace_model, cuda=True)
def cf(dace_model):
    dace_model.sdfg.view()
    dace_model.sdfg.apply_transformations_repeated(
            {
                ConstantFolding, RedundantSecondArray,
                ConstantDeviceCopyElimination, #PadConvFusion
                },
            validate_all=True,
            strict=True)
    dace_model.sdfg.view()
dace_model.prepend_post_onnx_hook("cf", cf)

#dace_model.prepend_post_onnx_hook(
#    "cf",
#    lambda onnx_model: onnx_model.sdfg.apply_transformations_repeated(
#        {
#            ConstantFolding, RedundantSecondArray,
#            ConstantDeviceCopyElimination, PadConvFusion
#        },
#        validate_all=True,
#        strict=True))


def param_to_trans(model):
    for name, _ in model.pytorch_model.named_parameters():
        parameter_to_transient(model, name)


CudnnConvolution.default_algorithm = "implicit_gemm"
# def set_conv(model):
#     for n in model.sdfg.node(0).nodes():
#         if "Conv_81" in n.label:
#             n._algorithm = "winograd_nonfused"
#dace_model.prepend_post_onnx_hook("set_conv", set_conv)
# dace_model.append_post_onnx_hook("param_to_transient", param_to_trans)
def instrument(model):
    for n in model.sdfg.node(0).nodes():
        if "Conv_81" in n.label or "Conv_92" in n.label:
            n.instrument = dtypes.InstrumentationType.Timer

#dace_model.append_post_onnx_hook("instrument", instrument)
#dace_model.append_post_onnx_hook("save", lambda model: model.sdfg.save("/tmp/sdfg.sdfg"))
def fuse_and_vec(model: DaceModule):
    model.sdfg.apply_transformations_repeated(MapFusion, validate=True)
    model.sdfg.apply_transformations_repeated(TaskletFusion, validate=True)
    state = model.sdfg.node(0)

    def apply_vectorization_to_map_following_access(data):
        access = [n for n in state.nodes() if isinstance(n, nd.AccessNode) and n.data == data][0]
        map_e = state.out_edges(access)[0].dst
        tasklet = state.out_edges(map_e)[0].dst
        map_x = state.exit_node(map_e)
        Vectorization.apply_to(model.sdfg, _map_entry=map_e, _tasklet=tasklet, _map_exit=map_x)

    apply_vectorization_to_map_following_access("ONNX_99")
    apply_vectorization_to_map_following_access("ONNX_111")

dace_model.append_post_onnx_hook("fuse_and_vectorize", fuse_and_vec)
#dace_model.append_post_onnx_hook("view", lambda model: model.sdfg.view())

import time
# warmup
dace_output = dace_model(inputs)
pt_output = pt_model(inputs)
time.sleep(1)
for i in range(100):
    dace_output = dace_model(inputs)
# time.sleep(1)
# pt_output = pt_model(inputs)

torch_tensors_close("output", pt_output, dace_output)

#for (pt_name, pt_buf), (dace_name,
#                        dace_buf) in zip(pt_model.named_buffers(),
#                                         dace_model.named_buffers()):
#    assert pt_name in dace_name
#    if "num_batches_tracked" not in pt_name:
#        torch_tensors_close(pt_name, pt_buf, dace_buf)
