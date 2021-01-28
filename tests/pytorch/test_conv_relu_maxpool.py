# Simple test for evaluating Conv-Relu-Maxpool

# TODO: conform to pytest syntax if needed
# TODO: render this a real test

from dace.transformation.interstate import FPGATransformSDFG, InlineSDFG


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import daceml.onnx as donnx
import dace
from daceml.pytorch import DaceModule, dace_module
import copy

from daceml.util import utils
from dace.transformation.dataflow import streaming_memory as sm
from dace.transformation.dataflow import PruneConnectors
from dace.transformation.interstate import InlineSDFG
import argparse


def get_access_node_by_name(sdfg, name):

    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.AccessNode):
            # print(node.label)
            if node.label == name:
                return node, state

    raise Exception("DataNode {} not found".format(name))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return x

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W",
                        type=int,
                        nargs="?",
                        default=1,
                        help="Vectorization width")

    args = vars(parser.parse_args())
    vec_width = args["W"]


    import daceml.onnx as donnx
    donnx.default_implementation = "pure"
    donnx.ONNXConv.default_implementation = 'im2col'

    ptmodel = Model()

    data_shape = (10, 1, 28, 28)
    x = torch.rand(data_shape)


    dace_model = DaceModule(ptmodel)
    dace_output = dace_model(x)

    torch_output = ptmodel(x)


    assert np.allclose(torch_output.detach().numpy(), dace_output, atol=1e-06)

    sdfg = dace_model.sdfg
    ##################################
    # Vectorize input and output container
    vec_width = vec_width

    vec_type = dace.vector(dace.float32, vec_width)

    # vectorize output of Conv
    utils.vectorize_array_and_memlet(sdfg, "ONNX_3", vec_type)
    # vectorize output of Relu
    utils.vectorize_array_and_memlet(sdfg, "ONNX_4", vec_type)

    sdfg.save('/tmp/out.sdfg')
    ###################################

    ############################################################
    # Transform to FPGA

    donnx.ONNXConv.default_implementation = "fpga"
    donnx.ONNXRelu.default_implementation = "fpga"
    donnx.ONNXMaxPool.default_implementation = "fpga"


    # Apply transformations

    sdfg.apply_transformations([FPGATransformSDFG])
    sdfg.expand_library_nodes()
    sdfg.apply_transformations_repeated([InlineSDFG])
    sdfg.save('/tmp/out_fpga_expanded.sdfg')
    dace_output_fpga = dace_model(torch.clone(x))

    #reshape if vec_width is different than 1
    dace_output_fpga= dace_output_fpga.reshape(dace_output.shape)


    torch_output_numpy = torch_output.detach().numpy()
    diff = np.linalg.norm(torch_output_numpy-dace_output_fpga)/dace_output_fpga.size

    print("Difference: ", diff)
    assert (diff < 1e-6)
