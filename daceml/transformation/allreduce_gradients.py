import copy
import logging
from collections import deque
from typing import Dict

import numpy as np

import dace
import torch
from dace import data as dt, dtypes
from dace import registry
from dace.properties import make_properties
from dace.transformation import transformation
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil
from dace.memlet import Memlet
import dace.libraries.mpi as mpi
from mpi4py import MPI as MPI4PY

import daceml.onnx as donnx
from daceml.onnx.converters import clean_onnx_name
from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml.onnx import ONNXModel

log = logging.getLogger(__name__)

# blocklist of nondeterministic ops
# yapf: disable
NONDETERMINISTIC_OPS = {'ONNXDropout',
                        'ONNXGradient',
                        'ONNXGraphCall',
                        'ONNXIf',
                        'ONNXLoop',
                        'ONNXMomentum',
                        'ONNXMultinomial',
                        'ONNXRandomNormal',
                        'ONNXRandomNormalLike',
                        'ONNXRandomUniform',
                        'ONNXRandomUniformLike',
                        'ONNXSVMClassifier',
                        'ONNXSVMRegressor',
                        'ONNXScan',
                        'ONNXTreeEnsembleClassifier',
                        'ONNXTreeEnsembleRegressor'}
# yapf: enable

global UNIQUE_ID
UNIQUE_ID = 0


@registry.autoregister_params(singlestate=True)
#@registry.autoregister
@make_properties
class AllreduceGradients(transformation.Transformation):
    """ Remove nodes where all inputs are known and replace them with constant nodes by precomputing the output.
    """
    # pattern matching only checks that the type of the node matches,
    _access_node = transformation.PatternNode(nd.AccessNode)

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(AllreduceGradients._access_node)]
        #return []

    @staticmethod
    def can_be_applied(graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                       candidate: Dict[nd.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):

        node: nd.AccessNode = graph.nodes()[candidate[AllreduceGradients._access_node]]

        # SDFG must be imported from an ONNXModel
        #if not hasattr(sdfg, "_parent_onnx_model"):
        #    return False

        #if not 'ONNX' + node.schema.name not in NONDETERMINISTIC_OPS:
        #    return False

        if (("bias_gradient" in node.data or "weight_gradient" in node.data) 
            and len(graph.out_edges(node)) == 0 
            and (not isinstance(graph.in_edges(node)[0].src, mpi.nodes.allreduce.Allreduce))): 
            return True
        else:
            return False

        return False

    @staticmethod
    def match_to_str(graph, candidate):
        node: nd.AccessNode = graph.nodes()[candidate[AllreduceGradients._access_node]]
        return "Allreduce on gradients"

    def apply(self, sdfg: dace.SDFG):
        # Extract the subgraph, execute it and insert an AccessNode to the result
        # this method of execution is slow but simple. A better option would be to call the ORT
        # C API from a python object (like the OpChecker).

        #parent: ONNXModel = sdfg._parent_onnx_model
        state = sdfg.nodes()[self.state_id]
        gradient_node = state.nodes()[self.subgraph[AllreduceGradients._access_node]]

        allreduce_lib_node = mpi.nodes.allreduce.Allreduce("allreduce")

        sdfg.add_array(gradient_node.data+"_allreduce_sbuffer", 
                       sdfg.arrays[gradient_node.data].shape, 
                       sdfg.arrays[gradient_node.data].dtype, 
                       transient=True)
        allreduce_input = state.add_access(gradient_node.data+"_allreduce_sbuffer")
        
        in_edge = state.in_edges(gradient_node)[0]
        node_src = state.in_edges(gradient_node)[0].src
        for e in state.memlet_tree(in_edge):
            e.data.data = allreduce_input.data

        new_memlet = copy.deepcopy(in_edge.data)
        new_memlet.data = allreduce_input.data
        state.add_edge(node_src, in_edge.src_conn, allreduce_input,
                       None, new_memlet)

        
        state.remove_edge(in_edge)


        state.add_memlet_path(allreduce_input,
                              allreduce_lib_node,
                              dst_conn="_inbuffer",
                              memlet=Memlet.from_array(allreduce_input.data, allreduce_input.desc(sdfg)))
        state.add_memlet_path(allreduce_lib_node,
                              gradient_node,
                              src_conn="_outbuffer",
                              memlet=Memlet.from_array(gradient_node.data, gradient_node.desc(sdfg)))
        in_edge.data.wcr = None
