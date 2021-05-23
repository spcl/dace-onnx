import torch
from dace.transformation.transformation import Transformation, PatternNode
from dace.sdfg.utils import node_path_graph
from dace import nodes, SDFG, SDFGState, registry, Memlet
from typing import Dict, Union

from daceml import onnx as donnx
from daceml.transformation.constant_folding import remove_node_and_computation
from daceml.util import utils


@registry.autoregister_params(singlestate=True)
class PadConvFusion(Transformation):
    """ Fuse a constant pad into a convolution.
    """

    pad = PatternNode(donnx.ONNXPad)
    data = PatternNode(nodes.AccessNode)
    conv = PatternNode(donnx.ONNXConv)

    @classmethod
    def expressions(cls):
        return [node_path_graph(cls.pad, cls.data, cls.conv)]

    def can_be_applied(self, graph: SDFGState, candidate: Dict[PatternNode,
                                                               int],
                       expr_index: int, sdfg: SDFG, strict: bool) -> bool:
        pad: donnx.ONNXPad = self.pad(sdfg)
        data_node: nodes.AccessNode = self.data(sdfg)
        conv: donnx.ONNXConv = self.conv(sdfg)

        if pad.mode != 'constant':
            return False

        # Check if data in access node is used anywhere else
        other_nodes = [
            node for state in sdfg.nodes() for node in state.nodes() if
            isinstance(node, nodes.AccessNode) and node.data == data_node.data
        ]
        if len(other_nodes) != 1:
            return False

        # conservative: padded value should be 4 dimensional
        if len(data_node.desc(sdfg).shape) != 4:
            return False

        # no other out edges
        if graph.in_degree(data_node) != 1 or graph.out_degree(data_node) != 1:
            return False

        # check that the two pad inputs can be folded
        constant_value = list(
            graph.in_edges_by_connector(pad, "constant_value"))[0].data.data
        pads = list(graph.in_edges_by_connector(pad, "pads"))[0].data.data
        if constant_value not in sdfg._parent_onnx_model.clean_weights:
            return False
        if pads not in sdfg._parent_onnx_model.clean_weights:
            return False

        pads_value: torch.Tensor = sdfg._parent_onnx_model.clean_weights[pads]
        constant_value_value: torch.Tensor = sdfg._parent_onnx_model.clean_weights[
            constant_value]
        if constant_value_value != 0:
            return False

        if len(pads_value.shape) != 1 or pads_value.shape[0] != 8:
            return False

        # can only eliminate the pad if it is along the spatial axes
        # pads_value[i::4] gets the padding at the start and end of the i-th axis
        if (not utils.iterables_equal(pads_value[0::4], [0, 0])
                and utils.iterables_equal(pads_value[1::4], [0, 0])):
            return False

        return True

    def apply(self, sdfg: SDFG):
        state: SDFGState = sdfg.node(self.state_id)
        pad: donnx.ONNXPad = self.pad(sdfg)
        data_node: nodes.AccessNode = self.data(sdfg)
        conv: donnx.ONNXConv = self.conv(sdfg)

        pads = list(state.in_edges_by_connector(pad, "pads"))[0].data.data

        pads_value: torch.Tensor = sdfg._parent_onnx_model.clean_weights[pads]

        conv.pads[0] += int(pads_value[2::4][0])
        conv.pads[2] += int(pads_value[2::4][1])
        conv.pads[1] += int(pads_value[3::4][0])
        conv.pads[3] += int(pads_value[3::4][1])

        in_edge = next(state.in_edges_by_connector(pad, "data"))
        state.add_edge(in_edge.src, in_edge.src_conn, conv, "X", in_edge.data)
        state.remove_edge(in_edge)
        remove_node_and_computation(sdfg, state, data_node)
