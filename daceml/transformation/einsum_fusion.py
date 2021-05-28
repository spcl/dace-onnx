import logging
from typing import Dict

import dace
import daceml.onnx as donnx
from dace import nodes
from dace import registry
from dace.properties import make_properties
from dace.sdfg import nodes as nd
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.transformation.transformation import PatternNode
from daceml.onnx import parse_variadic_param

log = logging.getLogger(__name__)


@registry.autoregister_params(singlestate=True)
@make_properties
class HorizontalEinsumFusion(transformation.Transformation):
    """ Fuse horizontal einsums
    """
    # pattern matching only checks that the type of the node matches,
    top = PatternNode(donnx.ONNXEinsum)
    access = PatternNode(nodes.AccessNode)
    bot = PatternNode(donnx.ONNXEinsum)

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(HorizontalEinsumFusion.top,
                                   HorizontalEinsumFusion.access,
                                   HorizontalEinsumFusion.bot)
        ]

    def can_be_applied(self,
                       graph: dace.sdfg.graph.OrderedMultiDiConnectorGraph,
                       candidate: Dict[nd.Node, int],
                       expr_index: int,
                       sdfg,
                       strict: bool = False):
        top: donnx.ONNXEinsum = self.top(sdfg)
        access: nodes.AccessNode = self.access(sdfg)
        bot: donnx.ONNXEinsum = self.bot(sdfg)
        top_inputs, top_outputs = top.equation.split("->")
        bot_inputs, bot_outputs = bot.equation.split("->")

        if len(top_inputs.split(",")) != 1 and len(bot_inputs.split(",")) != 1:
            return False

        if graph.in_degree(access) != 1:
            return False

        return True

    def apply(self, sdfg: dace.SDFG):
        state = sdfg.nodes()[self.state_id]

        top: donnx.ONNXEinsum = self.top(sdfg)
        access: nodes.AccessNode = self.access(sdfg)
        bot: donnx.ONNXEinsum = self.bot(sdfg)

        top_inputs, top_output = top.equation.split("->")
        top_inputs = top_inputs.split(",")
        bot_inputs, bot_output = bot.equation.split("->")
        bot_inputs = bot_inputs.split(",")

        if len(top_inputs) == 1:
            # fuse top into bottom
            # the index at which top connects to bot
            connector_on_bot = state.out_edges(access)[0].dst_conn
            name, input_idx = parse_variadic_param(connector_on_bot)
            assert name == "Inputs"

            input_str_to_fuse = bot_inputs[input_idx]
            assert len(input_str_to_fuse) == len(top_output)
            top_idx_to_bottom_idx = dict(zip(top_output, input_str_to_fuse))

            new_input_str = "".join(top_idx_to_bottom_idx[c]
                                    for c in top_inputs[0])
            bot_inputs[input_idx] = new_input_str
            bot.equation = f'{",".join(bot_inputs)}->{bot_output}'

            top_input_edge = state.in_edges(top)[0]
            state.add_edge(top_input_edge.src, top_input_edge.src_conn, bot,
                           connector_on_bot, top_input_edge.data)
            state.remove_node(access)
            state.remove_node(top)
        else:
            # fuse bottom into top
            assert len(bot_inputs) == 1

            connector_on_bot = state.out_edges(bot)[0].src_conn
            assert connector_on_bot == "Output"

            assert len(bot_inputs[0]) == len(top_output)
            bottom_idx_to_top_idx = dict(zip(bot_inputs[0], top_output))

            new_output_str = "".join(bottom_idx_to_top_idx[c]
                                     for c in bot_output)
            top.equation = f'{",".join(top_inputs)}->{new_output_str}'

            bot_output_edge = state.out_edges(bot)[0]
            state.add_edge(top, "Output", bot_output_edge.dst,
                           bot_output_edge.dst_conn, bot_output_edge.data)
            state.remove_node(access)
            state.remove_node(bot)
