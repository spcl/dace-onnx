import functools
import logging
import typing
from functools import wraps

import dace
from dace import nodes as nd
from dace.libraries import blas
from dace.sdfg.state import MultiConnectorEdge
from dace.transformation import interstate, dataflow
from dace import SDFG, SDFGState
import dace.data as dt
from dace import dtypes
from dace.transformation.auto.auto_optimize import set_fast_implementations

from daceml.onnx.nodes.onnx_op import ONNXOp
from daceml import transformation

log = logging.getLogger(__name__)


def is_desc_contiguous(desc: dt.Data) -> bool:
    if type(desc) is dt.Scalar:
        return True
    elif type(desc) is dt.Array:
        contiguous_strides = [
            dt._prod(desc.shape[i + 1:]) for i in range(len(desc.shape))
        ]
        return desc.strides == contiguous_strides
    else:
        raise ValueError("Unsupported data descriptor type {}".format(
            type(desc)))


def is_desc_contiguous(desc: dt.Data) -> bool:
    if type(desc) is dt.Scalar:
        return True
    elif type(desc) is dt.Array:
        contiguous_strides = [
            dt._prod(desc.shape[i + 1:]) for i in range(len(desc.shape))
        ]
        return desc.strides == contiguous_strides
    else:
        raise ValueError("Unsupported data descriptor type {}".format(
            type(desc)))


def in_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG,
                      name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to input connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the input connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[in_edge_with_name(node, state, name).data.data]


def out_desc_with_name(node: nd.Node, state: SDFGState, sdfg: SDFG,
                       name: str) -> dt.Data:
    """ Find the descriptor of the data that connects to output connector `name`.
        :param node: the node.
        :param sdfg: the sdfg.
        :param state: the state.
        :param name: the output connector name.
        :return: the descriptor of the data that connects to connector `name`.
     """
    return sdfg.arrays[out_edge_with_name(node, state, name).data.data]


def in_edge_with_name(node: nd.Node, state: SDFGState,
                      name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to input connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the input connector name.
        :return: the edge that connects to connector `name`.
     """

    cands = list(state.in_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError(
            "Expected to find exactly one edge with name '{}', found {}".
            format(name, len(cands)))
    return cands[0]


def out_edge_with_name(node: nd.Node, state: SDFGState,
                       name: str) -> MultiConnectorEdge:
    """ Find the edge that connects to output connector `name` on `node`.
        :param node: the node.
        :param state: the state.
        :param name: the output connector name.
        :return: the edge that connects to connector `name`.
     """
    cands = list(state.out_edges_by_connector(node, name))
    if len(cands) != 1:
        raise ValueError(
            "Expected to find exactly one edge with name '{}', found {}".
            format(name, len(cands)))
    return cands[0]


def find_str_not_in_set(existing: typing.Set[str],
                        target_str: typing.Optional[str]) -> str:
    """ Try to find a new str that is not in the set.

        :param existing: the existing strs.
        :param target_str: (optional) a target_str that should be used as a base for the new str.
        :return: a new str that is not in `existing`.
    """
    base_name = target_str or "temp"

    if base_name not in existing:
        return base_name

    i = 0
    while (base_name + "_" + str(i)) in existing:
        i += 1
    return base_name + "_" + str(i)


def vectorize_array_and_memlet(sdfg, array_name, type: dtypes.typeclass):
    '''
       Adjust the shape of a data container according to the vec width (only the last dimension).
       This will change its shape and strides
       together with the all the ingoin/outgoing memlets
    '''
    # find the array
    data = sdfg.arrays[array_name]
    if type == data.dtype:
        return
    #change the type
    data.dtype = type

    #adjust the shape
    vec_width = type.veclen
    if data.shape[-1] % vec_width != 0:
        raise ValueError("Shape of {} is not divisible by {}".format(
            data, vec_width))
    data.shape = data.shape[:-1] + (data.shape[-1] // vec_width, )

    # #adjust all the strides
    for stride in data.strides[:-1]:
        if stride % vec_width != 0:
            raise ValueError("Stride of {} is not divisible by {}".format(
                data.name, vec_width))

    data.strides = tuple(ti // vec_width
                         for ti in data.strides[:-1]) + (data.strides[-1], )

    # Search for all the memlets
    for state in sdfg.nodes():
        for edge in state.edges():
            if edge.data.data == array_name:
                # get the range
                start, stop, skip = edge.data.subset.ranges[-1]

                # Let's be conservative for the moment

                if start != 0 or skip != 1 or (stop + 1) % vec_width != 0:
                    raise ValueError(
                        "Memlet {} not able to convert its range".format(
                            edge.data))

                #update the range
                new_stop = (stop + 1) // vec_width - 1
                edge.data.subset.ranges[-1] = (start, new_stop, skip)


def expand_onnx_nodes(sdfg: dace.SDFG):
    """ Recursively expand all onnx library nodes in the SDFG, resulting in an SDFG that can be optimized by
        dace transformations. Will also specialize dace matmuls.

        :param sdfg: the sdfg to expand nodes on.
    """
    states = list(sdfg.states())
    while len(states) > 0:
        state = states.pop()
        expanded_something = False
        for node in list(state.nodes()):  # Make sure we have a copy
            if isinstance(node, nd.NestedSDFG):
                expand_onnx_nodes(node.sdfg)
            elif isinstance(node, ONNXOp) or isinstance(node, blas.MatMul):
                impl_name = node.expand(sdfg, state)
                print(
                    "Automatically expanded library node \"{}\" with implementation \"{}\"."
                    .format(str(node), impl_name))
                # We made a copy of the original list of nodes, so we keep
                # iterating even though this list has now changed
                expanded_something = True
        if expanded_something:
            states.append(state)  # Nodes have changed. Check state again


def auto_optimize(sdfg: dace.SDFG,
                  cuda,
                  apply_strict=False,
                  fold_constants=True):
    """ Automatically optimize ``sdfg``.

        :param sdfg: the sdfg to optimize (inplace).
        :param cuda: whether to optimize for cuda.
        :param apply_strict: whether to apply strict transformations to the sdfg after optimization.
        :param fold_constants: whether to apply constant folding.
    """
    log.debug("Applying automatic optimizations")
    if fold_constants:
        log.debug("Applying constant folding")
        sdfg.apply_transformations_repeated(
            [transformation.ConstantFolding, dataflow.RedundantSecondArray],
            validate_all=True,
            strict=True)
    log.debug("Expanding ONNX nodes")
    expand_onnx_nodes(sdfg)
    log.debug("Setting fast implementations")
    # MKL is currently broken
    set_fast_implementations(
        sdfg,
        dace.DeviceType.GPU if cuda else dace.DeviceType.CPU,
        blocklist=["MKL"])
    if apply_strict:
        log.debug("Applying strict transforms")
        # there is a nondeterministic bug in redundant array that appears if
        # we don't apply inline first
        sdfg.apply_transformations_repeated(interstate.InlineSDFG)
        sdfg.apply_strict_transformations()


def iterables_equal(a, b) -> bool:
    """ Return whether the two iterables ``a`` and ``b`` are equal. """
    if len(a) != len(b):
        return False
    return all(x == y for x, y in zip(a, b))


def prod(sequence):
    return functools.reduce(lambda a, b: a * b, sequence, 1)
