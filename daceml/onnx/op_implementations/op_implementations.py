import copy
from math import sqrt

import dace
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.symbolic import symstr

from daceml.onnx.implementation_repository import register_pure_expansion



@register_pure_expansion("Div")
def expansion(node, state, sdfg):
    print("div expanded")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def divop(A: atype, B: btype, C: ctype):
        C[:] = A / B

    return divop.to_sdfg()

@register_pure_expansion("Add")
def expansion(node, state, sdfg):
    print("add expanded")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def addop(A: atype, B: btype, C: ctype):
        C[:] = A + B

    return addop.to_sdfg()

@register_pure_expansion("Mul")
def expansion(node, state, sdfg):
    print("mul expanded")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def mulop(A: atype, B: btype, C: ctype):
        C[:] = A * B

    return mulop.to_sdfg()

@register_pure_expansion("Sub")
def expansion(node, state, sdfg):
    print("sub expanded")
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def subop(A: atype, B: btype, C: ctype):
        C[:] = A - B

    return subop.to_sdfg()


@register_pure_expansion("ReduceMean")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    axes = None
    keepdims = None
    for name, attr in node.schema.attributes.items():
        if hasattr(node, name):
            if str(node.schema.attributes[name]) == "axes":
                axes = getattr(node, name)
            elif str(node.schema.attributes[name]) == "keepdims":
                keepdims = getattr(node, name)

    assert(axes == [-1] and keepdims ==1)

    node.validate(sdfg, state)
    sdfg_exp = dace.SDFG('reducemeanExpansion')

    in_edges = state.in_edges(node)

    mm = in_edges[0].data.subset.size()[0]
    nn = in_edges[0].data.subset.size()[1]
    gg = in_edges[0].data.subset.size()[2]

    M = str(mm)
    N = str(nn)
    G = str(gg)

    sdfg_exp.add_array('data', (mm, nn, gg), dace.float32)
    sdfg_exp.add_array('reduced', (mm, nn, 1), dace.float32)
    state_exp = sdfg_exp.add_state()

    me, mx = state_exp.add_map('outer_map', dict(i='0:' + M, j='0:' + N))

    data = state_exp.add_read('data')
    reduced = state_exp.add_access('reduced')

    redsum = state_exp.add_reduce('lambda a1, b1: a1 + b1', None, 0)
    tmp_sum = state_exp.add_transient('tmp_sum', (1, ), dace.float32)

    tmean = state_exp.add_tasklet('meantasklet', {'tsum'}, {'mean'},
                                  'mean = tsum / (%s)' % G)

    state_exp.add_edge(
        data, None, me, None,
        dace.Memlet.simple(data, '0:' + M + ', 0:' + N + ', 0:' + G))
    state_exp.add_edge(me, None, redsum, None,
                       dace.Memlet.simple(data, 'i, j, 0:' + G))
    state_exp.add_edge(redsum, None, tmp_sum, None,
                       dace.Memlet.simple(tmp_sum, '0'))
    state_exp.add_edge(tmp_sum, None, tmean, 'tsum',
                       dace.Memlet.simple(tmp_sum, '0'))
    state_exp.add_edge(tmean, 'mean', mx, None,
                       dace.Memlet.simple(reduced, 'i, j, 0'))
    state_exp.add_edge(
        mx, None, reduced, None,
        dace.Memlet.simple(reduced, '0:' + M + ', 0:' + N + ', 0'))
    sdfg_exp.fill_scope_connectors()

    return sdfg_exp

@register_pure_expansion("Sqrt")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def sqrtop(X: atype, Y: btype):
        # Y[:] = X ** dace.float32(0.5)
        Y[:] = sqrt(X)

    return sqrtop.to_sdfg()

@register_pure_expansion("Pow")
def expansion(node, state, sdfg):
    inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
    node.validate(sdfg, state)

    in_edges = state.in_edges(node)
    out_edges = state.out_edges(node)

    atype = copy.deepcopy(sdfg.arrays[in_edges[0].data.data])
    btype = copy.deepcopy(sdfg.arrays[in_edges[1].data.data])
    ctype = copy.deepcopy(sdfg.arrays[out_edges[0].data.data])

    @dace.program
    def powop(X: atype, Y: btype, Z: ctype):
        Z[:] = X ** Y

    return powop.to_sdfg()
