#!/usr/bin/env python
# coding: utf-8

# In[8]:
import dace
import torch
from daceml.pytorch import dace_module
import daceml.onnx
from daceml import transformation
from daceml.util import utils
daceml.onnx.default_implementation = "pure"

# to check correctness later
class PyTorchMish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

def pt_no_module(x):
    x = x * (torch.tanh(torch.nn.functional.softplus(x)))
    return x

    
@dace_module(cuda=True, backward=True)
class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

pt_func = PyTorchMish()
dace_func = Mish()


# In[9]:


#from dace.transformation.dataflow import TrivialMapRangeElimination, Vectorization
#from dace.transformation.subgraph import SubgraphFusion
#def fuse_sg(module):
#    fwd_sdfg = module.sdfg
#    fwd_sdfg.apply_transformations_repeated(TrivialMapRangeElimination)
#    SubgraphFusion.apply_to(fwd_sdfg, *fwd_sdfg.node(0).nodes())
#
#dace_func.append_post_onnx_hook("auto_optimize",
#    lambda dace_module: utils.auto_optimize(dace_module.dace_onnx_model.sdfg,
#                                            True,
#                                            apply_strict=True))
#dace_func.append_post_onnx_hook("fuse_sg", fuse_sg)
#dace_func.append_post_onnx_hook("fuse_tasklets", lambda x:\
#        x.dace_onnx_model.sdfg.apply_transformations_repeated(transformation.TaskletFusion, validate=True))
#def vectorize(fwd, bwd):
#    fwd.apply_transformations(Vectorization, validate=True)
#    bwd.apply_transformations(Vectorization, validate=True)
#
#dace_func.append_post_autodiff_hook("vectorize", vectorize)
##dace_func.append_post_autodiff_hook("view",
##    lambda f, b: b.view())






# In[3]:


# setup input data
# shapes taken from the first YOLOv4 activation
size = [8, 32, 224, 224]
pt_inputs = torch.rand(*size).cuda()
dace_inputs = torch.clone(pt_inputs)
pt_inputs.requires_grad = True
dace_inputs.requires_grad = True
dy = torch.rand(*size).cuda()


pt_jit = torch.jit.trace(pt_no_module, [pt_inputs])
# In[4]:


# simple correctness check
pt_output = pt_func(pt_inputs)
dace_output = dace_func(dace_inputs)
torch.allclose(pt_output, dace_output)

dace_output.backward(dy)
pt_output.backward(dy)

torch.allclose(dace_inputs.grad, pt_inputs.grad)




# In[5]:

import time
time.sleep(1)

for i in range(1000):
    dace_output.grad = None
    dace_output = dace_func(dace_inputs)
    dace_output.backward(dy)
    
#for i in range(10):
#    pt_output.grad = None
#    pt_output = pt_jit(pt_inputs)
#    pt_output.backward(dy)
#
#for i in range(1000):
#    pt_output.grad = None
#    pt_output = pt_jit(pt_inputs)
#    pt_output.backward(dy)



