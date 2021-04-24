import logging

from daceml.onnx.shape_inference.symbolic_shape_infer import SymbolicShapeInference, get_opset


def infer_shapes(onnx_model):
    if logging.root.level <= logging.DEBUG:
        verbose = 1
    else:
        verbose = 0

    return SymbolicShapeInference.infer_shapes(onnx_model, verbose=verbose)
