from __future__ import absolute_import
import types
from autograd.tracer import primitive, notrace_primitive
import scipy.sparse as _sp

# ----- Non-differentiable functions -----

nograd_functions = [_sp.shape]

def wrap_intdtype(cls):
    class IntdtypeSubclass(cls):
        __new__ = notrace_primitive(cls.__new__)
    return IntdtypeSubclass

# need to add the sparse matrix types?
def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int, _np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in old.items():
        if obj in nograd_functions:
            new[name] = notrace_primitive(obj)
        elif type(obj) in function_types:
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj

wrap_namespace(_np.__dict__, globals())
