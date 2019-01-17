from __future__ import absolute_import
import scipy.sparse as osp
from . import sparse_wrapper as asp
from .sparse_boxes import ArrayBox
from autograd.tracer import primitive
from autograd.core import defvjp

# get numpy wrapper
from autograd.numpy import numpy_wrapper as anp

# ----- Binary ufuncs -----

defvjp(asp.add,         lambda g, ans, x, y : unbroadcast(x, g),
                        lambda g, ans, x, y : unbroadcast(y, g))
defvjp(asp.multiply,    lambda g, ans, x, y : unbroadcast(x, y * g),
                        lambda g, ans, x, y : unbroadcast(y, x * g))
defvjp(asp.subtract,    lambda g, ans, x, y : unbroadcast(x, g),
                        lambda g, ans, x, y : unbroadcast(y, -g))
defvjp(asp.divide,      lambda g, ans, x, y : unbroadcast(x,   g / y),
                        lambda g, ans, x, y : unbroadcast(y, - g * x / y**2))
defvjp(asp.true_divide, lambda g, ans, x, y : unbroadcast(x,   g / y),
                        lambda g, ans, x, y : unbroadcast(y, - g * x / y**2))
defvjp(asp.power,
    lambda g, ans, x, y: unbroadcast(x, g * y * x ** anp.where(y, y - 1, 1.)),
    lambda g, ans, x, y: unbroadcast(y, g * anp.log(replace_zero(x, 1.)) * x ** y))

def replace_zero(x, val):
    return anp.where(x, x, val)

def unbroadcast(target, g, broadcast_idx=0):
    while anp.ndim(g) > anp.ndim(target):
        g = anp.sum(g, axis=broadcast_idx)
    for axis, size in enumerate(anp.shape(target)):
        if size == 1:
            g = anp.sum(g, axis=axis, keepdims=True)
    if anp.iscomplexobj(g) and not anp.iscomplex(target):
        g = anp.real(g)
    return g

# ----- Simple grads -----

defvjp(anp.negative, lambda g, ans, x: -g)
defvjp(anp.exp,    lambda g, ans, x: ans * g)
defvjp(anp.log,    lambda g, ans, x: g / x)
defvjp(anp.tanh,   lambda g, ans, x: g / anp.cosh(x) **2)
defvjp(anp.sinh,   lambda g, ans, x: g * anp.cosh(x))
defvjp(anp.cosh,   lambda g, ans, x: g * anp.sinh(x))

defvjp(anp.where, None,
       lambda g, ans, c, x=None, y=None: anp.where(c, g, anp.zeros(g.shape)),
       lambda g, ans, c, x=None, y=None: anp.where(c, anp.zeros(g.shape), g))

defvjp(anp.reshape, lambda g, ans, x, shape, order=None:
       anp.reshape(g, anp.shape(x), order=order))

# ----- Dot grads -----

def _dot_vjp_0(g, ans, lhs, rhs):
  if max(anp.ndim(lhs), anp.ndim(rhs)) > 2:
    raise NotImplementedError("Current dot vjps only support ndim <= 2.")

  if anp.ndim(lhs) == 0:
    return anp.sum(rhs * g)
  if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 1:
    return g * rhs
  if anp.ndim(lhs) == 2 and anp.ndim(rhs) == 1:
    return g[:, None] * rhs
  if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 2:
    return anp.dot(rhs, g)
  return anp.dot(g, rhs.T)

def _dot_vjp_1(g, ans, lhs, rhs):
  if max(anp.ndim(lhs), anp.ndim(rhs)) > 2:
    raise NotImplementedError("Current dot vjps only support ndim <= 2.")

  if anp.ndim(rhs) == 0:
    return anp.sum(lhs * g)
  if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 1:
    return g * lhs
  if anp.ndim(lhs) == 2 and anp.ndim(rhs) == 1:
    return anp.dot(g, lhs)
  if anp.ndim(lhs) == 1 and anp.ndim(rhs) == 2:
    return lhs[:, None] * g
  return anp.dot(lhs.T, g)

defvjp(anp.dot, _dot_vjp_0, _dot_vjp_1)
