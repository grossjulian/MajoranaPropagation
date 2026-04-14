import warnings
from functools import partial
from typing import Any, Callable, Optional

import jax
from jax import numpy as jnp
import netket as nk
import flax
from flax import serialization
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket import nn as nknn
from netket.hilbert import AbstractHilbert
from netket.utils import maybe_wrap_module, wrap_afun, wrap_to_support_scalar
from netket.utils.types import PyTree, SeedT, NNInitFunc, Array, DType, Union
from netket.optimizer import LinearOperator
from netket.optimizer.qgt import QGTAuto

from netket.vqs.base import VariationalState
# from src.utils import safe_log

@jax.jit 
def safe_log(x):
    is_finite = x != 0
    x = jnp.where(is_finite, x, 1)
    inf_add = jnp.where(is_finite, 0, -jnp.inf)
    if not jnp.iscomplexobj(x): # save an addition or casting
        x = x + 0j
    return jnp.log(x+0j) + inf_add    

class VariationalVectorState(VariationalState):
    def __init__(
        self,
        hilbert: AbstractHilbert,
        vector_array: Union[Array, VariationalState],
        *,
        dtype: DType = None,
        normalize: bool = False,
    ):
        super().__init__(hilbert)
        if isinstance(vector_array, VariationalState):
            vector_array = vector_array.to_array()
        self._vector = jnp.array(vector_array).ravel()
        if dtype is not None:
            self._vector = self._vector.astype(dtype)
        if normalize:
            self.normalize()
        if not self.hilbert.is_indexable:
            raise Exception("Cannot create state if hilbert space is not indexable.")
        if self.hilbert.n_states != self._vector.size:
            raise Exception("Size of vector does not correspond to number of states in the hilbert space.")
        
    def to_complex(self):
        if not nkjax.is_complex_dtype(self._vector.dtype):
            dtype = nkjax.dtype_complex(self._vector.dtype)
            self._vector = self._vector.astype(dtype)
            
    def normalize(self):
        self._vector = _normalize(self._vector)
        
    def to_fullsum(self):
        def _logstate_init(key, shape, dtype):
            assert self.vector.shape == shape
            return safe_log(self.vector).astype(dtype)
        ma = nk.models.LogStateVector(self.hilbert, param_dtype=jnp.complex128, logstate_init=_logstate_init)
        vs_fs = nk.vqs.FullSumState(self.hilbert, ma)
        # print("fullsum = ", vs_fs.to_array())
        return vs_fs
        
    def to_array(self, normalize=True):
        if normalize:
            return self.normalized_vector
        else:
            return self.vector
        
    @property
    def vector(self):
        return self._vector
    
    @vector.setter
    def vector(self, other):
        self._normalized = False
        other = jnp.asarray(other).ravel()
        if other.shape != self._vector.shape:
            raise ValueError("Provided vector does not match internal vector in shape.")
        # make sure we never down cast
        _dtype = jnp.promote_types(self._vector.dtype, other.dtype)
        self._vector = other.astype(_dtype).copy()
        
    @property
    def normalized_vector(self):
        return _normalize(self._vector)
    
    @property
    def parameters(self):
        return {"vector" : self.vector}
    
    @parameters.setter
    def parameters(self, other):
        self.vector = other["vector"]
        
    def norm(self):
        return _compute_norm(self.vector)
    
@jax.jit
def _normalize(v):
    return v / jnp.linalg.norm(v)

@jax.jit
def _compute_norm(v):
    return (v.conj() * v).sum()

