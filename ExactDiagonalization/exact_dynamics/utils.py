from functools import cache, lru_cache
import scipy as sp
import numpy as np

def to_dense(O, cache=False):
    return to_matrix(O, sparse=False, cache=cache)

def to_sparse(O, cache=False, scipy=False):
    mat = to_matrix(O, sparse=True, cache=cache)
    if scipy:
        mat = mat.to_bcoo()
        data = np.array(mat.data)
        indixs = np.array(mat.indices[:,0])
        indptr = np.array(mat.indices[:,1])
        mat = sp.sparse.csc_matrix((data, (indixs, indptr)), shape=mat.shape)
    return mat

def to_matrix(O, sparse=True, cache=False):
    """
    Converts to sparse but also cache the sparsificated result to speed up.
    """
    if cache:
        return _to_matrix_cached(O, sparse=sparse)
    else:
        return _to_matrix_not_cached(O, sparse=sparse)

@lru_cache(10)
def _to_matrix_cached(O, sparse=True):
    """
    Converts to sparse but also cache the sparsificated result to speed up.
    """
    return _to_matrix_not_cached(O, sparse=sparse)

def _to_matrix_not_cached(O, sparse=True):
    """
    Converts to sparse but also cache the sparsificated result to speed up.
    """
    if sparse:
        return O.to_sparse()
    else:
        return O.to_dense()