#%%
import unittest
import sparse
import numpy as np
from zarr.sparse.sparse import Array, Sparse, zeros, full


#%%
class TestArray(unittest.TestCase):
    def test_indexing(self):
        x1 = zeros((3,3), dtype=np.float64)
        x2 = np.zeros((3,3), dtype=np.float64)
        def f(i, v):
            x1[i] = v
            x2[i] = v
            assert np.all(x1.x.todense()==x2)
        f(np.s_[:1,:1], 1)
        f(np.s_[0, 1:], 2)
        f(np.s_[[0,2], 2], 3)
        f(np.s_[[1], [False, True, False]], 4)
        f(np.s_[:4:2, :4:2], 5)
        f(np.s_[:4:2, :4:2], [[1,2], [3,4]])
        f(np.s_[:4:2, :4:2], [[1,2]])
        f(np.s_[:4:2, :4:2], [[1],[2]])
        f(np.s_[:2, 0], [1,2])
        
