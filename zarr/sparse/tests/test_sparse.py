#%%
import unittest
import sparse
import numpy as np
from zarr.sparse.sparse import Array, Sparse, zeros, full
from zarr.sparse.sparse import Sparse1
import pytest


#%%
class TestArray(unittest.TestCase):
    def test_creation(self):
        with pytest.raises(ValueError):
            #missing dtype
            Sparse1()

        with pytest.raises(ValueError):
            #missing shape
            Sparse1(fill_value=1)

        with pytest.raises(ValueError):
            #missing coords and index
            Sparse1(data=[1])

        with pytest.raises(ValueError):
            #both coords and index provided
            Sparse1(data=[1], coords=[1], index=[1])

        with pytest.raises(ValueError):
            #coords must be 2d
            Sparse1(data=[1], coords=[1])

        with pytest.raises(ValueError):
            #index must be 1d
            Sparse1(data=[1], index=[[1,2]])

        with pytest.raises(ValueError):
            #too many coords
            Sparse1(data=[1], coords=[[1, 2]])

        with pytest.raises(ValueError):
            #too many indices
            Sparse1(data=[1], index=[1,2])

        with pytest.raises(ValueError):
            #no shape
            Sparse1(data=[1], index=[1])

        with pytest.raises(ValueError):
            #coord too large
            Sparse1(data=[1], coords=[[3]], shape=(2,))

        with pytest.raises(ValueError):
            #index too large
            Sparse1(data=[1], index=[100], shape=(2,))

        Sparse1(data=[1], coords=[[0]])

        Sparse1(fill_value=1, shape=(2,2))

        Sparse1(fill_value=[1,2,3], shape=(2,2))

        Sparse1(dtype=str, shape=(2,2))

        Sparse1(dtype=object, shape=(2,2))

        Sparse1(data=[1], coords=[[0]], dtype=str)

        x1 = Sparse1(shape=(3,3), dtype=np.float64)
        x2 = np.zeros((3,3), dtype=np.float64)

    def test_indexing1(self):
        x1 = Sparse1(shape=(3,3), dtype=np.float64)
        x2 = np.zeros((3,3), dtype=np.float64)
        def f(i, v):
            x1[i] = v
            x2[i] = v
            x3 = x1.normalize()
            assert np.all(np.array(x3)==x2)
        f(np.s_[:1,:1], 1)
        f(np.s_[0, 1:], 2)
        f(np.s_[[0,2], 2], 3)
        f(np.s_[[1], [False, True, False]], 4)
        f(np.s_[:4:2, :4:2], 5)
        f(np.s_[:4:2, :4:2], [[1,2], [3,4]])
        f(np.s_[:4:2, :4:2], [[1,2]])
        f(np.s_[:4:2, :4:2], [[1],[2]])
        f(np.s_[:2, 0], [1,2])

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
        

# %%
