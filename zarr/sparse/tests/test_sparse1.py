#%%
import unittest
import sparse
import numpy as np
from zarr.sparse.sparse1 import Sparse
import pytest


#%%
class TestArray(unittest.TestCase):
    def test_conv(self):
        from zarr.sparse.sparse1 import (
            _conv_list, _conv_slice, _conv_single
        ) 

        c = _conv_list(np.array([0, 1, 3]))
        x1 = c.fwd(np.array([1, 1, 0, 3, 5, -2]))
        x2 = np.array([ 1,  1,  0,  2, -1, -1])
        assert np.all(x1==x2)

        x1 = c.rev(np.array([0, 2, 1, 4, -1, 5]))
        x2 = np.array([ 0,  3,  1, -1, -1, -1])
        assert np.all(x1==x2)

        c = _conv_slice(slice(10, 20, 2))
        x1 = c.fwd(np.array([10, 12, 13, 14, 22, 8]))
        x2 = np.array([ 0,  1, -1,  2, -1, -1])
        assert np.all(x1==x2)
        
        x1 = c.rev(np.array([0, 1, 2, 5, 4, -2, 6]))
        x2 = np.array([10, 12, 14, -1, 18, -1, -1])
        assert np.all(x1==x2)

        c = _conv_single(2)
        x1 = c.fwd(np.array([2, 3, 2, 4, 2]))
        x2 = np.array([ 0, -1,  0, -1,  0])
        assert np.all(x1==x2)

        x1 = c.rev(np.array([0, 1, -1, 2, 0, 2]))
        x2 = np.array([ 2, -1, -1, -1,  2, -1])
        assert np.all(x1==x2)

    def test_creation(self):
        with pytest.raises(ValueError):
            #missing dtype
            Sparse()

        with pytest.raises(ValueError):
            #coords must be 2d
            Sparse(data=[1], coords=[1])

        with pytest.raises(ValueError):
            #too many coords
            Sparse(data=[1], coords=[[1, 2]])

        with pytest.raises(ValueError):
            #coord too large
            Sparse(data=[1], coords=[[3]], shape=(2,))

        with pytest.raises(ValueError):
            #negative coord
            Sparse(data=[1], coords=[[-1]])

        with pytest.raises(ValueError):
            #float coord
            Sparse(data=[1], coords=[[1.1]])

        Sparse(fill_value=1)

        Sparse(data=[1])

        Sparse(data=[1], coords=[[0]])

        Sparse(fill_value=1, shape=(2,2))

        Sparse(fill_value=[1,2,3], shape=(2,2))

        Sparse(dtype=str, shape=(2,2))

        Sparse(dtype=object, shape=(2,2))

        Sparse(data=[1], coords=[[0]], dtype=str)

    def test_broadcasting(self):
        def f(x, s):
            try:
                x1 = np.array(x.broadcast_to(s))
            except Exception:
                x1 = 'err'

            try:
                x2 = np.broadcast_to(np.array(x), s)
            except Exception:
                x2 = 'err'

            np.testing.assert_array_equal(x1, x2)

        x = Sparse(data=[0], coords=[[0]])
        f(x, (2,))
        f(x, (2,2))
        f(x, (2,2,2))
        f(x, (0,))

        x = Sparse(data=[0,1], coords=[[0,1]])
        f(x, (0,))
        f(x, (1,))
        f(x, (2,))
        f(x, (1,2))
        f(x, (2,1))
        f(x, (2,2))
        f(x, (2,0))
        f(x, (0,2))
        f(x, (1,0))
        f(x, (0,1))

        x = Sparse(data=[0,1], coords=[[0, 1],[0, 0]])
        f(x, (0,))
        f(x, (1,))
        f(x, (2,))
        f(x, (1,2))
        f(x, (2,1))
        f(x, (2,2))
        f(x, (0,2))
        f(x, (2,0))
        f(x, (1,0))
        f(x, (0,1))

    def test_indexing(self):
        x1 = Sparse(shape=(3,3), dtype=np.float64)
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


# %%
