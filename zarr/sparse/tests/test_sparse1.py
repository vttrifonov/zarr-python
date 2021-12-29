#%%
import unittest
import sparse
import numpy as np
from zarr.sparse.sparse1 import Sparse
import pytest

def array(x, shape, dtype):
    x1 = np.empty(shape, dtype=dtype)
    x2 = x1.ravel()
    x2[...] = x
    return x1

#%%
class TestArray(unittest.TestCase):
    def test_conv(self):
        from zarr.sparse.sparse1 import (
            _conv_list, _conv_slice, _conv_single
        ) 

        c = _conv_list(np.array([]))
        x1 = c.fwd(np.array([1, 1, 0, 3, 5, -2]))
        x2 = np.array([-1, -1, -1, -1, -1, -1])
        assert np.all(x1==x2)

        x2 = c.fwd1([])
        assert x2==[]

        x1 = c.rev(np.array([-1,1]))
        x2 = np.array([-1,-1])
        assert np.all(x1==x2)

        c = _conv_list(np.array([0, 1, 3]))
        x1 = c.fwd(np.array([1, 1, 0, 3, 5, -2]))
        x2 = np.array([ 1,  1,  0,  2, -1, -1])
        assert np.all(x1==x2)

        x1 = c.rev(np.array([0, 2, 1, 4, -1, 5]))
        x2 = np.array([ 0,  3,  1, -1, -1, -1])
        assert np.all(x1==x2)

        x1 = c.rev(np.array([0, 2, 1, 4, -1, 5]))
        x2 = np.array([ 0,  3,  1, -1, -1, -1])
        assert np.all(x1==x2)

        c = _conv_list(np.array([1, 0, 1, 3, 0]))
        x1 = c.fwd(np.array([1, 1, 0, 3, 5, -2]))
        x2 = np.array([ 1,  1,  0,  2, -1, -1])
        assert np.all(x1==x2)

        x1 = x1[x1>=0]
        x1 = c.fwd1(x1)
        x2 = [[0,2], [0,2], [1,4], [3]]
        x2 = [np.array(x) for x in x2]
        x3 = [np.all(x==y) for x, y in zip(x1, x2)]
        assert all(x3)

        x1 = c.rev(np.array([0, 2, 1, 4, -1, 5]))
        x2 = np.array([-1,  1, -1,  0, -1, -1])
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

        def f(x1, x2):
            np.testing.assert_array_equal(
                np.array(x1), x2
            )

        f(Sparse(data=[]), array([0], (0,), np.float64))

        f(Sparse(data=np.empty((0,2))), array([], (0,2), np.float64))

        f(Sparse(data=[], coords=np.empty((0,0), dtype=np.int64)), array(0, (), np.float64))

        f(Sparse(data=[], coords=np.empty((3,0), dtype=np.int64)), array([], (0,0,0), np.float64))

        f(Sparse(fill_value=1), array(1, (), np.int64))

        f(Sparse(fill_value=0, shape=(0,)), array([], (0,), np.int64))

        f(Sparse(data=1), array([1], (), np.int64))

        f(Sparse(data=0), array([0], (), np.int64))

        f(Sparse(data=[[1,0], [0,2]]), array([1,0,0,2], (2,2), np.int64))

        f(Sparse(data=[1,2,3,4], shape=(2,2)), array([1,2,3,4], (2,2), np.int64))

        f(Sparse(data=[1]), array([1], (1,), np.int64))

        f(Sparse(data=[1], coords=[[0]]), array([1], (1,), np.int64))

        f(Sparse(fill_value=1, shape=(2,2)), array([1]*4, (2,2), np.int64))

        f(Sparse(fill_value=[1,2,3], shape=(2,2)), array([[1,2,3]]*4, (2,2), object))

        f(Sparse(dtype=str, shape=(2,2)), array(['']*4, (2,2), str))

        f(Sparse(dtype=object, shape=(2,2)), array([0]*4, (2,2), object))

        f(Sparse(data=[1], coords=[[0]], dtype=str), array(['1'], (1,), str))

        f(Sparse(data=[1,2], coords=[[0,0], [0,1]]), array([1,2], (1,2), np.int64))

        f(Sparse(data=[1,2], coords=[[0,1], [0,0]]), array([1,2], (2,1), np.int64))

        f(Sparse(data=[1,2], coords=[[0,1], [0,1]]), array([1,0,0,2], (2,2), np.int64))

    def test_reshape(self):
        def f(x, s, o):
            try:
                x1 = np.array(x.reshape(s, order=o))
            except Exception:
                x1 = 'err'

            try:
                x2 = np.array(x).reshape(s, order=o)
            except Exception:
                x2 = 'err'

            np.testing.assert_array_equal(
                x1, x2,
                err_msg = f'x: {np.array(x)}, s: {s}, o: {o}'
            )

        x = [
            Sparse(data=[], coords=np.empty((0,0), dtype=np.int64)),
            Sparse(data=[], coords=np.empty((3,0), dtype=np.int64)),
            Sparse(data=1),
            Sparse(data=[1], coords=[[0]]),
            Sparse(data=[1,2], coords=[[0,0], [0,1]]),
            Sparse(data=[1,2], coords=[[0,1], [0,0]]),
            Sparse(data=[1,2], coords=[[0,1], [0,1]])
        ]

        s = [
            (),
            (0,),
            (1,),            
            (2,),
            (1, 0),
            (0, 1),
            (2, 0),
            (0, 2),
            (1, 2),
            (2, 1),
            (2, 2),
            (1, 1, 2),
            (2, 1, 1),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2)
        ]

        o = ['F', 'C']

        for x1 in x:
            for s1 in s:
                for o1 in o:
                    f(x1, s1, o1)
        
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

        x = [
            Sparse(data=[], coords=np.empty((0,0), dtype=np.int64)),
            Sparse(data=[], coords=np.empty((3,0), dtype=np.int64)),
            Sparse(data=1),
            Sparse(data=[1], coords=[[0]]),
            Sparse(data=[1,2], coords=[[0,0], [0,1]]),
            Sparse(data=[1,2], coords=[[0,1], [0,0]]),
            Sparse(data=[1,2], coords=[[0,1], [0,1]])
        ]

        s = [
            (),
            (0,),
            (1,),            
            (2,),
            (1, 0),
            (0, 1),
            (2, 0),
            (0, 2),
            (1, 2),
            (2, 1),
            (2, 2),
            (1, 1, 2),
            (2, 1, 1),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2)
        ]

        for x1 in x:
            for s1 in s:        
                f(x1, s1)

    def test_getitem(self):
        def f(x, s):
            try:
                x3 = np.array(x[s])
            except Exception:
                x3 = 'err'

            try:
                x4 = np.array(x)[s]
            except Exception:
                x4 = 'err'

            np.testing.assert_array_equal(
                x3, x4,
                err_msg = f'x: {np.array(x)}, s: {s}'
            )

        x = [
            Sparse(data=[0,1,2], coords=[[0,1,2],[0,1,2]])
        ]

        s = [
            np.s_[0,0],
            np.s_[1,[1,2]],
            np.s_[1,:2],
            np.s_[[True, False, True], 1],
            np.s_[[1,2], :3],
            np.s_[[True, False, True], :3],
            np.s_[1:, :1],
            np.s_[3,:],
            np.s_[:3,:3],
            np.s_[:3,4:],
            np.s_[[],1],
            np.s_[[1,1],1],
            np.s_[[1,0,1,2],1],
            np.s_[[1,0,1,2],1:]
        ]

        for x1 in x:
            for s1 in s:
                f(x1, s1)

    def test_setitem(self):
        x1 = Sparse(shape=(3,3), dtype=np.float64)
        x2 = np.array(x1)
        def f(i, v):
            x1[i] = v
            x2[i] = np.array(v)
            np.testing.assert_array_equal(np.array(x1), x2)
        f(np.s_[:1,:1], Sparse(1))
        f(np.s_[0, 1:], Sparse(2))
        f(np.s_[[0,2], 2], Sparse(3))
        f(np.s_[[1], [False, True, False]], Sparse(4))
        f(np.s_[:4:2, :4:2], Sparse(5))
        f(np.s_[:4:2, :4:2], Sparse([[1,2], [3,4]]))
        f(np.s_[:4:2, :4:2], Sparse([[1,2]]))
        f(np.s_[:4:2, :4:2], Sparse([[1],[2]]))
        f(np.s_[:2, 0], Sparse([1,2]))
        f(np.s_[:2, :2], Sparse([[0,1],[2,0]]))
        f(np.s_[[0,1,1,2], 1], Sparse([1,2,3,4]))
        f(np.s_[[1,1,1,1], 1], Sparse([1,2,3,4]))
        f(np.s_[[2,1,1,0], 1], Sparse([1,2,3,4]))
        f(np.s_[[2,1,1,0], 1:], Sparse([[1,2],[3,4],[5,6],[7,8]]))
        f(np.s_[[2,1,1,0], 1:], Sparse([[1,2]]))
        f(np.s_[[2,1,1,0], 1:], Sparse(1))
        f(np.s_[[2,1,1,0], 1:], Sparse([0]))


# %%
