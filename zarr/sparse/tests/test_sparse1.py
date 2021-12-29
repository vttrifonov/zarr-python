#%%
import unittest
import sparse
import numpy as np
from zarr.sparse.sparse1 import array, full
import pytest

sparse_array = array
sparse_full = full

def np_array(x, shape, dtype):
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
            #coords must be 2d
            sparse_array([1], coords=[1])

        with pytest.raises(ValueError):
            #too many coords
            sparse_array([1], coords=[[1, 2]])

        with pytest.raises(ValueError):
            #coord too large
            sparse_array([1], coords=[[3]], shape=(2,))

        with pytest.raises(ValueError):
            #negative coord
            sparse_array([1], coords=[[-1]])

        with pytest.raises(ValueError):
            #float coord
            sparse_array([1], coords=[[1.1]])

        def f(x1, x2):
            np.testing.assert_array_equal(
                np.array(x1), x2
            )

        f(sparse_array([]), np_array([0], (0,), np.float64))

        f(sparse_array(np.empty((0,2))), np_array([], (0,2), np.float64))

        f(sparse_array([], coords=np.empty((0,0), dtype=np.int64)), np_array(0, (), np.float64))

        f(sparse_array([], coords=np.empty((3,0), dtype=np.int64)), np_array([], (0,0,0), np.float64))

        f(sparse_full((), 1), np_array(1, (), np.int64))

        f(sparse_full((0,), 0), np_array([], (0,), np.int64))

        f(sparse_array(1), np_array([1], (), np.int64))

        f(sparse_array(0), np_array([0], (), np.int64))

        f(sparse_array([[1,0], [0,2]]), np_array([1,0,0,2], (2,2), np.int64))

        f(sparse_array([1,2,3,4], shape=(2,2)), np_array([1,2,3,4], (2,2), np.int64))

        f(sparse_array([1]), np_array([1], (1,), np.int64))

        f(sparse_array([1], coords=[[0]]), np_array([1], (1,), np.int64))

        f(sparse_full((2,2), 1), np_array([1]*4, (2,2), np.int64))

        f(sparse_full((2,2), [1,2,3]), np_array([[1,2,3]]*4, (2,2), object))

        f(sparse_full((2,2), dtype=str), np_array(['']*4, (2,2), str))

        f(sparse_full((2,2), dtype=object), np_array([0]*4, (2,2), object))

        f(sparse_array([1], coords=[[0]], dtype=str), np_array(['1'], (1,), str))

        f(sparse_array([1,2], coords=[[0,0], [0,1]]), np_array([1,2], (1,2), np.int64))

        f(sparse_array([1,2], coords=[[0,1], [0,0]]), np_array([1,2], (2,1), np.int64))

        f(sparse_array([1,2], coords=[[0,1], [0,1]]), np_array([1,0,0,2], (2,2), np.int64))

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
            sparse_array([], coords=np.empty((0,0), dtype=np.int64)),
            sparse_array([], coords=np.empty((3,0), dtype=np.int64)),
            sparse_array(1),
            sparse_array([1], coords=[[0]]),
            sparse_array([1,2], coords=[[0,0], [0,1]]),
            sparse_array([1,2], coords=[[0,1], [0,0]]),
            sparse_array([1,2], coords=[[0,1], [0,1]])
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
        x = [
            sparse_array([], coords=np.empty((0,0), dtype=np.int64)),
            sparse_array([], coords=np.empty((3,0), dtype=np.int64)),
            sparse_array(1),
            sparse_array([1], coords=[[0]]),
            sparse_array([1,2], coords=[[0,0], [0,1]]),
            sparse_array([1,2], coords=[[0,1], [0,0]]),
            sparse_array([1,2], coords=[[0,1], [0,1]])
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

        def f(xi, si):
            x3 = x[xi]
            s3 = s[si]

            try:
                x1 = np.array(x3.broadcast_to(s3))
            except Exception:
                x1 = 'err'

            try:
                x2 = np.broadcast_to(np.array(x3), s3)
            except Exception:
                x2 = 'err'

            np.testing.assert_array_equal(
                x1, x2,
                err_msg=f'x: {xi}, s: {si}'
            )

        for xi in range(len(x)):
            for si in range(len(s)):
                f(xi, si)

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
            sparse_array([0,1,2], coords=[[0,1,2],[0,1,2]])
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
        x1 = sparse_full((3,3), dtype=np.float64)
        x2 = np.array(x1)
        def f(i, v):
            x1[i] = v
            x2[i] = np.array(v)
            np.testing.assert_array_equal(np.array(x1), x2)
        f(np.s_[:1,:1], sparse_array(1))
        f(np.s_[0, 1:], sparse_array(2))
        f(np.s_[[0,2], 2], sparse_array(3))
        f(np.s_[[1], [False, True, False]], sparse_array(4))
        f(np.s_[:4:2, :4:2], sparse_array(5))
        f(np.s_[:4:2, :4:2], sparse_array([[1,2], [3,4]]))
        f(np.s_[:4:2, :4:2], sparse_array([[1,2]]))
        f(np.s_[:4:2, :4:2], sparse_array([[1],[2]]))
        f(np.s_[:2, 0], sparse_array([1,2]))
        f(np.s_[:2, :2], sparse_array([[0,1],[2,0]]))
        f(np.s_[[0,1,1,2], 1], sparse_array([1,2,3,4]))
        f(np.s_[[1,1,1,1], 1], sparse_array([1,2,3,4]))
        f(np.s_[[2,1,1,0], 1], sparse_array([1,2,3,4]))
        f(np.s_[[2,1,1,0], 1:], sparse_array([[1,2],[3,4],[5,6],[7,8]]))
        f(np.s_[[2,1,1,0], 1:], sparse_array([[1,2]]))
        f(np.s_[[2,1,1,0], 1:], sparse_array(1))
        f(np.s_[[2,1,1,0], 1:], sparse_array([0]))


# %%
