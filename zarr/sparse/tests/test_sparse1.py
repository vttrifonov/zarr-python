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

def assert_array_equal(x, y, err_msg=None):
    if hasattr(x, 'todense'):
        x = x.todense()

    if hasattr(y, 'todense'):
        y = y.todense()

    np.testing.assert_array_equal(x, y, err_msg=err_msg)

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
                x1.todense(), x2
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
            (2, 2, 2),
            0, 1, 2, -1, -2,
            (-2, 2), (1.2, -1),
            (-1, 2), 
            (-1, -1),
            (-1, 1)
        ]

        o = ['F', 'C']

        def f(xi, si, oi):
            x3 = x[xi]
            s3 = s[si]
            o3 = o[oi]            

            try:
                x1 = x3.reshape(s3, order=o3).todense()
            except Exception:
                x1 = 'err'

            try:
                x2 = x3.todense().reshape(s3, order=o3)
            except Exception:
                x2 = 'err'

            np.testing.assert_array_equal(
                x1, x2,
                err_msg = f'x: {xi}, s: {si}, o: {oi}'
            )

        f(0, 20, 0)

        for xi in range(len(x)):
            for si in range(len(s)):
                for oi in range(len(o)):
                    f(xi, si, oi)
        
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
                x1 = x3.broadcast_to(s3).todense()
            except Exception:
                x1 = 'err'

            try:
                x2 = np.broadcast_to(x3.todense(), s3)
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
        x = [
            sparse_array(3),
            sparse_array([1,2,3]),
            sparse_array([0,1,2], coords=[[0,1,2],[0,1,2]])
        ]

        s = [
            np.s_[...],
            np.s_[()],
            np.s_[-1],
            np.s_[1],
            np.s_[-1,-1],
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

        def f(xi, si):
            x1 = x[xi]
            s1 = s[si]
            try:
                x3 = x1[s1]
                if hasattr(x3, 'todense'):
                    x3 = x3.todense()
            except Exception:
                x3 = 'err'

            try:
                x4 = x1.todense()[s1]
            except Exception:
                x4 = 'err'

            np.testing.assert_array_equal(
                x3, x4,
                err_msg = f'x: {xi}, s: {si}'
            )

        for xi in range(len(x)):
            for si in range(len(s)):
                f(xi, si)

    def test_setitem(self):
        x1 = sparse_array(3)
        x1[()] = 4
        np.testing.assert_array_equal(x1.todense(), np.array(4))

        x1 = sparse_full((3,3), dtype=np.float64)
        x2 = x1.todense()
        def f(i, v):
            x1[i] = v
            if hasattr(v, 'todense'):
                v = v.todense()
            x2[i] = v
            np.testing.assert_array_equal(x1.todense(), x2)
        f(np.s_[:1,:1], 1)
        f(np.s_[0, 1:], 1)
        f(np.s_[[0,2], 2], 3)
        f(np.s_[[1], [False, True, False]], 3)
        f(np.s_[:4:2, :4:2], 5)
        f(np.s_[:4:2, :4:2], [[1,2], [3,4]])
        f(np.s_[:4:2, :4:2], [[1,2]])
        f(np.s_[:4:2, :4:2], [[1],[2]])
        f(np.s_[:2, 0], [1,2])
        f(np.s_[:2, :2], [[0,1],[2,0]])
        f(np.s_[[0,1,1,2], 1], [1,2,3,4])
        f(np.s_[[1,1,1,1], 1], [1,2,3,4])
        f(np.s_[[2,1,1,0], 1], [1,2,3,4])
        f(np.s_[[2,1,1,0], 1:], [[1,2],[3,4],[5,6],[7,8]])
        f(np.s_[[2,1,1,0], 1:], [[1,2]])
        f(np.s_[[2,1,1,0], 1:], 1)
        f(np.s_[[2,1,1,0], 1:], [0])
        f(np.s_[:2, :2], array([1,0], fill_value=1))

    def test_subarray(self):
        x1 = np.array([[1,2],[3,4]])
        x2 = sparse_array(x1, dtype='(2,2)i4')
        assert_array_equal(x1, x2)

        x3 = x2.broadcast_to((2,2))
        x4 = np.broadcast_to(x1, (2,2,2,2)).copy()
        assert_array_equal(x3, x4)

        assert_array_equal(x3[0,0], x4[0,0])
        assert_array_equal(x3[:2,0], x4[:2,0])
        assert_array_equal(x3[0,1:], x4[0,1:])
        assert_array_equal(x3[[0,1],1:], x4[[0,1],1:])

        x3[0,0] = [[0,0],[0,0]]
        assert x3.data.shape == (3,2,2)

        x4[0,0,:,:] = [[0,0],[0,0]]
        assert_array_equal(x3,x4)

        x5 = x3.reshape((4,))
        x6 = x4.reshape((4,2,2))
        assert_array_equal(x5, x6)

        x5[2:4] = [[0,0],[0,0]]    
        assert x5.data.shape == (1,2,2)

        x6[2:4,:,:] = [[0,0],[0,0]]    
        assert_array_equal(x5, x6)

        x1 = np.zeros((2,2,2,2), dtype='i4')
        x2 = sparse_full(shape=(2,2), dtype='(2,2)i4')
        assert_array_equal(x1, x2)

        x1 = np.zeros((0,2,2), dtype='i4')
        x2 = sparse_full(shape=(0,), dtype='(2,2)i4')
        assert_array_equal(x1, x2)

        with pytest.raises(ValueError):
            x1 = np.array([[1,2],[3,4]])
            x2 = sparse_array(x1)
            x3 = x2.astype(dtype='(2,2)i4')

    def test_structured(self):
        x = np.array([
                (b'a', 1),
                (b'b', 2)
            ], 
            dtype=[('foo', 'S3'), ('bar', 'i4')]
        )
        x = sparse_array(x)
        x1 = x.todense()

        assert x[0]==x1[0]

        x[1] = (b'c', 3)
        x1[1] = (b'c', 3)
        assert x[1]==x1[1]

        x[0] = (b'', 0)
        np.testing.assert_array_equal(
            x.data, np.array([(b'c', 3)], dtype=x.dtype)
        )

        np.testing.assert_array_equal(
            x['foo'].todense(), 
            x.todense()['foo']
        )

        x = np.array([
                (0, ((0, 1, 2), (1, 2, 3)), b'aaa'),
                (1, ((1, 2, 3), (2, 3, 4)), b'bbb'),
                (2, ((2, 3, 4), (3, 4, 5)), b'ccc')
            ],
            dtype=[
                ('foo', 'i8'), 
                ('bar', '(2, 3)f4'), 
                ('baz', 'S3')
            ]
        )
        x = sparse_array(x)
        x1 = x.todense()

        assert x[0]==x1[0]

        for f in x.dtype.names:
            np.testing.assert_array_equal(
                x[f].todense(), x1[f],
                err_msg=f'f: {f}'
            )

    def test_object(self):
        x1 = sparse_full((3,), dtype=object)
        x1[0] = 1
        assert x1[0]==1
        x1[1] = 'a'
        assert x1[1]=='a'
        x1[2] = [1,'a']
        assert x1[2]==[1,'a']
        x1[0] = np.array([1,2])
        np.testing.assert_array_equal(x1[0], np.array([1,2]))
        
# %%
