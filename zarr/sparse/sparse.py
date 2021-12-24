import sparse 
import numpy as np
import io
from zarr.sparse.core import Array as _Array
import itertools as it

class Sparse1:
    def __init__(self, 
        data = None, coords = None, index = None, 
        fill_value = None, dtype = None,
        shape = None, 
        order = 'C',  normalized = False
    ):
        if order is None:
            raise ValueError('missing order')

        if data is not None:
            data = np.asanyarray(data)
            if data.ndim!=1:
                raise ValueError('data must be 1D')
            if coords is None and index is None:
                raise ValueError('must provide coords or index')

        if index is not None:
            if coords is not None:
                raise ValueError('either index or coords, not both')

            index = np.asanyarray(index, dtype=np.int64)

            if index.ndim!=1:
                raise ValueError('index must be 1D')
            
            if data is None:
                if len(index)>0:
                    ValueError('data and index must have same length')
            else:
                if len(index) != len(data):
                    raise ValueError('data and index must have same length')
        else:
            if coords is None:
                index = np.array([], dtype=np.int64)

        if coords is not None:
            coords = np.asanyarray(coords)     
            if coords.ndim!=2:
                raise ValueError('coords must be 2D')
            if data is None:
                if coords.shape[1]>0: 
                    raise ValueError('data and coords must have same length')
            else:
                if coords.shape[1] != len(data):
                    raise ValueError('data and coords must have same length')
                if np.any(coords<0):
                    raise ValueError('coords cannot be negative')

        if dtype is None:
            if data is not None:
                dtype = data.dtype
            else:
                if fill_value is None:
                    raise ValueError('cannot infer dtype')
                try:
                    dtype = np.full((), fill_value).dtype
                except ValueError:
                    dtype = object
                
        if fill_value is None:
            fill_value = np.zeros((), dtype=dtype)[()]
        else:
            if dtype!=object:
                fill_value = np.full((), fill_value)[()]
        
        if shape is None:
            if coords is None:
                raise ValueError('cannot infer shape')
            shape = np.max(coords, axis=1)+1
        shape = tuple(shape)

        if index is None:
            index = np.ravel_multi_index(coords, shape, order=order)
        else:
            #validate index
            _ = np.unravel_index(index, shape, order=order) 

        if data is None:
            data = np.array([], dtype=dtype)
        else:
            data = data.astype(dtype)

        normalized = np.full((), normalized, dtype=bool)[()]
        
        self._index = index        
        self._data = data
        self._shape = shape
        self._fill_value = fill_value
        self._order = order
        self._normalized = normalized

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def order(self):
        return self._order

    @property
    def coords(self):
        return np.unravel_index(
            self._index, self._shape, order=self._order
        )

    @property
    def data(self):
        return self._data

    @property
    def normalized(self):
        return self._normalized

    def normalize(self):
        index = self._index
        data = self._data
        index, _ = np.unique(index, return_index=True)        
        data = data[_]
        _ = np.where(data!=self._fill_value)[0]
        index, data = index[_], data[_]
        _ = np.argsort(index)
        index, data = index[_], data[_]
        return self.__class__(
            index = index, 
            data = data,
            fill_value = self._fill_value, 
            shape = self._shape, order = self._order, 
            normalized = True
        )

    def reshape(self, shape = None, order = None):
        order = order or self._order
        shape = shape or self._shape

        if shape == self._shape and order == self._order:
            return self

        if order != self.__order:
            coords = self.coords
            index = np.ravel_multi_index(
                coords, shape, order=order
            )
            normalized = False
        else:
            index = self._index
            normalized = self._normalized

        return self.__class__(
            index = index, 
            data = self._data,
            fill_value = self._fill_value, 
            shape = shape, order = order, 
            normalized = normalized
        )

    def astype(self, dtype):
        if dtype == self.dtype:
            return self

        data = self._data.astype(dtype)

        return self.__class__(
            index = self._index, 
            data = data,
            fill_value = self._fill_value, 
            shape = self._shape, order = self._order, 
            normalized = self._normalized
        )

    def ravel_key(self, k):
        if not isinstance(k, tuple):
            k = (k,)

        e = np.where([x==Ellipsis for x in k])[0]
        if len(e)>1:
            raise IndexError('too many ellipsis')            
        if len(e)==1:
            e = e[0]
            k = k[:e] + tuple([slice(None)]*(self.ndim-len(k)+1)) + k[(e+1):]

        def _to_list(i, x):
            if isinstance(x, slice):
                x = np.arange(*x.indices(i))
                return x, len(x)

            x = np.asanyarray(x)
            if x.ndim>1:
                raise IndexError('int, array-like of int, or bool')

            if x.ndim==0:
                return x.reshape(1), 0

            if x.dtype==bool:
                if x.shape[0]!=i:
                    raise IndexError('bools len mismatch')
                x = np.where(x)[0]
            else:
                x = np.array([i+y if y<0 else y for y in x])
                if any(y>=i or y<0 for y in x):
                    raise IndexError('out of bounds')
            return x, len(x)

        i = [_to_list(i, x) for i, x in zip(self._shape, k)]
        s = tuple(s for _, s in i if s>0)

        i = [x for x, _ in i]
        i = list(it.product(*i))
        i = list(np.asanyarray(i).T)
        i = np.ravel_multi_index(i, self._shape, order = self._order)
        return i, s

    def __getitem__(self, k):
        if not self._normalized:
            raise ValueError('not normalized')
        i, shape = self.ravel_key(k)  
        coords = (range(x) for x in shape)
        coords = list(it.product(*coords))
        coords = np.asanyarray(coords).T
        _ = np.searchsorted(self._index, i)
        _ = np.where(_<len(self._index))[0]
        i, coords = i[_], coords[:,_]
        _ = np.where(self._index[_]==i)[0]
        coords, data = coords[:,_], self._data[_]
        
        return self.__class__(
            coords = coords, 
            data = data,
            fill_value = self._fill_value, 
            shape = shape, order = self._order, 
            normalized = self._normalized
        )

    def __setitem__(self, k, data):
        index, shape = self.ravel_key(k)
        if hasattr(data, 'todense'):
            data = data.todense()
        else:
            data = np.asanyarray(data)
        data = data.astype(self.dtype)
        data = data.reshape(data.shape, order = self._order)
        data = np.broadcast_to(data, shape)
        data = data.ravel()
        self._index = np.r_[index, self._index]
        self._data = np.r_[data, self._data]
        self._normalized = False

    def __array__(self):
        a = np.full(np.prod(self._shape), self._fill_value)
        a[self._index] = self._data
        a = a.reshape(self._shape, order=self._order)
        return a


def tocoo(v, fill_value):
    if hasattr(v, 'tocoo'):
        v = v.tocoo()
    if isinstance(v, sparse.COO) and v.fill_value!=fill_value:
        v = v.todense()
    if not isinstance(v, sparse.COO):
        if hasattr(v, 'todense'):
            v = v.todense()
        else:
            v = np.asarray(v)
        v = sparse.COO.from_numpy(v, fill_value=fill_value)
    return v


class Sparse:
    def __init__(self, x):
        self.x = x

    def tocoo(self):
        return self.x

    @property
    def ndim(self):
        return self.x.ndim

    @property
    def shape(self):
        return self.x.shape

    @property
    def fill_value(self):
        return self.x.fill_value

    @property
    def dtype(self):
        return self.x.dtype

    def __getitem__(self, k):
        return Sparse(self.x[k])

    def __setitem__(self, k, v):
        if not isinstance(k, tuple):
            k = (k,)

        e = np.where([x==Ellipsis for x in k])[0]
        if len(e)>1:
            raise IndexError('too many ellipsis')            
        if len(e)==1:
            e = e[0]
            k = k[:e] + tuple([slice(None)]*(self.ndim-len(k)+1)) + k[(e+1):]

        if len(k)!=self.ndim:
            raise IndexError('invalid dim')

        def _to_list(i, x):
            if isinstance(x, slice):
                x = np.arange(*x.indices(i))
                return x, len(x)
            x = np.asanyarray(x)
            if x.ndim>1:
                raise IndexError('int, array-like of int, or bool')
            if x.ndim==0:
                return x.reshape(1), 0
            if x.dtype==bool:
                if x.shape[0]!=i:
                    raise IndexError('bools len mismatch')
                x = np.where(x)[0]
                s = len(x)
            else:
                x = np.array([i+y if y<0 else y for y in x])
                if any(y>=i or y<0 for y in x):
                    raise IndexError('out of bounds')
                s = len(x)
            return x, s

        v_coords = [_to_list(i, x) for i, x in zip(self.shape, k)]
        d = tuple(s for _, s in v_coords if s>0)
        v_coords = [x for x, _ in v_coords]
        v_coords = list(it.product(*v_coords))
        v_coords = np.array(v_coords).T
        if isinstance(v, Sparse):
            v = v.x
        if hasattr(v, 'todense'):
            v = v.todense()
        else:
            v = np.asanyarray(v)
        v = np.broadcast_to(v, d).ravel()
        
        x = self.x
        coords = np.c_[v_coords, x.coords]
        data = np.r_[v, x.data]
        if all(x>0 for x in coords.shape):
            i = np.ravel_multi_index(coords, x.shape)
            i = np.unique(i, return_index=True)[1]
            coords = coords[:,i]
            data = data[i]
        x = sparse.COO(coords, data, shape=x.shape, fill_value=x.fill_value, sorted=False)

        self.x = x

    def astype(self, dtype, **kwargs):
        self.x = self.x.astype(dtype)
        return self

    def reshape(self, shape, **kwargs):
        self.x = self.x.reshape(shape)
        return self
        
    def view(self, dtype):
        return self.astype(dtype)

    def copy(self, *args, **kwargs):
        return self

def zeros(shape, dtype):
    return Sparse(sparse.zeros(shape, dtype=dtype))

def full(shape, fill_value, dtype=None):
    return Sparse(sparse.full(shape, fill_value, dtype=dtype))

class Array(_Array):
    def _decode_chunk_postprocess(self, chunk, expected_shape):
        # view as numpy array with correct dtype
        #VTT
        #chunk = ensure_ndarray(chunk)
        chunk = io.BytesIO(chunk)
        chunk = sparse.load_npz(chunk)
        chunk = Sparse(chunk)

        # special case object dtype, because incorrect handling can lead to
        # segfaults and other bad things happening
        if self._dtype != object:
            #VTT
            chunk = chunk.view(self._dtype)
        elif chunk.dtype != object:
            # If we end up here, someone must have hacked around with the filters.
            # We cannot deal with object arrays unless there is an object
            # codec in the filter chain, i.e., a filter that converts from object
            # array to something else during encoding, and converts back to object
            # array during decoding.
            raise RuntimeError('cannot read object array without object codec')

        # ensure correct chunk shape
        #VTT
        chunk = chunk.reshape(-1, order='A')
        chunk = chunk.reshape(expected_shape or self._chunks, order=self._order)

        return chunk

    def _encode_chunk_preprocess(self, chunk):
        b = io.BytesIO()
        chunk = tocoo(chunk, self._fill_value)
        sparse.save_npz(b, chunk)
        b = b.getvalue()
        return b

    def _zeros(self, shape, dtype=None, order=None):
        return zeros(shape, dtype=dtype or self._dtype)

    def _empty(self, shape, dtype=None, order=None):
        return self._zeros(shape, dtype=dtype)

    def _full(self, shape, fill_value=None, dtype=None, order=None):
        fill_value = fill_value or self._fill_value
        if fill_value is None:
            return self._zeros(shape, dtype=dtype)
        else:
            return full(shape, fill_value, dtype=dtype or self._dtype)

    def __getitem__(self, *args, **kwargs):
        x = super().__getitem__(*args, **kwargs)
        if isinstance(x, Sparse):
            x = x.x
        return x

    def get_basic_selection(self, *args, **kwargs):
        x = super().get_basic_selection(*args, **kwargs)
        if isinstance(x, Sparse):
            x = x.x
        return x

    def get_coordinate_selection(self, *args, **kwargs):
        x = super().get_coordinate_selection(*args, **kwargs)
        if isinstance(x, Sparse):
            x = x.x
        return x

    def get_mask_selection(self, *args, **kwargs):
        x = super().get_mask_selection(*args, **kwargs)
        if isinstance(x, Sparse):
            x = x.x
        return x

    def get_orthogonal_selection(self, *args, **kwargs):
        x = super().get_orthogonal_selection(*args, **kwargs)
        if isinstance(x, Sparse):
            x = x.x
        return x

    def __array__(self, *args):
        x = super().__array__(*args)
        x = x.todense()
        return x