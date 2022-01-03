import numpy as np
import itertools as it
from numcodecs.compat import ensure_bytes, ensure_ndarray
from collections.abc import MutableMapping

class _conv_slice:
    def __init__(self, s):
        self.s = s
        self.len = np.int64(np.ceil((s.stop-s.start)/s.step))
        self.keep = True

    def fwd(self, x):
        r = (x - self.s.start)/self.s.step
        f = (x<self.s.start) | (x>=self.s.stop) | (np.floor(r) != r)
        r[f] = -1
        return r.astype(np.int64)

    def fwd1(self, x):
        return [[j] for j in x]

    def rev(self, x):
        r = self.s.start + self.s.step*x
        f = (x<0) | (r>=self.s.stop)
        r[f] = -1
        return r.astype(np.int64)

class _conv_list:
    def __init__(self, l):
        n = np.int64(len(l))
        o = np.argsort(l)
        l, i = np.unique(l[o], return_index=True)
        i = i[1:]
        i = np.split(np.arange(n)[o], i)
        r = np.full(n, -1, dtype=np.int64)
        if len(l)>0:
            r[[x[-1] for x in i]] = l
        self.l = l
        self.i = i
        self.r = r
        self.len = n
        self.keep = True

    def fwd(self, x):
        r1 = np.searchsorted(self.l, x)
        i = np.where(r1<len(self.l))[0]
        i = i[x[i]==self.l[r1[i]]]
        r = np.full(len(x), -1, dtype=np.int64)
        r[i] = r1[i]
        return r.astype(np.int64)

    def fwd1(self, x):
        return [self.i[j] for j in x]

    def rev(self, x):
        r = np.full(len(x), -1, dtype=np.int64)
        i = np.where((x>=0) & (x<self.len))[0]
        r[i] = self.r[x[i]]
        return r.astype(np.int64)

class _conv_single:
    def __init__(self, x):
        self.x = x
        self.len = np.int64(0)
        self.keep = False
    
    def fwd(self, x):
        return np.where(x==self.x, 0, -1).astype(np.int64)

    def fwd1(self, x):
        return [[j] for j in x]

    def rev(self, x):
        return np.where(x==0, self.x, -1).astype(np.int64)

def _conv(n, x):
    if isinstance(x, slice):
        return _conv_slice(slice(*x.indices(n)))

    x = np.asanyarray(x)
    if x.ndim>1:
        raise IndexError('int, array-like of int, or bool')

    if x.ndim==0:
        x = x if x>=0 else n+x
        if x>=n or x<0:
            raise IndexError('out of bounds')
        return _conv_single(x)

    if x.dtype==bool:
        if len(x) != n:
            raise IndexError('bools len mismatch')
        return _conv_list(np.where(x)[0])

    x = np.array([n+y if y<0 else y for y in x])
    if any(y>=n or y<0 for y in x):
        raise IndexError('out of bounds')
    return _conv_list(x)

def _conv_coords(conv, coords):
    for i in range(len(conv)):
        coords[i,:] = conv[i](coords[i,:])

def _conv_coords1(conv, coords, data):
    if len(data)==0 or coords.shape[0]==0:
        return coords, data

    r = np.empty(coords.shape, dtype=object)
    for i in range(len(conv)):
        r[i,:] = conv[i](coords[i,:])
    
    n = np.vectorize(len)(r).prod(axis=0)
    data1 = np.repeat(data, n, axis=0)

    coords1 = np.empty(
        (coords.shape[0], len(data1)), 
        dtype=coords.dtype
    )
    j = 0
    for i in range(len(data)):        
        for k in it.product(*r[:,i]):
            coords1[:,j] = k
            j += 1

    return coords1, data1

class _conv_key:
    def __init__(self, k, s):
        if not isinstance(k, tuple):
            k = (k,)

        e = np.where([x==Ellipsis for x in k])[0]
        if len(e)>1:
            raise IndexError('too many ellipsis')            
        if len(e)==1:
            e = e[0]
            k = k[:e] + tuple([slice(None)]*(len(s)-len(k)+1)) + k[(e+1):]

        if len(k) < len(s):
            k = k + (slice(None),)*(len(s)-len(k))

        if len(k) != len(s):
            raise IndexError('invalid key')

        self.conv = [_conv(n, x) for n, x in zip(s, k)]

    def fwd(self, coords):
        _conv_coords(
            [x.fwd for x in self.conv], 
            coords
        )

    def fwd1(self, coords, data):
        return _conv_coords1(
            [x.fwd1 for x in self.conv],
            coords, data
        )

    def rev(self, coords):
        _conv_coords(
            [x.rev for x in self.conv], 
            coords
        )

    @property
    def shape1(self):
        return tuple(x.len for x in self.conv if x.keep)

    @property
    def shape2(self):
        return tuple(x.len if x.keep else 1 for x in self.conv)

    @property
    def keep_dims(self):
        return [x.keep for x in self.conv]

class Sparse:
    def __init__(self, 
        data, coords, fill_value, dtype, shape, order, normalized
    ):
        self._data = data
        self._coords = coords
        self._fill_value = fill_value
        self._dtype = dtype
        self._shape = shape        
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
        return self._dtype

    @property
    def order(self):
        return self._order

    @property
    def size(self):
        return np.prod(self._shape).astype(np.int64)

    @property
    def coords(self):
        return self._coords

    @property
    def _index(self):
        return np.ravel_multi_index(
            self._coords, self._shape
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
        coords = self._coords
        index, _ = np.unique(index, return_index=True)        
        data, coords = data[_], coords[:, _]
        _ = np.where(data!=self._fill_value)[0]
        index, data, coords = index[_], data[_], coords[:,_]
        _ = np.argsort(index)
        data, coords = data[_], coords[:,_]
        return self.__class__(
            data, coords, self._fill_value, 
            self._shape, self._order, 
            True
        )

    def reshape(self, shape = None, order = None):
        order = order or self._order
        shape = self._shape if shape is None else shape

        if shape != ():
            shape = np.asarray(shape)
            if shape.ndim==0:
                shape = shape.reshape(1)
            else:
                if shape.ndim != 1:
                    raise ValueError('shape must be 1D')
            try:
                _ = shape.astype(np.int64, casting='safe')
            except TypeError:
                raise ValueError('shape must be int')
            neg = np.where(shape<0)[0]
            if len(neg)>0:            
                if len(neg)>1:
                    raise ValueError('invalid shape')
                neg = neg[0]
                shape[neg] = 1
                x = self.size/np.prod(shape)
                if np.floor(x)!=x:
                    raise ValueError('invalid shape')
                shape[neg] = x.astype(shape.dtype)
            shape = tuple(shape)

        if self.size != np.prod(shape).astype(np.int64):
            raise ValueError('cannot reshape')

        if shape == self._shape and order == self._order:
            return self

        data = self._data
        if self.shape == ():
            coords = np.zeros(
                (len(shape), self.coords.shape[1]),
                dtype=self.coords.dtype
            )
        else:
            if shape == ():
                coords = np.zeros(
                    (0,len(data)), dtype=np.int64
                )
            else:
                coords = np.unravel_index(
                    self._index, shape=shape
                )
                coords = np.array(coords)
        normalized = False

        return self.__class__(
            data, coords, self._fill_value, 
            self._dtype, shape, order, 
            normalized
        )

    def astype(self, dtype):
        if dtype is None:
            return self

        dtype = np.empty((), dtype=[('f', dtype)])
        dtype = dtype.dtype['f']

        if dtype == self.dtype:
            return self

        if dtype.shape != self.dtype.shape:
            raise ValueError('reshaping dtype is not implemented')

        data = self._data.astype(dtype)
        _ = np.empty((), dtype=self.dtype)
        _[()] = self._fill_value
        _ = _.astype(dtype)
        fill_value = _[()]
        return self.__class__(
            data, self._coords, fill_value, 
            dtype, self._shape, self._order, 
            self._normalized
        )

    def _conv_key(self, k):
        return _conv_key(k, self._shape)

    class FieldsView:
        def __init__(self, src, fields):
            self._src = src
            self._fields = fields

        @property
        def _reduced_src(self):
            src = self._src
            fields = self._fields
            data = src._data[fields]
            fill_value = src._fill_value[fields]
            dtype = src.dtype[fields]
            x = self.__class__(
                data, src._coords, fill_value, 
                dtype, src._shape, src._order, 
                src._normalized
            )
            return x

        def __getitem__(self, k):
            return self._reduced_src[k]

        def todense(self):
            return self._reduced_src.todense()

        def __setitem__(self, k, v):
            src = self._src
            dtype = src.dtype[self.fields]
            fill_value = src.fill_value[self.fields]
            order = src.order

            conv = src._conv_key(k)

            if isinstance(v, Sparse):
                v = v.broadcast_to(conv.shape1)
            else:
                _ = np.empty(conv.shape1, dtype=dtype)
                if _.shape == ():
                    _[()] = v
                else:
                    _[...] = v
                v = _
            v = array(
                v, 
                fill_value=fill_value, dtype=dtype
            )
            v = v.reshape(shape=conv.shape2, order=order)
            v_coords = v.coords.copy()
            conv.rev(v_coords)
            i = np.all(v_coords>=0, axis=0)
            v_coords, v_data = v_coords[:,i], v.data[i]
            v_index = np.ravel_multi_index(v_coords, src.shape)
            o = np.argsort(v_index)
            v_coords, v_data = v_coords[:,o], v_data[o]

            coords1 = src._coords.copy()
            conv.fwd(coords1)   
            i = np.any(coords1<0, axis=0)

            data, coords = src.data[i], src._coords[:,i]

            coords1, data1 = coords1[:,~i], src._data[~i]
            coords1, data1 = conv.fwd1(coords1, data1)
            index1 = np.ravel_multi_index(coords1, src.shape)
            index1, i = np.unique(index1, return_index=True)
            coords1, data1 = coords1[:,i], data1[i]

            i = np.searchsorted(v_index, index1)
            j = i<len(v_index)            
            i[j] = -1
            j = np.where(~j)[0]
            j = j[index1[j]!=v_index[i[j]]]
            i[j] = -1
            j = i!=-1
            i = i[j]
            coords1, data1 = coords1[:,j], src._data[j]
            data1[self.fields] = v_data[i]

            i = ~v_index.isin(v_index[i])
            v_coords, v_data = v_coords[:,i], v_data[i]
            v_data1 = np.full(v_data.shape, fill_value)
            v_data1[self.fields] = v_data

            src._coords = np.c_[v_coords, coords1, coords]
            src._data = np.r_[v_data1, data1, data]
            src._normalized = False

    def __getitem__(self, k):
        if (
            isinstance(k, str) or
            isinstance(k, list) and all(isinstance(x, str) for x in k)
        ):
            return Sparse.FieldsView(self, k)

        conv = self._conv_key(k)
        shape = conv.shape1 
        coords = self._coords.copy()
        conv.fwd(coords)
        i = np.all(coords>=0, axis=0)        
        coords, data = coords[:,i], self._data[i]
        coords, data = conv.fwd1(coords, data)
        coords = coords[conv.keep_dims,:]

        r = self.__class__(
            data, coords, self._fill_value, 
            self._dtype, shape, self._order, 
            self._normalized
        )
        if shape==():
            if self._shape != () or k not in (Ellipsis, (Ellipsis,)):
                r = r.todense()[()]
        return r

    def __setitem__(self, k, v):
        conv = self._conv_key(k)

        if isinstance(v, Sparse):
            v = v.broadcast_to(conv.shape1)
        else:
            _ = np.empty(conv.shape1, dtype=self.dtype)
            if _.shape == ():
                _[()] = v
            else:
                _[...] = v
            v = _
        v = array(
            v, 
            fill_value=self._fill_value, dtype=self.dtype
        )
        v = v.reshape(shape=conv.shape2, order=self._order)
        v_coords = v.coords.copy()
        conv.rev(v_coords)
        i = np.all(v_coords>=0, axis=0)
        v_coords, v_data = v_coords[:,i], v.data[i]

        i = self._coords.copy()
        conv.fwd(i)
        i = np.any(i<0, axis=0)
        data, coords = self.data[i], self._coords[:,i]

        self._coords = np.c_[v_coords, coords]
        self._data = np.r_[v_data, data]
        self._normalized = False

    def todense(self):
        a = np.empty(self._shape, dtype=self.dtype)
        if self.dtype==object:
            a.fill(self._fill_value)
        else:
            a[...] = self._fill_value
        if a.shape==():
            if len(self._data)>0:
                a[()] = self._data[0]
        else:
            if len(self._data)>0:
                a[tuple(self._coords)] = self._data
        return a

    def broadcast_to(self, shape):
        s = self._shape
        if len(shape)<len(s):
            raise ValueError('cannot broadcast')
        
        d = len(shape)-len(s)
        shape = list(zip(d*(1,) + s, shape))
        if any([
            x!=y and x!=1 for (x, y) in shape
        ]):
            raise ValueError('cannot broadcast')            
        shape = ((y if x==1 else x, x) for x, y in shape)
        shape = list(enumerate(shape))

        s2 = [(i, x) for i, (x, y) in shape if x!=y or i<d]
        if any(x==0 for _, x in s2):
            coords1 = np.array([], dtype=np.int64)
            coords1 = coords1.reshape((len(s2),0))
        else:
            coords1 = [np.arange(x, dtype=np.int64) for _, x in s2]
            coords1 = list(it.product(*coords1))
            coords1 = np.array(coords1).T

        coords2 = np.tile(self.coords, coords1.shape[1])
        data = np.tile(self._data, (coords1.shape[1],)+(1,)*(self._data.ndim-1))
        coords1 = np.repeat(coords1, len(self._data), axis=1)

        coords3 = np.empty(
            (len(shape), coords1.shape[1]), dtype=np.int64
        )

        i = np.array([i for i, _ in s2])
        if len(i)>0:
            coords3[i,:] = coords1

        i = np.array([i for i, (x, y) in shape if x==y and i>=d])
        if len(i)>0:
            coords3[i,:] = coords2[i-d,:]

        shape = tuple(x for _, (x,_) in shape)
        
        return self.__class__(
            data, coords3, self._fill_value, 
            self._dtype, shape, self._order, 
            self._normalized
        )

    def view(self, dtype):
        return self.astype(dtype)

    def copy(self, order=None):
        return self.__class__(
            self._data.copy(), 
            self._coords.copy(), 
            self._fill_value,             
            self._dtype,
            self._shape, order,
            self._normalized
        )
        
def full( 
    shape, fill_value = None, dtype = None,
    order = 'C'
):
    if order is None:
        raise ValueError('missing order')

    shape = np.asarray(shape)
    if shape.ndim != 1:
        raise ValueError('coords must be 1D')
    if np.any(shape<0):
        raise ValueError('shape cannot be negative')
    if len(shape)>0:
        try:
            _ = shape.astype(np.int64, casting='safe')
        except TypeError:
            raise ValueError('shape must be int')
    shape = tuple(shape)

    if dtype is None:
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
            _ = np.empty((), dtype=dtype)
            _[()] = fill_value
            fill_value = _[()]
    
    data = np.array([], dtype=dtype)
    coords = np.empty((len(shape),0), dtype=np.int64)

    return Sparse(
        data, coords, fill_value, dtype, shape, order, True
    )

def zeros(shape, dtype, order = 'C'):
    return full(shape, dtype=dtype, order=order)

def _not_equal(x, y):
    if isinstance(x, np.ndarray) != isinstance(y, np.ndarray):
        return True
    
    if not isinstance(x, np.ndarray):
        return x != y

    if x.shape != y.shape or x.dtype != y.dtype:
        return True

    if x.dtype != object:
        return (x!=y).all()

    return any(_not_equal(a, b) for a, b in zip(x.ravel(), y.ravel()))

def array( 
    data, coords = None,
    fill_value = None, dtype = None,
    shape = None, 
    order = 'C',  normalized = False
):
    if isinstance(data, Sparse):
        if fill_value is not None and np.any(fill_value != data.fill_value):
            data = data.todense()
            coords = None
        else:
            data = data.astype(dtype)
            data = data.reshape(shape, order=order)
            return data

    if dtype is None:
        data = np.asarray(data)
        dtype = data.dtype
    else:
        dtype = np.empty((), dtype=[('f', dtype)])
        dtype = dtype.dtype['f']
        data = np.asarray(data, dtype=dtype.base)


    if fill_value is None:
        fill_value = np.zeros((), dtype=dtype)[()]
    else:
        _ = np.empty((), dtype=dtype)
        _[()] = fill_value
        fill_value = _[()]

    if order is None:
        raise ValueError('missing order')

    if coords is not None:
        coords = np.asarray(coords)
        if coords.ndim != 2:
            raise ValueError('coords must be 2D')
        if np.any(coords<0):
            raise ValueError('coords cannot be negative')
        try:
            _ = coords.astype(np.int64, casting='safe')
        except TypeError:
            raise ValueError('coords must be int')

        if coords.shape[1] != len(data):
            raise ValueError('data and coords must have same length')

        if shape is None:
            if coords.shape[1]>0:
                shape = np.max(coords, axis=1)+1
            else:
                shape = (0,)*coords.shape[0]
        else:
            if len(shape) != coords.shape[0]:
                raise ValueError('shape and coords must have same dims')

            if coords.shape[1]>0:
                if any(x>=y for x, y in zip(np.max(coords, axis=1), shape)):
                    raise ValueError('coordinate too large')
    else:
        if shape is None:
            if data.ndim == 0:
                shape = ()
            else:
                shape = data.shape[:(data.ndim-len(dtype.shape))]

    shape = np.asarray(shape)
    if shape.ndim != 1:
        raise ValueError('coords must be 1D')
    if np.any(shape<0):
        raise ValueError('shape cannot be negative')
    if len(shape)>0:
        try:
            _ = shape.astype(np.int64, casting='safe')
        except TypeError:
            raise ValueError('shape must be int')
    shape = tuple(shape)
        
    if coords is None:
        data = data.reshape(shape+dtype.shape, order=order)
        if shape==(): 
            data = data[np.newaxis,...]
        if data.dtype == object:
            coords = np.vectorize(lambda x: _not_equal(x, fill_value))(data)
        else:
            coords = data!=fill_value
        coords = coords.any(
            axis=tuple(range(len(shape), data.ndim))
        )
        coords = np.nonzero(coords)
        data = data[coords]
        if shape == ():
            coords = np.zeros((0,len(data)), dtype=np.int64)
        else:
            coords = np.array(coords)
        normalized = True

    normalized = np.full((), normalized, dtype=bool)[()]

    return Sparse(
        data, coords, fill_value, 
        dtype, shape, order, 
        normalized
    )

from zarr.sparse.core import Array as _Array
import pickle

class Array(_Array):
    def _full(self, shape, fill_value=None, dtype=None, order=None):
        fill_value = fill_value if fill_value is not None else self._fill_value
        if fill_value is None:
            return self._zeros(shape, dtype=dtype)
        else:
            return full(shape, fill_value, dtype=dtype or self._dtype)

    def _zeros(self, shape, dtype=None, order=None):
        return zeros(shape, dtype=dtype or self._dtype)

    def _empty(self, shape, fill_value=None, dtype=None, order=None):
        return self._full(shape, fill_value, dtype=dtype)

    def __getitem__(self, *args, **kwargs):
        x = super().__getitem__(*args, **kwargs)
        return x

    def get_basic_selection(self, *args, **kwargs):
        x = super().get_basic_selection(*args, **kwargs)
        return x

    def get_coordinate_selection(self, *args, **kwargs):
        x = super().get_coordinate_selection(*args, **kwargs)
        return x

    def get_mask_selection(self, *args, **kwargs):
        x = super().get_mask_selection(*args, **kwargs)
        return x

    def get_orthogonal_selection(self, *args, **kwargs):
        x = super().get_orthogonal_selection(*args, **kwargs)
        return x

    def __array__(self, *args):
        x = super().__array__(*args)
        x = x.todense()
        return x

    def _encode_chunk(self, chunk):
        chunk = array(chunk, fill_value=self._fill_value)

        data = chunk.data
        # apply filters
        if self._filters:
            for f in self._filters:
                data = f.encode(data)

        # check object encoding
        #VTT
        if ensure_ndarray(data).dtype == object:
            raise RuntimeError('cannot write object array without object codec')

        chunk = (
            data, chunk._coords,
            chunk._fill_value, chunk._dtype, chunk._shape, 
            chunk._order, chunk._normalized
        )
        chunk = pickle.dumps(chunk)

        # compress
        if self._compressor:
            cdata = self._compressor.encode(chunk)
        else:
            cdata = chunk

        # ensure in-memory data is immutable and easy to compare
        if isinstance(self.chunk_store, MutableMapping):
            cdata = ensure_bytes(cdata)

        return cdata

    def _decode_chunk(self, cdata, start=None, nitems=None, expected_shape=None):
        # decompress
        if self._compressor:
            # only decode requested items
            if (
                all(x is not None for x in [start, nitems])
                and self._compressor.codec_id == "blosc"
            ) and hasattr(self._compressor, "decode_partial"):
                chunk = self._compressor.decode_partial(cdata, start, nitems)
            else:
                chunk = self._compressor.decode(cdata)
        else:
            chunk = cdata

        chunk = pickle.loads(chunk)
        chunk = list(chunk)                
        
        # apply filters
        data =  chunk[0]
        if self._filters:
            for f in reversed(self._filters):
                data = f.decode(data)
        chunk[0] = data

        chunk = Sparse(*chunk)

        # view as numpy array with correct dtype
        #VTT
        #chunk = ensure_ndarray(chunk)


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


