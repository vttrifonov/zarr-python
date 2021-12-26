import numpy as np
import itertools as it

class _conv_slice:
    def __init__(self, s):
        self.s = s
        self.len = np.int64(np.ceil((s.stop-s.start)/s.step))

    def fwd(self, x):
        r = (x - self.s.start)/self.s.step
        f = (x<self.s.start) | (x>=self.s.stop) | (np.floor(r) != r)
        r[f] = -1
        return r.astype(np.int64)

    def rev(self, x):
        r = self.s.start + self.s.step*x
        f = (x<0) | (r>=self.s.stop)
        r[f] = -1
        return r.astype(np.int64)

class _conv_list:
    def __init__(self, l):
        self.l = l
        self.len = np.int64(len(l))

    def fwd(self, x):        
        r1 = np.searchsorted(self.l, x)
        i = np.where(r1<len(self.l))[0]
        i = i[x[i]==self.l[r1[i]]]
        r = np.full(len(x), -1, dtype=np.int64)
        r[i] = r1[i]
        return r.astype(np.int64)

    def rev(self, x):
        r = np.full(len(x), -1, dtype=np.int64)
        i = np.where((x>=0) & (x<len(self.l)))[0]
        r[i] = self.l[x[i]]
        return r.astype(np.int64)

class _conv_single:
    def __init__(self, x):
        self.x = x
        self.len = np.int64(0)
    
    def fwd(self, x):
        return np.where(x==self.x, 0, -1).astype(np.int64)

    def rev(self, x):
        return np.where(x==0, self.x, -1).astype(np.int64)

class Sparse:
    def __init__(self, 
        data = None, coords = None,
        fill_value = None, dtype = None,
        shape = None, 
        order = 'C',  normalized = False
    ):
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

            if data is None:
                if coords.shape[1]>0:
                    raise ValueError('data and coords must have same length')
            else:
                data = np.asarray(data)
                if data.ndim != 1:
                    raise ValueError('data must be 1D')
                if coords.shape[1] != len(data):
                    raise ValueError('data and coords must have same length')
        else:
            if data is not None:            
                try:
                    data = np.full((1,), data)
                except ValueError:
                    _ = np.empty((1,), dtype=object)
                    _[0] = data
                    data = _
                coords = np.empty((0,1), dtype=np.int64)

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
                shape = ()
            else:
                shape = np.max(coords, axis=1)+1
        else:
            if coords is not None:
                if len(shape) != coords.shape[0]:
                    raise ValueError('shape and coords must have same dims')

                if coords.shape[1]>0:
                    if any(x>=y for x, y in zip(np.max(coords, axis=1), shape)):
                        raise ValueError('coordinate too large')

        shape = tuple(shape)

        if data is None:
            data = np.array([], dtype=dtype)
        else:
            data = data.astype(dtype)

        if coords is None:
            coords = np.empty((len(shape),0), dtype=np.int64)

        normalized = np.full((), normalized, dtype=bool)[()]
        
        self._data = data
        self._coords = coords
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
    def size(self):
        return np.prod(self._shape)

    @property
    def coords(self):
        return self._coords

    @property
    def index(self):
        return np.ravel_multi_index(
            self._coords, self._shape, order=self._order
        )

    @property
    def data(self):
        return self._data

    @property
    def normalized(self):
        return self._normalized

    def normalize(self):
        index = self.index
        data = self._data
        coords = self._coords
        index, _ = np.unique(index, return_index=True)        
        data, coords = data[_], coords[:, _]
        _ = np.where(data!=self._fill_value)[0]
        index, data, coords = index[_], data[_], coords[:,_]
        _ = np.argsort(index)
        data, coords = data[_], coords[:,_]
        return self.__class__(
            data = data, coords=coords,
            fill_value = self._fill_value, 
            shape = self._shape, order = self._order, 
            normalized = True
        )

    def reshape(self, shape = None, order = None):
        order = order or self._order
        shape = shape or self._shape

        if shape == self._shape and order == self._order:
            return self

        coords = np.unravel_index(
            self.index, shape=shape, order=order
        )
        normalized = False

        return self.__class__(
            data = self._data,
            coords = coords,
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

    def _conv_key(self, k):
        if not isinstance(k, tuple):
            k = (k,)

        e = np.where([x==Ellipsis for x in k])[0]
        if len(e)>1:
            raise IndexError('too many ellipsis')            
        if len(e)==1:
            e = e[0]
            k = k[:e] + tuple([slice(None)]*(self.ndim-len(k)+1)) + k[(e+1):]

        def conv(i, x):
            if isinstance(x, slice):
                return _conv_slice(x.indices(i))

            x = np.asanyarray(x)
            if x.ndim>1:
                raise IndexError('int, array-like of int, or bool')

            if x.ndim==0:
                return _conv_single(x)

            if x.dtype==bool:
                if len(x)!=i:
                    raise IndexError('bools len mismatch')
                return _conv_list(np.where(x)[0])

            x = np.array([i+y if y<0 else y for y in x])
            if any(y>=i or y<0 for y in x):
                raise IndexError('out of bounds')
            return _conv_list(x)

        return [conv(i, x) for i, x in zip(self._shape, k)]

    def __getitem__(self, k):
        conv = self._conv_key(k)
        coords = zip(conv, list(self._coords))
        coords = [x[0].fwd(x[1]) for x in coords]
        coords = np.array(coords)
        i = np.all(coords>=0, axis=0)
        coords = coords[:,i]
        coords = coords[[x.len>0 for x in conv],:]
        data = self._data[i]
        shape = tuple(x.len for x in conv if x.len>0)
        
        return self.__class__(
            coords = coords, 
            data = data,
            fill_value = self._fill_value, 
            shape = shape, order = self._order, 
            normalized = self._normalized
        )

    def __setitem__(self, k, v):
        conv = self._conv_key(k)
        shape = tuple(x.len if x.len>0 else 1 for x in conv)
        v = v.astype(self.dtype)
        v = v.reshape(shape, order = self._order)
        #needs broadcasting

        coords = zip(conv, list(self._coords))
        coords = [x[0].fwd(x[1]) for x in coords]
        coords = np.array(coords)
        i = np.any(coords<0, axis=0)
        data, coords = self._index[i], self._coords[:,i]

        coords1 = zip(conv, list(v.coords))
        coords1 = [x[0].rev(x[1]) for x in coords1]        
        coords1 = np.array(coords1)

        self._coords = np.c_[coords1, coords]
        self._data = np.r_[v.data, data]
        self._normalized = False

    def __array__(self):
        a = np.full(self.size, self._fill_value)
        a[self.index] = self._data
        a = a.reshape(self._shape, order=self._order)
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
        data = np.tile(self._data, coords1.shape[1])    
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
            coords = coords3, 
            data = data,
            fill_value = self._fill_value, 
            shape = shape, order = self._order, 
            normalized = self._normalized
        )




        
        

