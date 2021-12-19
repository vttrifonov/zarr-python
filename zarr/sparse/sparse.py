import sparse 
import numpy as np
import io
from zarr.sparse.core import Array as _Array

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
        if isinstance(v, Sparse):
            self[k] = v.x
            return

        if not isinstance(k, tuple):
            k = (k,)

        if len(k)!=self.ndim:
            raise IndexError('invalid dim')

        x = self.x
        v = tocoo(v, x.fill_value)
        def _to_list(i, x):
            if isinstance(x, slice):
                return np.arange(*x.indices(i))
            x = np.asanyarray(x)
            if x.ndim>1:
                raise IndexError('int, array-like of int, or bool')
            if x.ndim==0:
                x = x.reshape(1)
            if x.dtype==bool:
                if x.shape[0]!=i:
                    raise IndexError('bools len mismatch')
                x = np.where(x)[0]
            else:
                x = np.array([i+y if y<0 else y for y in x])
                if any(y>=i or y<0 for y in x):
                    raise IndexError('out of bounds')
            return x
        i = [_to_list(i, x) for i, x in zip(self.shape, k)]        
        v = np.broadcast_to(v, tuple(len(y) for y in i))
        
        x_coords = x.coords
        x_data = x.data

        if all(x>0 for x in x_coords.shape):
            j = np.apply_along_axis(
                lambda x: any(~np.isin(x, y) for x, y in zip(x, i)),
                0, x_coords
            )
            x_coords = x_coords[:, j]
            x_data = x_data[j]

        def _translate_coord(c):
            return [x[y] for x, y in zip(i, c)]
        v_coords = v.coords
        if all(x>0 for x in v_coords.shape):            
            v_coords = np.apply_along_axis(_translate_coord, 0, v_coords)
        
        coords = np.c_[v_coords, x_coords]
        data = np.r_[v.data, x_data]
        if all(x>0 for x in coords.shape):
            i = np.apply_along_axis(np.ravel_multi_index, 0, coords, x.shape)
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