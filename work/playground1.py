#%%
import sparse
import numpy as np
import itertools as it
from zarr.sparse.sparse1 import Sparse
from zarr.sparse.sparse1 import _conv_list, _conv_single, _conv_slice

#%%

np.broadcast_to(
    np.array([]).reshape((2,0)),
    (2,1)
)

#%%

x = Sparse(shape=(3,3), dtype=np.float64)
x[0,0] = Sparse(data=[1], coords=[[0],[0]])
x[0,1:] = Sparse(data=[1,2], coords=[[0, 0],[0,1]])

np.broadcast_shapes((2,), (2,1))


# %%

c = _conv_list(np.array([0, 1, 3]))
c.fwd(np.array([1, 1, 0, 3, 5, -2]))
c.rev(np.array([0, 2, 1, 4, -1, 5]))

c = _conv_slice(slice(10, 20, 2))
c.fwd(np.array([10, 12, 13, 14, 22, 8]))
c.rev(np.array([0, 1, 2, 5, 4, -2, 6]))

c = _conv_single(2)
c.fwd(np.array([2, 3, 2, 4, 2]))
c.rev(np.array([0, 1, -1, 2, 0, 2]))

# %%
