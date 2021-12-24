#%%
import sparse
import numpy as np
import itertools as it
from zarr.sparse.sparse import Sparse1

# %%
x1 = Sparse1(shape=(3,3), dtype=np.float64)
x2 = np.zeros((3,3), dtype=np.float64)
def f(i, v):
    x1[i] = v
    x2[i] = v
    x3 = x1.normalize()
    assert np.all(np.array(x3)==x2)
f(np.s_[:2, 0], [1,2])


# %%
