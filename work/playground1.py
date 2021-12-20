#%%
import sparse
import numpy as np
import itertools as it

# %%
x = sparse.random(
    (10, 10, 10), 
    nnz=int(500), 
    data_rvs=lambda nnz: np.random.random(nnz)
)

# %%
c1 = np.ravel_multi_index(x.coords, x.shape)
c2 = np.unravel_index(c1, shape=(1000,))
x1 = x.reshape((1000,))
c3 = np.ravel_multi_index(x1.coords, x1.shape)

np.all(c1==c3)