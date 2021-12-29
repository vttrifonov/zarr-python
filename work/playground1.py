#%%
import sparse
import numpy as np
import itertools as it
from zarr.sparse.sparse1 import array
import pickle
from pathlib import Path

#%%

x3 = Path('xxx.pickle')
if not x3.exists():
    x1 = np.random.random(int(10e6))
    x2 = np.random.randint(int(1e9), size=len(x1))
    x2 = np.sort(x2)
    x2 = np.unravel_index(x2, (1000,)*3)
    with x3.open('wb') as file:
        pickle.dump(
            array(x1, x2, shape=(1000,)*3),
            file
        )
with x3.open('rb') as file:
    x3 = pickle.load(file)


#%%
%timeit x4 = x3[(slice(100,200),)*3]

#%%
%timeit x4 = x3[(list(range(100, 200)),)*3]


#%%
r = [[1], [2,3], [4], [5,6,7], [10,11]]

%timeit np.array(list(it.product(*r))).T

# %%

x = np.empty((2,2), dtype=object)
x[0,:] = [[1,2], [1]]
x[1,:] = [[], [1,2,3]]
np.vectorize(len)(x).prod(axis=0).sum()