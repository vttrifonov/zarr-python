#%%
import sparse
import numpy as np
import itertools as it
from zarr.sparse.sparse1 import array
import pickle
from pathlib import Path

#%%
class A:
    pass

x = np.empty((3,), dtype=object)
x[0] = np.array([1,[1,2]], dtype=object)
x[1] = 0
x[2] = A()

x1 = np.empty((), dtype=object)
x1[()] = [1,[1,2]]

np.equal(x, x1, dtype=object)


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
n = [1,2,3]
np.repeat(n, [2,2,1])
# %%
