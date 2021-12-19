#%%
import sparse
import numpy as np
import itertools as it

# %%
x = np.array(['abc', '', 'def'], dtype=object)
x = sparse.COO.from_numpy(x, fill_value='')
sparse.save_npz('xxx.npz', x)
x = sparse.load_npz('xxx.npz')

# %%
np.broadcast_to(x, (100,1))

x = np.where([False])[0]