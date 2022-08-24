import numpy as np
import scipy
import h5py
from register import *
# from mesh import TriangleSurface

# locals().update(loadmat(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\MATLAB\surface_matching_complete_code\nonrigidICP\translation.mat', ['v','f']))

# m = TriangleSurface(V=v,F=f.astype(int))

# v, f = m.V.view(Pointset), m.F
# v_to = v*.3+[[1,2,3]]
# T = procrustes(v, v_to)

# ind = (v@T).knn_in(v_to, k=2)
# print(ind)
A = np.asarray([[0.22673895,0.65789433,0.11181066],
    [0.55487529,0.09631115,0.05480821],
    [0.25265662,0.7790902,0.46922793]])

B = np.asarray([[0.86033266,0.7259154,0.39855505],
    [0.51279269,0.00707718,0.98554707],
    [0.25784092,0.42248201,0.4500361 ]])

err, At, T = procrustes(A, B, scaling=True, full_return=True)
print(A)
print(B)
print(At)
print(err)
print(T)

_, At, d = scipy.spatial.procrustes(B,A)

pass

# locals().update(loadmat(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\MATLAB\surface_matching_complete_code\nonrigidICP\translation.mat', ['target','source','Indices_edgesS','Indices_edgesT']))
# e,r,t = icp()

