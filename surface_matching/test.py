import numpy as np
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

locals().update(loadmat(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\MATLAB\surface_matching_complete_code\nonrigidICP\translation.mat', ['target','source','Indices_edgesS','Indices_edgesT']))
e,r,t = icp()

