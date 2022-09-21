'''
This module is currently not used for non rigid-body surface registration purpose
'''

import numpy as np
from numpy import sum, isin, inf, exp
from scipy.spatial import KDTree
from time import perf_counter as toc
import register

sigmoid = lambda x: 1 / (1 + exp(-x))

def icp(SRC, TAR, weighted=False, max_iteration=inf):

    t0 = toc()
    if max_iteration < 2: max_iteration = 2
    v_src_start = SRC.V.copy()
    SRC = SRC.copy()
    T = register.Transform()         # reused
    err = []
    src_nn = lambda x,k=1:KDTree(SRC.V).query(x,k=k)[1].flatten() # late binding on SRC.V, updating SRC.V has effect
    tar_nn = lambda x,k=1,tar_tree=KDTree(TAR.V): tar_tree.query(x,k=k)[1].flatten()
    converged = lambda err: len(err) >= 2 and (err[-2] - err[-1] < 1e-6  or err[-1] < err[0]*.01)

    if weighted:
        c_src, c_tar = SRC.C, TAR.C
        c_src[c_src==0] = max(c_src)
        c_tar[c_tar==0] = max(c_tar)

    while len(err) <= max_iteration and not converged(err):

        IDX1, IDX2 = tar_nn(SRC.V), src_nn(TAR.V)
        x1,   x2   = ~isin(IDX1, TAR.E), ~isin(IDX2, SRC.E)
        ERR1 = sum((TAR.V[IDX1[x1]]-SRC.V[x1,:])**2, axis=1)**.5
        ERR2 = sum((SRC.V[IDX2]-TAR.V)**2, axis=1)**.5
        x2[ERR2 > ERR1.mean() + 1.96*ERR1.std(ddof=1)] = False

        data_src = np.vstack((SRC.V[x1,:], SRC.V[IDX2[x2],:]))
        data_tar = np.vstack((TAR.V[IDX1[x1],:], TAR.V[x2,:]))

        W = sigmoid(np.hstack((c_src[x1], c_src[IDX2[x2]]))*np.hstack((c_tar[IDX1[x1]], c_tar[x2]))) if weighted else None
        er, _, T[...] = register.procrustes(data_src, data_tar, W=W)
        err.append(er)
        SRC.V.transform(T)
    else:
        print(f'icp ran {len(err)} iterations, {toc()-t0} seconds')
        err, SRC.V[...], T[...] = register.procrustes(v_src_start, SRC.V)
        return err, SRC, T


def icp_o3d(SRC, TAR):

    import open3d as o3d
    _, IDX1 = KDTree(TAR.V).query(SRC.V)
    _, IDX2 = KDTree(SRC.V).query(TAR.V)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(SRC.V[~isin(IDX1,TAR.E),:])    
    tar = o3d.geometry.PointCloud()
    tar.points = o3d.utility.Vector3dVector(TAR.V[~isin(IDX2,SRC.E),:])    
    reg_p2p = o3d.pipelines.registration.registration_icp( src, tar, 1,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    T = reg_p2p.transformation.T
    print(T)
    SRC.V.transform(T)
    return (reg_p2p.inlier_rmse, T)


