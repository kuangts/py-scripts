import os, sys, pkg_resources
required_pkg = ['numpy','scipy','pyvista','vtk']
try:
    pkg_resources.require(required_pkg)
except Exception as e:
    print('Required packages:', *required_pkg)
    sys.exit(e)


import numpy as np
from basic import Transform, Pointset, Poly
from numpy import all, any, eye, sum, mean, sort, unique, bincount, isin, exp, inf
from scipy.spatial import KDTree, distance_matrix
from numpy.linalg import svd, det, solve
from time import perf_counter as toc
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

N = 3
sigmoid = lambda x: 1 / (1 + exp(-x))



def procrustes(A, B, W=None, scaling=False, reflection=False): # B ~== b * A @ R + c
    
    Ac, Bc = A.mean(axis=A.ndim-2, keepdims=True), B.mean(axis=A.ndim-2, keepdims=True)
    A, B = A-Ac, B-Bc
    U,S,V = svd(A.transpose((*range(A.ndim-2),-1,-2)) @ B) if W is None else \
            svd(A.transpose((*range(A.ndim-2),-1,-2)) @ (B*W[:,None]))
    s = sum(S, axis=-1, keepdims=True) / sum(A**2, axis=(-1,-2), keepdims=True) if scaling else 1
    R = U @ V 
    id = det(R)<0
    if any(id) and not reflection:
        SN = eye(N)
        SN[-1,-1] = -1
        R[id,:,:] = U[id,:,:] @ SN @ V[id,:,:]
    At = s * A @ R + Bc
    T = Transform(*At.shape[0:-2]).translate(-Ac).scale(s).rotate(R).translate(Bc)
    d = sum((A-B)**2, axis=(-1,-2)) / sum(B**2, axis=(-1,-2))
    return d, At, T


def icp(SRC, TAR, weighted=False, max_iteration=inf):

    t0 = toc()
    if max_iteration < 2: max_iteration = 2
    v_src_start = SRC.V.copy()
    SRC = SRC.copy()
    T = Transform()         # reused
    err = []
    src_nn = lambda x,k=1:KDTree(SRC.V).query(x,k=k)[1].flatten() # late binding/lexical capturing on SRC.V, updating SRC.V has effect
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
        er, _, T[...] = procrustes(data_src, data_tar, W=W)
        err.append(er)
        SRC.V.transform(T)
    else:
        print(f'icp ran {len(err)} iterations, {toc()-t0} seconds')
        err, SRC.V[...], T[...] = procrustes(v_src_start, SRC.V)
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
    return (reg_p2p.inlier_rmse, T)


def nicp(SRC, TAR, iterations=20, return_error=False):
    icp = lambda *args: (0., args[0], Transform()) # disable icp for our particular use since two surfaces are aligned

    if isinstance(SRC, str):
        SRC = Poly.read(SRC)
    elif isinstance(SRC, dict):
        SRC = Poly(**SRC)
    elif not isinstance(SRC, Poly):
        raise ValueError('wrong input to nicp')

    if isinstance(TAR, str):
        TAR = Poly.read(TAR)
    elif isinstance(TAR, dict):
        TAR = Poly(**TAR)
    elif not isinstance(TAR, Poly):
        raise ValueError('wrong input to nicp')
    
    t1 = toc()
    pace = 1/5 # cut distance by this portion each iteration, in concept
    # from icp import icp # enable icp
    _, SRC, _ = icp(SRC, TAR) # initial icp registration
    SRC, TAR, v_src, vn_src = SRC.copy(), TAR.copy(), SRC.V.copy(), SRC.VN.copy()

    # nearest-neighbor trees for query
    tar_nn = lambda x,k=1,tar_tree=KDTree(TAR.V): tar_tree.query(x,k=k)[1].flatten()
    src_nn = lambda x,k=1:KDTree(SRC.V).query(x,k=k)[1].flatten() # late binding, updating src_tree has effect
    target_normal_nn = lambda x,k=1,target_normal_tree=KDTree(np.hstack((TAR.V,TAR.VN*SRC.mean_edge_len))): \
         target_normal_tree.query(x,k=k)
    source_normal_nn = lambda x,k=1: KDTree(np.hstack((SRC.V,SRC.VN*SRC.mean_edge_len))).query(x,k=k)
    
    mm, MM = SRC.V.min(axis=0), SRC.V.max(axis=0)
    mindim = min(MM - mm)
    kernel = np.linspace(1,1.5,iterations)[::-1]
    nrseeding = (10**np.linspace(2.1,2.4,iterations)).round()
    kk = 12+iterations
    T = Transform(SRC.V.shape[0]) # reused   

    for i,ker in enumerate(kernel):
        side = mindim/nrseeding[i]**(1/3)
        seedingmatrix = np.concatenate(
            np.meshgrid(
                np.arange(mm[0], MM[0], side),
                np.arange(mm[1], MM[1], side), 
                np.arange(mm[2], MM[2], side), 
                indexing='ij')
            ).reshape(3,-1).T
        # seedingmatrix = SRC.V.copy() # time is ~linear to seedingmatrix size

        D = distance_matrix(SRC.V, seedingmatrix)
        IDX1 = tar_nn(SRC.V)
        IDX2 = src_nn(TAR.V)
        x1 = ~isin(IDX1, TAR.E)
        x2 = ~isin(IDX2, SRC.E)
        sourcepartial = SRC.V[x1,:]
        targetpartial = TAR.V[x2,:]
        _, IDXS = KDTree(targetpartial).query(sourcepartial)
        _, IDXT = KDTree(sourcepartial).query(targetpartial)
        vectors = np.vstack((targetpartial[IDXS]-sourcepartial, targetpartial-sourcepartial[IDXT]))

        ################### Gaussian RBF ################
        D_par = np.vstack((D[x1,:], D[x1,:][IDXT,:]))
        basis = exp(-1/(2*D_par.mean()**ker) * D_par**2) 
        modes = solve( basis.T @ basis + 0.001*eye(basis.shape[1]), basis.T @ vectors )
        SRC.V += exp(-1/(2*D.mean()**ker)*D**2) @ modes * pace

        _, SRC, _ = icp(SRC, TAR)

        ################### locally rigid deformation ################
        k = kk-i-1
        arr = np.hstack((SRC.V, SRC.VN*SRC.mean_edge_len))
        Dsource, IDXsource = source_normal_nn(arr, k=k)
        Dtarget, IDXtarget = target_normal_nn(arr, k=3)

        targetV = TAR.V.copy()
        if len(TAR.E): # very important for mesh size difference
            for nei in range(3):
                correctionfortargetholes = isin(IDXtarget[:,nei], TAR.E)
                IDXtarget[correctionfortargetholes,nei] = targetV.shape[0] + np.arange(sum(correctionfortargetholes))
                Dtarget[correctionfortargetholes,nei] = 1e-5
                targetV = np.vstack((targetV, SRC.V[correctionfortargetholes,:]))

        Wtarget = (1 - Dtarget/(Dtarget.sum(axis=1,keepdims=True)))/(3-1)
        targetV = sum(Wtarget[...,None]*targetV[IDXtarget,:], axis=1)
        *_, T[...] = procrustes(SRC.V[IDXsource,:], targetV[IDXsource,:], scaling=False)
        Wsource = (1 - Dsource/(Dsource.sum(axis=1,keepdims=True)))/(k-1)
        T[...] = sum(Wsource[...,None,None] * T[IDXsource,:,:], axis=1)

        SRC.V = v_src + sum((SRC.V[:,None,:].transform(T).squeeze() - v_src) * vn_src, axis=1, keepdims=True) * vn_src * pace


        # end of iteration, keep result for next
        v_src[...] = SRC.V
        vn_src[...] = SRC.VN

    SRC.V = np.ascontiguousarray(SRC.V)
    if return_error:

        return SRC, SRC.V - TAR.V[tar_nn(SRC.V)]
    return SRC


if __name__=='__main__':
    if len(sys.argv) == 1:
        import h5py as h5
        with h5.File(r'.\asset\matlab.mat') as f:
            f_from = np.asarray(f['f_from']).T
            v_from = np.asarray(f['v_from']).T
            f_to = np.asarray(f['f_to']).T
            v_to = np.asarray(f['v_to']).T
            v_from, ind = np.unique(v_from, axis=0, return_inverse=True)
            f_from = np.unique(ind[f_from.astype(int)-1], axis=0)
            v_to, ind = np.unique(v_to, axis=0, return_inverse=True)
            f_to = np.unique(ind[f_to.astype(int)-1], axis=0)
        nicp(
            dict(V=v_from, F=f_from),
            dict(V=v_to, F=f_to)
        ) 
    else:
        # use in powershell: Start-Process "py" -ArgumentList "register.py  from.stl  to.stl" -RedirectStandardOutput registered.stl
        import struct
        REG = nicp( sys.argv[1], sys.argv[2] )
        data = np.hstack((REG.FN, REG.V[REG.F,:].reshape(-1,9))).tolist() # to write in single precision
        # write binary stream to std out
        sys.stdout.buffer.write(
            bytearray(80) + \
            struct.pack('I', len(data)) + \
            b''.join( [struct.pack('f'*len(d), *d) + b'\x00\x00' for d in data] )
        )

