import os, sys, pkg_resources
required_pkg = ['numpy','scipy','pyvista','h5py']
try:
    pkg_resources.require(required_pkg)
except Exception as e:
    print('Required packages:', *required_pkg)
    sys.exit(e)

os.chdir(r'C:\py-scripts\surface_matching')

import h5py, pyvista
import numpy as np
from numpy import all, any, eye, sum, mean, sort, unique, bincount, isin, exp
from scipy.spatial import KDTree, distance_matrix
from numpy.linalg import svd, det, solve
from time import perf_counter as toc
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


N = 3
icp = lambda *args: (0., args[0], Transform()) # icp and icp_o3d are two other options, disable icp for our particular use


class Transform(np.ndarray):
    # Affine transformation only
    def __new__(cls, *shape):
        # Transform(2,3) creates id transformation matrix of size (2,3,4,4)
        # this shape setup is compatible with many vectorized numpy functions
        return np.tile(eye(N+1), (*shape, 1, 1)).view(cls)

    def translate(self, t):
        self[..., N:, :N] += self[..., N:, N:] * t
        return self

    def rotate(self, R):
        self[..., :, :N] = self[..., :, :N] @ R
        return self

    def scale(self, s):
        self[..., N:, N:] /= s
        return self


class Pointset(np.ndarray):

    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    @property
    def centroid(self):
        return self.mean(axis=self.ndim-2, keepdims=True)

    def setdiff(self, other:'Pointset'):
        this, other = self.tolist(), other.tolist()
        return self.__class__([x for x in this if x not in other])

    # def pca(self):
    #     this_self = self - self.centroid
    #     _, _, W = svd(this_self.T@this_self)
    #     return W.T, this_self@W.T

    def transform(self, T):
        self[...] = (self @ T[...,:N,:N] + T[...,N:,:N]) / T[...,N:,N:]
        return self


class Poly(pyvista.PolyData):

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.last_modified = dict(V=toc())

    @property # vertices
    def V(self):
        return self.points.view(Pointset)

    def update(self, attr, v):
        if attr not in self.last_modified or self.last_modified[attr] < self.last_modified['V']:
            setattr(self, '_'+attr, v)
            self.last_modified[attr] = toc()

    @V.setter
    def V(self, v):
        assert self.points.size == v.size
        self.points[...] = v
        self.last_modified['V'] = toc()

    @property # faces
    def F(self):
        return self.faces.reshape(-1,4)[:,1:]


    @property # vertex indicex of free/boundary points
    def E(self):
        if not hasattr(self, '_E'):
            edg_all = sort(self.F[:,[0,1,1,2,2,0]].reshape(-1,2), axis=1)
            edg, ind = unique(edg_all, axis=0, return_inverse=True)
            edg = unique(edg[bincount(ind)==1])
            setattr(self, '_E', edg)
        return self._E


    @property # vertex normal
    def VN(self):
        self.update('VN', self.point_normals)
        return self._VN

    @property # face normal
    def FN(self):
        self.update('FN', self.face_normals)
        return self._FN

    @property # curvature
    def C(self):
        self.update('C', self.curvature())
        return self._C

    @property
    def mean_edge_len(self):
        self.update(
            'mean_edge_len', 
            mean(((self.V[self.F[:,[0,1,2]],:]-self.V[self.F[:,[1,2,0]],:])**2).sum(axis=-1)**.5)
        )
        return self._mean_edge_len


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


def nicp(SRC, TAR, iterations=20):

    t1 = toc()
    pace = 4/iterations # cut distance by this portion each iteration, in concept
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
    
        print(f'nicp finished {i+1} iteration(s) in {toc()-t1:1.4f} seconds', end='\r')
    print('\n')
    return SRC


if __name__=='__main__':
    file = sys.argv[1]
    with h5py.File(file,'r') as f:
        vars = ['f_from','f_to','v_from','v_to','v_registered']
        for v in vars:
            locals()[v] = np.asarray(f[v]).T

    vtkfaces = lambda f: np.hstack(np.hstack((np.tile(3,(f.shape[0],1)), f.astype(int)-1)))
    REG = nicp(
        Poly(v_from, vtkfaces(f_from)), 
        Poly(v_to, vtkfaces(f_to)), 
    )

    with h5py.File(file,'r+') as f:
        f['v_registered'][...] = REG.V.T


