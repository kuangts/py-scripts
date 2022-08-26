import sys, h5py
from numpy import *
from numpy.linalg import svd, det
from time import perf_counter
from sklearn.neighbors import KDTree

class Transform(ndarray):
    # typing only for hinting the editor

    def __new__(cls):
        return asarray(eye(4)).view(cls)

    def __matmul__(self, other:'Transform'):
        assert isinstance(other, self.__class__)
        return super().__matmul__(other)

    def translate(self, t) -> 'Transform':
        T = self.__class__()
        T[3, 0:3] = t
        return self@T

    def rotate(self, R) -> 'Transform':
        T = self.__class__()
        T[0:3, 0:3] = R
        return self@T

    def scale(self, s) -> 'Transform':
        T = self.__class__()
        T[3, 3] = 1/s
        return self@T

class Pointset(ndarray):

    @property
    def centroid(self):
        return self.mean(axis=self.ndim-2).reshape(*self.shape[0:-2],1,self.shape[-1])

    def setdiff(self, other:'Pointset'):
        this, other = self.tolist(), other.tolist()
        return self.__class__([x for x in this if x not in other])

    def knn(self, qry, k=1):
        # no tree, slow for large set
        if not hasattr(self,'tree'):
            setattr(self, 'tree', KDTree(self))
        ind = self.tree.query(qry, k=k, return_distance=False)
        return ind

    def pca(self):
        c = self.centroid
        _, _, W = svd(self-c)
        T = Transform().translate(-c).rotate(W.T)
        return T

    def transform(self, T):
        return (self @ T[:3,:3] + T[-1:,:3]) / T[...,3,3]
    # return hstack((edg_bd[:,0], edg_bd[:,1]))


###################################################################################


def procrustes(A:Pointset, B:Pointset, scaling=True, reflection=False, full_return=False, rotation_only=False): # B ~== b * A @ R + c
    
    # translation
    Ac, Bc = A.centroid, B.centroid
    A -= Ac
    B -= Bc
    U,S,V = svd(A.T @ B)
    # scaling
    b = S.sum()/sum(A**2) if scaling else 1
    A *= b
    # rotation
    R = U @ V
    if det(R)<0 and not reflection:
        SN = eye(3)
        SN[-1,-1] = -1
        R = U @ SN @ V
    A = A @ R
    T = Transform().translate(-Ac).scale(b).rotate(R).translate(Bc)
    
    if full_return:
        return (
            sum((A-B)**2) / sum(B**2), # procrustes distance
            A+Bc,                      # aligned A
            T,                         # transformation A->B
        ) 
    else:
        return T


def prcsts(A:ndarray, B:ndarray, T:Transform, d:ndarray, scaling=True, reflection=False): # B ~== b * A @ R + c
    
    assert all(A.shape==B.shape) and A.ndim>=2
    n = A.ndim
    At = A
    Ac, Bc = A.centroid, B.centroid
    A -= Ac
    B -= Bc
    U,S,V = svd(A.transpose((*range(n-2),-1,-2)) @ B)
    b = S.sum(-1) / sum(A**2, (-1,-2)) if scaling else 1
    R = matmul(U,V)
    id = det(R)<0
    if any(id) and not reflection:
        SN = eye(3)
        SN[-1,-1] = -1
        R[id,:,:] = U[id,:,:] @ SN @ V[id,:,:]

    A = array(b)[...,None,None] * A @ R + Bc
    T[:] = Transform().translate(-Ac).scale(b).rotate(R).translate(Bc)
    At[:] = A
    d[:] = sum((A-B)**2) / sum(B**2)







def detect_edges(_, f):
    edg_all = sort(f[:,[0,1,1,2,2,0]].reshape(-1,2), axis=1)
    edg, ind = unique(edg_all, axis=0, return_inverse=True)
    edg_bd = edg[bincount(ind)==1]
    return unique(edg_bd)


def icp(v_src:Pointset=None, v_tar:Pointset=None, edg_src=None, edg_tar=None, full_return=False):

    v_src_start = v_src.copy()
    error = []

    while len(error)<2 or error[-2]-error[-1]>1e-6:

        IDX1 = v_tar.knn(v_src).flatten()
        IDX2 = v_src.knn(v_tar).flatten()
        x1 = logical_not(isin(IDX1, edg_tar))
        x2 = logical_not(isin(IDX2, edg_src))
        ERR1 = sum((v_tar[IDX1[x1]]-v_src[x1])**2, axis=1)**.5
        ERR2 = sum((v_src[IDX2]-v_tar)**2, axis=1)**.5
        x2[ERR2 > ERR1.mean() + 1.96*ERR1.std(ddof=1)] = False

        d_src = vstack((v_src[x1,:], v_src[IDX2[x2],:])).view(Pointset)
        d_tar = vstack((v_tar[IDX1[x1],:], v_tar[x2,:])).view(Pointset)

        err, _, T = procrustes(d_src, d_tar, full_return=True)
        v_src = v_src.transform(T)
        error.append(err)

    T = procrustes(v_src_start, v_src)
    if full_return:
        return (error[-1], v_src, T)
    else:
        return T


def nicp(v_src:Pointset, v_tar:Pointset, f_src, f_tar, iterations):
    # define cutoff
    edg = f_src[:,[0,1,1,2,2,0]].reshape(-1,2)
    d = ((v_src[edg[:,1]]-v_src[edg[:,0]])**2).sum(axis=1)**.5
    cutoff = mean(d)

    # detect edges
    edg_src = detect_edges(v_src, f_src)
    edg_tar = detect_edges(v_tar, f_tar)

    # initial alignment and scaling
    err, v_src, T = icp(v_src, v_tar, full_return=True)


if __name__=='__main__':
    file = sys.argv[1]
    with h5py.File(file,'r') as f:
        vars = ['f_from','f_to','v_from','v_to']
        for v in vars:
            locals()[v] = asarray(f[v]).T

    f_from = f_from.astype(int)-1
    f_to = f_to.astype(int)-1
    edg_src = detect_edges(None, f_from)
    edg_tar = detect_edges(None, f_to)

    t1 = perf_counter()
    err, v_aligned, T = icp(v_from.view(Pointset), v_to.view(Pointset), edg_src, edg_tar, full_return=True)
    t2 = perf_counter()
    print(f'{t2-t1} seconds elapsed')
    print(f'error: {err}')
    print(f'transformation matrix: \n{T}')
