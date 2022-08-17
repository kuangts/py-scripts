import numpy as np
from numpy.linalg import svd, det
import h5py as h5
import scipy, os, sys


class Transform(np.ndarray):

    def __init__(self):
        super().__init__(np.eye(4))

    def __matmul__(self, other):
        assert isinstance(other, self.__class__)
        return super().__matmul__(other)

    def apply(self, pointset):
        x = pointset.v4 * self
        return pointset.__class__(x[:,0:3]/x[:,3]) 

    @classmethod
    def identity(cls):
        T = cls(np.eye(4))
        return T

    def translate(self, t):
        T = self.identity()
        T[3, 0:3] = t
        return self@T

    def rotate(self, R):
        T = self.identity()
        T[0:3, 0:3] = R
        return self@T

    def scale(self, s):
        T = self.identity()
        T[3, 3] = 1/s
        return self@T

class Pointset(np.ndarray):

    @property
    def centroid(self):
        return self.mean(axis=0)[None,...]

    def __init__(self, ps):
        super().__init__(ps).reshape(-1,3)

    def __matmul__(self, other):
        if isinstance(other, Transform):
            return other.apply(self)
        else:
            return super().__matmul__(other)

    def v4(self, ps):
        return np.hstack((self, np.ones((ps.shape[0],1))))

    def setdiff(self, other):
        this, other = self.tolist(), other.tolist()
        return self.__class__([x for x in this if x not in other])

    def knn_in(self, tar, k=1):
        d = tar[None,:,:] - self[:,None,:]
        ind = np.sum(d**2, axis=2).argsort(axis=1)
        return ind[:, np.arange(k)]

    def pca(self):
        c = self.centroid
        _, _, W = svd(self-c)
        T = Transform.identity().translate(-c).rotate(W.T)
        return T

def detect_edges(_, f):
    edg_all = f[:,[0,1,1,2,2,0]].reshape(-1,2).sort(axis=1)
    edg, ind = np.unique(edg_all, axis=0, return_inverse=True)
    edg_bd = edg[np.bincount(ind)==1]
    return np.unique(edg_bd)


def define_cutoff(v, f):
    edg = f[:,[0,1,1,2,2,0]].reshape(-1,2)
    d = ((v[edg[:,1]]-v[edg[:,0]])**2).sum(axis=1)**.5
    return np.mean(d)


def loadmat(file, varnames):
    with h5.File(file,'r') as f:
        return {v:np.asarray(f[v]) for v in varnames}


###################################################################################


def procrustes(v_src, v_tar, scaling=True, reflection=False): # B === b * A @ R + c
    # R = scipy.linalg.sqrtm(H.T @ H) @ scipy.linalg.inv(H) # one liner, but might be slower

    T = Transform.identity()
    A, B = v_src, v_tar
    Ac, Bc = A.centroid, B.centroid
    A, B = A - Ac, B - Bc
    T = T.translate(-Ac)
    if scaling:
        b = np.sum(B**2)**.5/np.sum(A**2)**.5
        A = A*b
        T = T.scale(b)
    H = A.T @ B
    U,S,V = svd(H)
    R = V.T @ U.T
    if det(R)<0 and not reflection:
        SN = np.eye(3)
        SN[-1,-1] = -1.
        R = V.T @ SN @ U.T
    T = T.rotate(R).translate(Bc)
    return T


def icp(v_src, v_tar, edg_src, edg_tar, pre_aligned=False):
    T = Transform()
    if not pre_aligned:
        T_src_pca = v_src.pca()
        T_tar_pca = v_tar.pca()
        v_src = v_src @ T_src_pca
        v_tar = v_tar @ T_tar_pca
        T = T @ T_tar_pca
    error = []

    v_src_start = v_src.copy()
    while len(error)<2 or error[-2]-error[-1]>1e-6:

        IDX1 = v_src.knn_in(v_tar)
        IDX2 = v_tar.knn_in(v_src)
        IDX1 = dict(zip(range(len(IDX1)), zip(IDX1, np.sum((v_tar(IDX1)-v_src)**2, axis=1)**.5)))
        IDX2 = dict(zip(range(len(IDX2)), zip(IDX2, np.sum((v_src(IDX2)-v_tar)**2, axis=1)**.5)))

        IDX1 = { k:v for k,v in IDX1.items() if v[0] not in np.unique(edg_tar) }
        IDX2 = { k:v for k,v in IDX2.items() if v[0] not in np.unique(edg_src) }
        d = np.array([ v[1] for v in IDX1.values() ])
        thres = d.mean() + 1.96*d.std()
        IDX2 = { k:v for k,v in IDX2.items() if v[1]<thres }

        id1,_ = zip(*IDX1.values())
        id2,_ = zip(*IDX2.values())
        v_src = np.vstack((v_src[list(IDX1.keys()),:], v_src[id2,:]))
        v_tar = np.vstack((v_tar[id1,:], v_tar[list(IDX2.keys()),:]))

        v_src = v_src @ procrustes(v_src, v_tar)
        error.append(np.sum((v_src-v_tar)**2) / np.sum(v_tar**2))
        
    T = T @ procrustes(v_src_start, v_src)

    if not pre_aligned:
        T = T @ T_tar_pca.T

    return T

if __name__=='__main__':
    file = sys.argv[1]
    locals().update(loadmat(file,['v_registered','f_from','v_to','f_to']))
