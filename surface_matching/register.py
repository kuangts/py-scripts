import numpy as np
from numpy.linalg import svd, det
import h5py as h5
import scipy, os, sys


class Transform(np.ndarray):

    def __new__(cls):
        return np.asarray(np.eye(4)).view(cls)

    def __matmul__(self, other):
        assert isinstance(other, self.__class__)
        return super().__matmul__(other)

    def translate(self, t):
        T = self.__class__()
        T[3, 0:3] = t
        return self@T

    def rotate(self, R):
        T = self.__class__()
        T[0:3, 0:3] = R
        return self@T

    def scale(self, s):
        T = self.__class__()
        T[3, 3] = 1/s
        return self@T

class Pointset(np.ndarray):

    @property
    def v4(self):
        return np.hstack((self, np.ones((self.shape[0],1))))

    @property
    def centroid(self):
        return self.mean(axis=0)[None,...]

    def __matmul__(self, other):
        if isinstance(other, Transform):
            x = self.v4 @ other
            return (x[:,0:3]/x[:,[3]]).view(self.__class__)
        else:
            return super().__matmul__(other)

    def setdiff(self, other):
        this, other = self.tolist(), other.tolist()
        return self.__class__([x for x in this if x not in other])

    def knn_in(self, tar, k=1):
        d = scipy.spatial.distance_matrix(self, tar)
        return d.argsort(axis=1)[:, np.arange(k)]

    def pca(self):
        c = self.centroid
        _, _, W = svd(self-c)
        T = Transform().translate(-c).rotate(W.T)
        return T

def detect_edges(_, f):
    edg_all = np.sort(f[:,[0,1,1,2,2,0]].reshape(-1,2), axis=1)
    edg, ind = np.unique(edg_all, axis=0, return_inverse=True)
    edg_bd = edg[np.bincount(ind)==1]
    return np.unique(edg_bd)
    # return np.hstack((edg_bd[:,0], edg_bd[:,1]))


def define_cutoff(v, f):
    edg = f[:,[0,1,1,2,2,0]].reshape(-1,2)
    d = ((v[edg[:,1]]-v[edg[:,0]])**2).sum(axis=1)**.5
    return np.mean(d)


def loadmat(file, varnames):
    with h5.File(file,'r') as f:
        return {v:np.asarray(f[v]).T for v in varnames}


###################################################################################


def procrustes(v_src, v_tar, scaling=True, reflection=False, full_return=False): # B === b * A @ R + c
    # R = scipy.linalg.sqrtm(H.T @ H) @ scipy.linalg.inv(H) # one liner, but might be slower

    T = Transform()
    A, B = v_src.view(Pointset), v_tar.view(Pointset)
    Ac, Bc = A.centroid, B.centroid
    A, B = A - Ac, B - Bc
    T = T.translate(-Ac)
    if scaling:
        b = (np.sum(B**2)**.5)/(np.sum(A**2)**.5)
        A = A*b
        T = T.scale(b)
    H = B.T @ A
    U,S,V = svd(H)
    R = V.T @ U.T
    print((S**.5))
    if det(R)<0 and not reflection:
        SN = np.eye(3)
        SN[-1,-1] = -1.
        R = V.T @ SN @ U.T
    A = A @ R + Bc
    T = T.rotate(R).translate(Bc)

    if full_return:
        At = v_src.view(Pointset) @ T
        err = np.sum((A-v_tar)**2) / np.sum(B**2)
        return (err, A, T)
    else:
        return T


def icp(v_src=None, v_tar=None, edg_src=None, edg_tar=None, pre_aligned=False):
    mat = loadmat(r'C:\Users\tmhtxk25\OneDrive - Houston Methodist\MATLAB\surface_matching_complete_code\nonrigidICP\translation.mat', ['target','source','Indices_edgesS','Indices_edgesT','x','y'])
    v_src = mat['source'].view(Pointset)
    v_tar = mat['target'].view(Pointset)
    edg_src = mat['Indices_edgesS'].astype(int).flatten()-1
    edg_tar = mat['Indices_edgesT'].astype(int).flatten()-1
    pre_aligned=True
    x = mat['x'].astype(int).flatten()-1
    y = mat['y'].astype(int).flatten()-1

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

        IDX1 = v_src.knn_in(v_tar).flatten()
        IDX2 = v_tar.knn_in(v_src).flatten()
        x1 = np.logical_not(np.isin(IDX1, edg_tar))
        x2 = np.logical_not(np.isin(IDX2, edg_src))
        print(x1.sum())
        print(x2.sum())
        ERR1 = np.sum((v_tar[IDX1[x1]]-v_src[x1])**2, axis=1)**.5
        ERR2 = np.sum((v_src[IDX2]-v_tar)**2, axis=1)**.5
        x2[ERR2 > ERR1.mean() + 1.96*ERR1.std(ddof=1)] = False

        v_src = np.vstack((v_src[x1,:], v_src[IDX2[x2],:])).view(Pointset)
        v_tar = np.vstack((v_tar[IDX1[x1],:], v_tar[x2,:])).view(Pointset)

        err, v_src, T_temp = procrustes(v_src, v_tar, full_return=True)
        error.append(err)
        
    T = T @ procrustes(v_src_start, v_src)

    if not pre_aligned:
        T = T @ T_tar_pca.T

    return T

if __name__=='__main__':
    file = sys.argv[1]
    locals().update(loadmat(file,['v_registered','f_from','v_to','f_to']))
