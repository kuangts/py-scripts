import numpy as np
import os
from collections import namedtuple


class ndarray(np.ndarray):
    def toset(self):
        return {tuple(a) for a in self.tolist()}

def setdiff2d(s0,s1):
    return np.asarray(list(s0.toset()-s1.toset()))

def union2d(s0,s1):
    return np.asarray(list(s0.toset()-s1.toset()))

def procrustes(B, A): # B === A @ R * s + c
    Am = A.mean(axis=0).reshape(1,-1)
    Bm = B.mean(axis=0).reshape(1,-1)
    A = A - Am
    B = B - Bm
    s = np.sum(B**2)**.5/np.sum(A**2)**.5
    A = A*s
    H = A.T @ B
    U,S,V = np.svd(H)
    R = V.T @ U.T
    T = np.det(R)
    if T<0:
        SN = np.eye(3)
        SN[-1,-1] = -1.
        R = V.T @ SN @ U.T
    # R = scipy.linalg.sqrtm(H.T @ H) @ scipy.linalg.inv(H) # one liner, but might be slower
    Transform = namedtuple('Transform', ['b', 'T', 'c'])
    T = Transform(s, T=R, c=Bm-Am@R*s)
    At = A*R + Bm
    return (At, T)

def knnsearch(B, A, k=1):
    d = B[None,:,:] - A[:,None,:]
    ind = np.sum(d**2, axis=2).argsort(axis=1)
    return ind[:, np.arange(k)]
    
def ICP_manu_align(v_src, v_tar, edg_src, edg_tar):
    IDX1, d1 = knnsearch(v_tar, v_src)
    IDX2, d2 = knnsearch(v_src, v_tar)
    IDX1 = dict(zip(range(len(IDX1)), zip(IDX1, d1)))
    IDX2 = dict(zip(range(len(IDX2)), zip(IDX2, d2)))

    IDX1 = { k:v for k,v in IDX1.items() if v[0] not in np.unique(edg_tar) }
    IDX2 = { k:v for k,v in IDX2.items() if v[0] not in np.unique(edg_src) }
    d = np.array([ v[1] for v in IDX1.values() ])
    thres = d.mean() + 1.96*d.std()
    IDX2 = { k:v for k,v in IDX2.items() if v[1]<thres }

    id1,_ = zip(*IDX1.values())
    id2,_ = zip(*IDX2.values())
    dataset_src = np.vstack((v_src[list(IDX1.keys()),:], v_src[id2,:]))
    dataset_tar = np.vstack((v_tar[id1,:], v_tar[list(IDX2.keys()),:]))

    _, T = procrustes(dataset_tar, dataset_src)
    return T.b * v_src @ T.T + T.c

