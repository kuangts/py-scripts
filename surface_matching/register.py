import numpy as np
import h5py as h5
import scipy
import os, sys

def loadmat(file, varnames):
    with h5.File(file,'r') as f:
        return {v:np.asarray(f[v]) for v in varnames}

def icp(v_src, v_tar, edg_src, edg_tar, flag):
    v_src_original = v_src.copy()
    if not flag:
        v_src, v_tar, T = Preall(v_src, v_tar)
    error = []
    v_src_aligned = v_src
    while len(error)<2 or error[-2]-error[-1]>1e-6:
        [err, v_src_aligned] = ICPmanu_allign2(v_tar, v_src_aligned, edg_src, edg_tar)
        error.append(err)
    if not flag:
        v_src_aligned = v_src_aligned*T.T + T.c[1,1:3]
    [_ ,v_src_aligned, T] = procrustes(v_src_aligned, v_src_original)
    return (v_src_aligned, T, error[-1])

def detect_edges(_,f):
    edg = f[:,[0,1,1,2,2,0]].reshape(-1,2)
    edg.sort(axis=1)
    edg_unique, ind = np.unique(edg, axis=0, return_index=True)
    edg_remaining = np.delete(edg, ind, axis=0)
    edg_diff = np.array([e for e in edg_unique if not np.any(np.all(e==edg_remaining, axis=1))])
    return edg_diff.T.flatten()

def define_cutoff(v,f):
    edg = f[:,[0,1,1,2,2,0]].reshape(-1,2)
    d = ((v[edg[:,1]]-v[edg[:,0]])**2).sum(axis=1)**.5
    print(d)
    return np.mean(d)

if __name__=='__main__':
    file = sys.argv[1]
    locals().update(loadmat(file,['v_registered','f_from','v_to','f_to']))
    print(v_to)