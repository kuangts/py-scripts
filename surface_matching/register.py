from random import seed
import sys, h5py
from xml.dom import minidom
import numpy as np
from numpy import all, any, eye, sum, mean, matmul, sort, unique, bincount, isin, inf, isinf, exp
from scipy.spatial import KDTree, distance_matrix
from scipy.linalg import block_diag, lstsq
from numpy.linalg import svd, det, inv
from time import perf_counter
import pyvista

class Transform(np.ndarray):
    # Affine transformation only
    def __new__(cls, *shape):
        # Transform(2,3) creates id transformation matrix of size (2,3,4,4)
        # this shape setup is compatible with many vectorized numpy functions
        return np.tile(eye(4), (*shape, 1, 1)).view(cls)

    def translate(self, t):
        self[..., 3:, :3] += self[..., 3:, 3:] * t
        return self

    def rotate(self, R):
        self[..., :, :3] = self[..., :, :3] @ R
        return self

    def scale(self, s):
        self[..., 3:, 3:] /= s
        return self

class Pointset(np.ndarray):

    @property
    def centroid(self):
        return self.mean(axis=self.ndim-2, keepdims=True)

    def setdiff(self, other:'Pointset'):
        this, other = self.tolist(), other.tolist()
        return self.__class__([x for x in this if x not in other])

    def pca(self):
        c = self.centroid
        _, _, W = svd(self-c)
        T = Transform().translate(-c).rotate(W.T)
        return T

    def transform(self, T):
        self[...] = (self @ T[...,:3,:3] + T[...,3:,:3]) / T[...,3:,3:]
        return self

###################################################################################


def knn(A:Pointset, B:Pointset, k=1, method='tree', return_distance=False):
    if method == 'tree':
        # rebuild tree after changing data
        d, ind = KDTree(A).query(B, k=k)
    elif method == 'gpu':
        pass
    if return_distance:
        return (ind, d)
    else:
        return ind

def procrustes(A:Pointset, B:Pointset, scaling=True, reflection=False, full_return=False, rotation_only=False): # B ~== b * A @ R + c
    
    # translation
    Ac, Bc = A.centroid, B.centroid
    A = A-Ac
    B = B-Bc
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


def prcsts(A, B, T=None, d=None, scaling=True, reflection=False): # B ~== b * A @ R + c
    
    At = A
    Ac, Bc = A.mean(axis=A.ndim-2, keepdims=True), B.mean(axis=A.ndim-2, keepdims=True)
    A,B = A-Ac, B-Bc
    U,S,V = svd(A.transpose((*range(A.ndim-2),-1,-2)) @ B)
    s = sum(S, axis=-1, keepdims=True) / sum(A**2, axis=(-1,-2), keepdims=True) if scaling else 1
    R = U@V
    id = det(R)<0
    if any(id) and not reflection:
        SN = eye(3)
        SN[-1,-1] = -1
        R[id,:,:] = U[id,:,:] @ SN @ V[id,:,:]
    At[...] = s * A @ R + Bc
    if T is not None:
        T[...] = Transform(*T.shape[0:-2]).translate(-Ac).scale(s).rotate(R).translate(Bc)
    if d is not None:
        x = sum((A-B)**2, axis=(-1,-2)) / sum(B**2, axis=(-1,-2))
        d[...] = x
        
def detect_edges(_, f):
    edg_all = sort(f[:,[0,1,1,2,2,0]].reshape(-1,2), axis=1)
    edg, ind = unique(edg_all, axis=0, return_inverse=True)
    edg_bd = edg[bincount(ind)==1]
    return unique(edg_bd)


def icp(v_src, v_tar, edg_src, edg_tar, n_iteration=inf, full_return=False):

    v_src_start = v_src.copy()
    T = Transform() # reused
    err_prev = np.zeros(()) # reused
    err_this = np.zeros(()) # reused


    while not n_iteration or not err_prev or not err_this or err_prev - err_this > 1e-6:
        err_prev[...] = err_this[...] 

        IDX1 = knn(v_tar, v_src).flatten()
        IDX2 = knn(v_src, v_tar).flatten()
        x1 = ~isin(IDX1, edg_tar)
        x2 = ~isin(IDX2, edg_src)
        ERR1 = sum((v_tar[IDX1[x1]]-v_src[x1])**2, axis=1)**.5
        ERR2 = sum((v_src[IDX2]-v_tar)**2, axis=1)**.5
        x2[ERR2 > ERR1.mean() + 1.96*ERR1.std(ddof=1)] = False

        data_src = np.vstack((v_src[x1,:], v_src[IDX2[x2],:]))
        data_tar = np.vstack((v_tar[IDX1[x1],:], v_tar[x2,:]))

        prcsts(data_src, data_tar, T, err_this)
        v_src.transform(T)

        n_iteration -= 1

    else:
        prcsts(v_src_start, v_src, T)
            
    if full_return:
        return (err_this.tolist(), v_src, T)
    else:
        return T






def icp_failed(v_src, v_tar, edg_src, edg_tar, full_return=False):

    v_src_start = v_src.copy()
    T = Transform() # reused
    err_prev = np.zeros(()) # reused
    err_this = np.zeros(()) # reused
    IDX1 = knn(v_tar, v_src).flatten()
    IDX2 = knn(v_src, v_tar).flatten()
    x1 = ~isin(IDX1, edg_tar)
    x2 = ~isin(IDX2, edg_src)
    v_src_inside = v_src[x1,:]
    v_tar_inside = v_tar[x2,:]

    while not err_prev or not err_this or err_prev - err_this > 1e-6:
        err_prev[...] = err_this[...] 

        IDX1 = knn(v_tar, v_src_inside).flatten()
        IDX2 = knn(v_src, v_tar_inside).flatten()
        ERR1 = sum((v_tar[IDX1]-v_src_inside)**2, axis=1)**.5
        ERR2 = sum((v_src[IDX2]-v_tar_inside)**2, axis=1)**.5
        ERR2 = ERR2[ERR2 > ERR1.mean() + 1.96*ERR1.std(ddof=1)]

        data_src = np.vstack((v_src_inside, v_src[IDX2,:]))
        data_tar = np.vstack((v_tar[IDX1,:], v_tar_inside))

        prcsts(data_src, data_tar, T, err_this)
        v_src_inside.transform(T)
        v_src.transform(T)
    else:
        prcsts(v_src_start, v_src, T)
            
    if full_return:
        return (err_this.tolist(), v_src, T)
    else:
        return T







def vertex_normal(V,F):

    vf = pyvista.PolyData(V, np.hstack(np.hstack((3*np.ones((F.shape[0],1)), F))).astype(int))
    vn = vf.compute_normals(cell_normals=False, point_normals=True)['Normals']
    return vn






def nicp(v_src:Pointset, v_tar:Pointset, f_src, f_tar, iterations):
    # define cutoff
    edg = f_src[:,[0,1,1,2,2,0]].reshape(-1,2)
    d = ((v_src[edg[:,1]]-v_src[edg[:,0]])**2).sum(axis=1)**.5
    cutoff = mean(d)

    # detect edges
    edg_src = detect_edges(v_src, f_src)
    edg_tar = detect_edges(v_tar, f_tar)

    # initial alignment and scaling
    _, v_src, _ = icp(v_src, v_tar, edg_src, edg_tar, full_return=True)
    sourceV = v_src
    targetV = v_tar
    sourceF = f_src
    targetF = f_tar
    p = sourceV.shape[0]

    # General deformation
    mm, MM = sourceV.min(axis=0), sourceV.max(axis=0)
    mindim = min(MM - mm)
    kernel = np.linspace(1,1.5,iterations+1)[::-1]
    nrseeding = (10**np.linspace(2.1,2.4,iterations+1)).round()

    for i in range(iterations):
        seedingmatrix = np.concatenate(
            np.meshgrid(
                np.arange(mm[0], MM[0], mindim/nrseeding[i]**(1/3)), 
                np.arange(mm[1], MM[1], mindim/nrseeding[i]**(1/3)), 
                np.arange(mm[2], MM[2], mindim/nrseeding[i]**(1/3)), 
                indexing='ij')
            ).reshape(3,-1).T
        
        q = seedingmatrix.shape[0]
        IDX1 = knn(targetV, sourceV).flatten()
        IDX2 = knn(sourceV, targetV).flatten()
        x1 = ~isin(IDX1, edg_tar)
        x2 = ~isin(IDX2, edg_src)

        sourcepartial = sourceV[x1,:]
        targetpartial = targetV[x2,:]
        IDXS = knn(targetpartial, sourcepartial).flatten()
        IDXT = knn(sourcepartial, targetpartial).flatten()

        D = distance_matrix(sourcepartial, seedingmatrix)
        
        gamma = 1/(2*D.mean()**kernel[i])
        Datasetsource = np.vstack((sourcepartial,sourcepartial[IDXT]))
        Datasettarget = np.vstack((targetpartial[IDXS],targetpartial))

        vectors = Datasettarget-Datasetsource
        r = vectors.shape[0]

        # define radial basis width for deformation points 
        rep = exp(-gamma * np.vstack((D,D[IDXT]))**2) # gaussian
        tempy2 = block_diag(rep, rep, rep)
        print(tempy2.shape)
        # solve optimal deformation directions with regularisation term      
        lamb = 0.001
        tx = perf_counter()
        # ppi, *_ = lstsq(tempy2.T@tempy2 + lamb * eye(3*q), tempy2.T, check_finite=False, lapack_driver='gelsy')
        modes = inv(tempy2.T@tempy2 + lamb * eye(3*q)) @ tempy2.T @ vectors.T.reshape(-1,1)
        print(perf_counter()-tx)
        
        D2 = distance_matrix(sourceV, seedingmatrix)
        gamma2 = 1/(2*D2.mean()**kernel[i])
        
        rep = exp(-gamma2 * D2**2)
        tempyfull2 = block_diag(rep, rep, rep)

        test2 = tempyfull2 @ modes

        # deforme source mesh
        sourceV = sourceV + test2.reshape(3,-1).T
        
        _, sourceV, T = icp(sourceV, targetV, edg_src, edg_tar, full_return=True)
        

        
        
    # local deformation
    kk = 12+iterations
    normalsT = vertex_normal(targetV, targetF)*cutoff

    # define local mesh relation
    normalsS = vertex_normal(sourceV, sourceF)*cutoff
    arr = np.hstack((sourceV, normalsS))
    IDXsource, Dsource = knn(arr, arr, k=kk, return_distance=True)

    # check normal direction
    IDXcheck = knn(targetV,sourceV)
    testpos = sum((normalsS-normalsT[IDXcheck,:])**2)
    testneg = sum((normalsS+normalsT[IDXcheck,:])**2)

    if testneg < testpos:
        normalsT = -normalsT
        targetF = targetF[:,[1,3,2]]

        
    for ddd in range(iterations):
        k = kk-ddd-1

        normalsS = vertex_normal(sourceV, sourceF)*cutoff


        sumD = Dsource[:,:k].sum(axis=1,keepdims=True)
        sumD2 = np.tile(sumD,(1,k))
        sumD3 = sumD2-Dsource[:,:k]
        sumD2 = sumD2*(k-1)
        weights = sumD3/sumD2

        IDXtarget, Dtarget = knn(np.hstack((targetV,normalsT)), np.hstack((sourceV,normalsS)), k=3, return_distance=True)
        pp1 = targetV.shape[0]
        pp = pp1
        if len(edg_tar):
            for nei in range(3):
                correctionfortargetholes = isin(IDXtarget[:,nei], edg_tar)
                IDXtarget[correctionfortargetholes,nei] = targetV.shape[0] + np.arange(sum(correctionfortargetholes))
                Dtarget[correctionfortargetholes,nei] = 0.00001
                targetV = np.vstack((targetV, sourceV[correctionfortargetholes,:]))


        
        summD = Dtarget.sum(axis=1,keepdims=True)
        summD2 = np.tile(summD,(1,3))
        summD3 = summD2-Dtarget
        weightsm = summD3/summD2/2

        Targettempset = sum(weightsm[...,None]*targetV[IDXtarget,:], axis=1)
        targetV = targetV[:pp1,:]
        sourceVapprox = sourceV.copy()

        sourceset = sourceV[IDXsource[:,:k],:]
        targetset = Targettempset[IDXsource[:,:k],:]
        T = Transform(sourceset.shape[0])
        prcsts(sourceset, targetset, T, scaling=False)
        T = sum(weights[...,None,None] * T[IDXsource[:,:k],:,:], axis=1)
        sourceV = sourceV[:,None,:].transform(T).squeeze()
        sourceV = (sourceV + sourceVapprox)/2

    return sourceV













if __name__=='__main__':
    file = sys.argv[1] if len(sys.argv)>1 else r'surface_matching\test.mat'
    with h5py.File(file,'r') as f:
        vars = ['f_from','f_to','v_from','v_to']
        for v in vars:
            locals()[v] = np.asarray(f[v]).T

    f_from = f_from.astype(int)-1
    f_to = f_to.astype(int)-1
    edg_src = detect_edges(None, f_from)
    edg_tar = detect_edges(None, f_to)

    t1 = perf_counter()
    v_registered = nicp(v_from.view(Pointset), v_to.view(Pointset), f_from, f_to, iterations=1)
    t2 = perf_counter()

    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)

    print(f'time: {t2-t1} seconds')
    # print(f'error: {err}')
    # # print(f'transformation matrix:\n{T}'.replace('\n','\n\t'))
    # print(f'rotation:\n{T[:3,:3]}'.replace('\n','\n\t'))
    # print(f'translation:\n{T[3,:3]/T[3,3]}'.replace('\n','\n\t'))
    # print(f'scaling:\n{1/T[3,3]}'.replace('\n','\n\t'))
