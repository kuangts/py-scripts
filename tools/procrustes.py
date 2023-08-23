#!/usr/bin/env python3
'''
Tianshu Kuang
Houston Methodist Hospital
07/2023
'''

'''
A FEW COMMENTS:

because weighting relates to multiple factors, it must be considered separately for each use case
    one way to do it is to first use uniform weights, examine the level of mismatch, 
    and then choose parameters for the weighting function

normalization and scaling
    scaling is not allowed with iterative procrustes
    depending on use case, one can normalize shapes by their individual sizes before procrustes, or
    he can directly input the shapes for procrustes

when aligning multiple human faces, weighting is cautioned, with reasons explained in the following thought experiment
    weighting is a great choice for aligning multiple readings of the same measurement
    because multiple readings come from one truth, and noises have certain distribution
    one could also imagine human faces as coming from one source shape, with wild and random deviation among and within individual subjects
    but in this case, noise is completely unpredictable, rendering weighting useless at best
    calculating left-right symmetry within each face is different
    left and right half-faces can be seen as coming from the same half-shape, therefore, weighting makes sense 

and finally, procrustes analysis and symmetry are two separate concepts
    symmetry is intrinsic to each individual; procrustes concerns relationship among individuals
    symmetry found with the mirroring approach is not a perfect one, but a close approximation
aligning multiple shapes can consider symmetry of the individual shapes
    for example, one could align 2x half faces instead of 1x whole faces to achieve that, implicitly, with functions provided in this module and simple math
    to do so in current application, however, we sacrifice a lot to gain little
    as far as advanced registration method is concerned, surface-based registration would make more sense
'''

import sys, pkg_resources
try:
    pkg_resources.require(['numpy'])
except Exception as e:
    sys.exit(e)

import numpy as np
from numpy.linalg import svd, det
from numpy import nansum as sum
from numpy import nanmean as mean


# a transform wrapper for convenience
class _Transform(np.ndarray):

    # Similarity transformation only
    # always post-multiplied onto coordinates

    def __new__(cls, shape=(), ndim=3):
        # if shape is (I,J), a _Transform object of shape (I,J, ndim+1, ndim+1) will be generated
        # where every [i,j,...] is an identity matrix
        # this shape is consistent with numpy convention
        return np.tile(np.eye(ndim+1, ndim+1),(*shape,1,1)).view(cls)

    def translate(self, t):
        T = self.copy()
        T[..., -1, :-1] += T[..., -1, -1] * t
        return T

    def rotate(self, R):
        T = self.copy()
        T[..., :-1] = T[..., :-1] @ R
        return T

    def scale(self, s):
        T = self.copy()
        T[..., -1, -1] /= s
        return T

    def inv(self):
        return np.linalg.inv(self).view(self.__class__)

    def transform_points(self, P):
        v4 = np.concatenate((P, np.ones((*P.shape[:-1],1))), axis=-1)
        v4 = v4 @ self
        return (v4[...,:-1]/v4[...,[-1]]).view(P.__class__)


#---------------------------------------------------------------------------#
# the following three functions are tested only on simple shapes
# further testing is recommended
# also, currently each call to the iterative method loops 50 times 
#     there are good convergence criteria, instead of a fixed number of iterations
#     practically, unless abused the iterative method converges nicely
#---------------------------------------------------------------------------#


def procrustes(
          target:np.ndarray,           # shape (npts, ndim)
          source:np.ndarray,           # shape (npts, ndim)
          weights=None,                # shape (npts, 1) or (npts,), use None for ordinary procrustes
          scaling=True,                # whether to allow scaling
          reflection=True,             # whether to allow reflection
):
    
    '''finds similarity transformation between two shapes
            `source` -> `target`, weighted by `weights` (per point)
    such that least square error is minimized upon transformation
    transformation is rigid if not `scaling`
    transformation is proper if not `reflection`
    nan values are allowed for missing points - there must be enough non-nan values for SVD to converge, however
    returns
        `d`  -> the similarity measure, or procrustes distance
        `C`  -> `source` transformed by `T`
        `T`  -> transformation that matches `source` towards `target`
    these are similar to MATLAB function `procrustes`
    refer to matlab function description for more info
    '''

    A, B, W = source, target, weights

    # exclude nan values, resulting possibily from missing points, or flawed homology
    mask = (~np.isnan(A) & ~np.isnan(B)).all(axis=1)

    # initialize weights
    if W is None:
        W = np.ones((B.shape[0],1))
    else:
        W = W.squeeze()[:,None]
    
    d, _, T = _core(A[mask], B[mask], W[mask], 
                                      scaling=scaling, reflection=reflection)
    C = T.transform_points(A)
    
    return d, C, T


def iterative_procrustes(
               *shapes,                     # N shapes to be registered (N>=2)
               weighting_func=None,         # a `Callable`, or lambda, that maps distance into weight
               reflection=True,             # whether to allow reflection
               target_shape_index=None,     # use this shape, `shapes[target_shape_index]`, as reference for alignment
               result_shape_index=None,     # return result for only this shape, `shapes[result_shape_index]`, instead of everyone
            ):
    
    '''provides framework for iteratively solving weighted procrustes or generalized procrustes (more than two shapes) problems
        (scaling is disabled because it is difficult to get weights and scaling right together)
    nan values are allowed for missing points - there must be enough non-nan values for SVD to converge

    motivation for weighting:
        orthogonal procrustes is optimal in least square sense
        but suffers the problem of outlier skew
        thus we use weights inversely related to fitting errors
        such that large distance's contribution is down-weighted
    '''

    # https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1111/j.2044-8317.1995.tb01050.x 
    # starting at page 6
    # the T, Z, V steps are implemented here
    # "immediate extension of registering two shapes, with weighting" - further investigation into how scaling somehow interferes with weighting

    shapes = np.array(shapes)
    A = shapes - mean(shapes, axis=1, keepdims=True)  # `A` keeps updating
    d = np.zeros((A.shape[0],))
    W = np.ones((A.shape[0:2]))
    T = _Transform((A.shape[0],), ndim=A.shape[-1])

    if weighting_func is None:
        weighting_func = lambda x: np.ones(x.shape)

    # exclude nan values, resulting possibily from missing points, or flawed homology
    mask = (~np.isnan(A)).all(axis=2)

    for _ in range(50):

        # update transformation for each shape cyclicly with fixed weights
        for i in range(A.shape[0]):
            Am = mean(np.delete(A, i, axis=0), axis=0) # mean shape of all other shapes
            # exclude more points in case `Am` still as NaN values
            m = mask[i].copy()
            m[np.isnan(Am).any(axis=1)] = False
            d[i], _, t = _core(A[i,m,:], Am[m,:], W[i,m], scaling=False, reflection=reflection)
            A[i,mask[i],:] = t.transform_points(A[i,mask[i],:])

        # update weights
        W = weighting_func(sum((A - mean(A, axis=0))**2, axis=2)**.5)

    # find transformation for each shape
    W[...] = 1
    for i in range(A.shape[0]):
        _, _, T[i] = _core(shapes[i,mask[i],:], A[i,mask[i],:], W[i,mask[i]], scaling=False, reflection=reflection)

    if target_shape_index is not None:
        Tinv = T[target_shape_index].inv()
        T = T @ Tinv
        A = Tinv.transform_points(A)

    if result_shape_index is not None:
        d, A, T = d[result_shape_index], A[result_shape_index], T[result_shape_index]

    return d, A, T


def midsagittal_plane(left, midline, right, weighting_func=None):

    '''returns midsagittal plane in form of plane equation coefficients `a`,`b`,`c`,`d`, such that `ax + by + cz + d = 0`
    `left`, `midline`, and `right` are of shape (npts, ndim), 
    `left` and `right` must be corresponding
    `weighting_func` is a `Callable`, or lambda, that maps distance into weight
    '''

    # remove missing points
    mask_lr = (~np.isnan(left) & ~np.isnan(right)).all(axis=1)
    left, right = left[mask_lr], right[mask_lr]
    
    mask_mid = (~np.isnan(midline)).all(axis=1)
    midline = midline[mask_mid]

    # use uniform weight if no weighting function is provided
    if weighting_func is None:
        weighting_func = lambda x: np.ones(x.shape)

    # set up two shapes
    lmr = np.vstack((left, midline, right)) # lmr: left-midline-right
    rml = np.vstack((right, midline, left)) # rml: right-midline-left

    # mirror rml by negating x coordinates
    # in theory, this could be done about any plane
    # here, y-z plane is used
    rml_mirrored = rml.copy() 
    rml_mirrored[:,0] = -rml_mirrored[:,0]

    # rml_mirror and lmr have point-to-point correspondence for alignment
    # align rml_mirrored towards lmr
    d, rml_mirrored, T = iterative_procrustes(lmr, rml_mirrored, weighting_func=weighting_func, reflection=False, target_shape_index=0, result_shape_index=1)

    # these midpoints fall close to optimal midsagittal plane for clinical purposes
    midpoints = rml/2 + rml_mirrored/2

    # a plane could be determined by one the following two:
    # 1) normal direction and a point on the plane, or
    # 2) a,b,c,d such that ax + by + cz + d = 0,
    # the second one is more succint
    # if a point on plane is needed, then use `-[a,b,c]*d` given `[a,b,c]` is normalized
    # this point happens to also be the point closest to origin of the reference frame

    plane_normal = _principle_components(midpoints)[-1,:]
    plane_origin = rml.mean(axis=0)
    plane_abcd = np.array([*plane_normal, -plane_normal.dot(plane_origin)])
    return plane_abcd


def _principle_components(X):
    
    X = X - X.mean(axis=0)
    _,_,W = np.linalg.svd(X.T @ X) # already sorted in descending order
    return W


def _core(
          A:np.ndarray,                # source, (npts, ndim)
          B:np.ndarray,                # target, (npts, ndim)
          W:np.ndarray,                # weights, (npts, 1) or (npts,)
          scaling=True,                # whether to allow scaling
          reflection=True,             # whether to allow reflection
        ) -> tuple[float, np.ndarray, _Transform]:
    
    # an unprotected function for the core math of procrustes
    # not for external use

    if W.ndim < 2:
        W = W[:,None]

    # align centers of gravity
    ac, bc = np.sum(A*W, axis=0)/np.sum(W), np.sum(B*W, axis=0)/np.sum(W)
    A, B = A-ac, B-bc

    # rotate
    U,S,V = svd(A.T @ (B*W))
    R = U @ V
    scale = 1
    if scaling:
        scale = np.sum(S) / np.sum(A**2)
    if det(R)<0 and not reflection:
        sign = np.eye(B.shape[-1])
        sign[-1,-1] = -1
        R = U @ sign @ V

    # return
    C = scale * A @ R
    d = np.sum((C-B)**2) / np.sum(B**2)
    C = C + bc
    T = _Transform(ndim=B.shape[-1]).translate(-ac).scale(scale).rotate(R).translate(bc)

    return d, C, T


def _sigmoid(c1, c2):
    return lambda x: 1-1/(1+np.exp(-c1*(x-c2)))


def _test():
    
    A = np.random.rand(10,3)
    B = np.random.rand(10,3)

    # add some missing values
    A[[1,3],:] = np.nan
    B[[2,4],:] = np.nan

    # for our purpose, no scaling no reflection
    print('TWO RANDOM SHAPES TO TEST CORRECTNESS\n')
    d0, C0, T0 = procrustes(source=A, target=B, scaling=False, reflection=False)
    d, C, T = iterative_procrustes(A, B, weighting_func=None, reflection=False, target_shape_index=1)
    d1, C1, T1 = d[0], C[0], T[0]
    print('DIFFERENCE:\n')
    print('procrustes distance', f'{d0-d1} over {d0}', '',
          'coordinates', C0-C1, '',
          'transformation', T0-T1, '', sep='\n')
    print('CONCLUSION:',
          '`iterative_procrustes` falls back to ordinary `procrustes`',
          'with 1) same two random input shapes 2) uniform weights 3) same target shape', sep='\n')
    print('-'*88)
    ####################################################################################

    print('ADDING TWO ADDITIONAL SHAPES\n')
    D = np.random.rand(10,3)
    E = np.random.rand(10,3) 
    N = np.array((A,B,D,E))
    d1, C1, T1 = iterative_procrustes(*N, weighting_func=None, reflection=False)
    d1, C1, T1 = iterative_procrustes(A, B, D, E, weighting_func=None, reflection=False, target_shape_index=1, result_shape_index=0)
    print('DIFFERENCE:\n')
    print('procrustes distance', f'{d0-d1} over {d0}', '',
          'coordinates', C0-C1, '',
          'transformation', T0-T1, '', sep='\n')
    print('CONCLUSION:',
          'additional shapes could change the optimal alignment between existing shapes', sep='\n')
    print('-'*88)
    ####################################################################################

    print('THE EFFECT OF WEIGHTING\n')
    B = np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0]]) * 200 # same dimension as human head
    A = B.copy()
    A[2,0:2] += 10,10 # some quadeformity
    # for clinical purposes, `A` and `B` are perfectly aligned right now, or
    # identity matrix is the best transformation
    
    d1, C1, T1 = iterative_procrustes(A, B, weighting_func=None, reflection=False, target_shape_index=1, result_shape_index=0)

    weighting_func = _sigmoid(.8, 5)

    d0, C0, T0 = procrustes(source=A, target=B, scaling=False, reflection=False)
    d2, C2, T2 = iterative_procrustes(A, B, weighting_func=weighting_func, reflection=False, target_shape_index=1, result_shape_index=0)
    print('DIFFERENCE:\n')
    print('procrustes distance', f'{d2-d0} over {d0}', '',
          'coordinates', C2-C0, '',
          'transformation', T2-T0, '',
          'clincial error (orthogonal) A', C1-A, '',
          'clincial error (weighted) A', C2-A, '', sep='\n')
    print('CONCLUSION:',
          "weighting alleviates outliers' influence", sep='\n')
    print('-'*88)
    ####################################################################################

    print('SOME ADDITIONAL CROOKED QUADS\n')
    D = B.copy()
    E = B.copy()
    F = B.copy()
    D[1,0:2] += -10,10 # some quadeformity
    E[3,0:2] += 10,-10 # some quadeformity
    F[0,0:2] += -10,-10 # some quadeformity
    d, C, T = iterative_procrustes(A, B, D, E, F, weighting_func=None, reflection=False, target_shape_index=1)
    d1, C1, T1 = d[0], C[0], T[0]
    d2, C2, T2 = d[2], C[2], T[2]
    d3, C3, T3 = d[3], C[3], T[3]
    d4, C4, T4 = d[4], C[4], T[4]
    print('DIFFERENCE WITHOUT WEIGHTING:\n')
    print('procrustes distance', f'{d0-d1}, {d0-d2}, {d0-d3}, {d0-d4} over {d0}', '',
          'clincial error A', C1-A, '',
          'clincial error D', C2-D, '',
          'clincial error E', C3-E, '', 
          'clincial error F', C4-F, '', sep='\n')
    d, C, T = iterative_procrustes(A, B, D, E, F, weighting_func=weighting_func, reflection=False, target_shape_index=1)
    d1, C1, T1 = d[0], C[0], T[0]
    d2, C2, T2 = d[2], C[2], T[2]
    d3, C3, T3 = d[3], C[3], T[3]
    d4, C4, T4 = d[4], C[4], T[4]
    print('DIFFERENCE WITH WEIGHTING:\n')
    print('procrustes distance', f'{d0-d1}, {d0-d2}, {d0-d3}, {d0-d4} over {d0}', '',
          'clincial error A', C1-A, '',
          'clincial error D', C2-D, '',
          'clincial error E', C3-E, '',
          'clincial error F', C4-F, '', sep='\n')
    print('CONCLUSION:',
          'additional shapes could change the optimal alignment between existing shapes', sep='\n')
    print('-'*88)
    ####################################################################################

    print('MIDSAGITTAL PLANE\n')
    left = np.array(((-1,-1,-1),(-.5,1,0)))*100 + np.random.normal(scale=3, size=(2,3))
    right = np.array(((1,-1,-1),(.5,1,0)))*100 + np.random.normal(scale=3, size=(2,3))
    midline = np.array(((0,-.5,0),(0,.5,0)))*100 + np.random.normal(scale=3, size=(2,3))

    print('with only fluctuating asymmetry:\n')
    plane_abcd = midsagittal_plane(left, midline, right)
    print('plane function (not weighted):', '{:+.6f}x {:+.6f}y {:+.6f}z {:+.6f} = 0'.format(*plane_abcd), sep='\n')
    plane_abcd = midsagittal_plane(left, midline, right, weighting_func=_sigmoid(.8, 5))
    print('plane function (weighted):', '{:+.6f}x {:+.6f}y {:+.6f}z {:+.6f} = 0'.format(*plane_abcd), '', sep='\n')

    # adding deformity
    left[0,0] -= 15
    midline[1,0] += 15

    print('with deformity:\n')
    plane_abcd = midsagittal_plane(left, midline, right)
    print('plane function (not weighted):', '{:+.6f}x {:+.6f}y {:+.6f}z {:+.6f} = 0'.format(*plane_abcd), sep='\n')
    plane_abcd = midsagittal_plane(left, midline, right, weighting_func=_sigmoid(.8, 5))
    print('plane function (weighted):', '{:+.6f}x {:+.6f}y {:+.6f}z {:+.6f} = 0'.format(*plane_abcd), '', sep='\n')


if __name__ == '__main__':
    _test()