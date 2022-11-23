#! /usr/bin/env python 

required_pkg = ['numpy','scipy','pyvista','open3d']

import os, sys, pkg_resources
err = []
while len(required_pkg):
    try:
        pkg_resources.require(required_pkg.pop())
    except Exception as e:
        err.append(e)
else:
    if err:
        sys.exit('\n'.join([str(e) for e in err]))

import open3d as o3d
import numpy as np
from basic import *
from register import procrustes
import plotly.graph_objects as go
from image import Image, seg2surf
from skimage.measure import label
import image

# def reflection_plane_from_transformation(T):
#     t = -T[-1,:-1]/2
#     d = np.sum(t**2)**.5
#     plane = np.array((*t/d,d))
#     return plane

two_colors=[[.2,.5,.5],[.5,.2,.2]]

def show(geoms, colors=None):
    o3d.visualization.draw_geometries([
        (g.to_o3d() if isinstance(g, Poly) else g).paint_uniform_color(c) 
        for g,c in zip(geoms, colors if colors else np.random.normal(size=(len(geoms),3)))
    ]) 

def clip_yz_plane_with_bounding_box(pln, s):
    bd_min, bd_max = s.V.min(axis=0)-50, s.V.max(axis=0)+50
    yz = np.asarray([[0,0,0,0],[bd_min[1],bd_min[1],bd_max[1],bd_max[1]],[bd_min[2],bd_max[2],bd_min[2],bd_max[2]]]).T
    yz[:,0] = - (yz.dot(pln.n) + pln.d) / pln[0]
    pln_o3d = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(yz),
                    triangles=o3d.utility.Vector3iVector(np.asarray(((0,1,3),(0,3,2),(0,3,1),(0,2,3))))
                ).compute_vertex_normals().paint_uniform_color([.2,.2,.2])
    return pln_o3d


def ct2stl(ct):
    # # read this current case
    # seg = Image.from_gdcm(ct) + 1024
    # _n = 2 # to speed up
    # # seg = Image(image.Resample(seg,size=(256,256,128)))
    # resampler = image.ResampleImageFilter()
    # resampler.SetSize([a//_n for a in seg.GetSize()])
    # resampler.SetTransform(image.Transform())
    # resampler.SetInterpolator(image.sitkLinear)
    # resampler.SetOutputOrigin(seg.GetOrigin())
    # resampler.SetOutputSpacing([a*_n for a in seg.GetSpacing()])
    # resampler.SetOutputDirection(seg.GetDirection())
    # resampler.SetDefaultPixelValue(0.0)
    # resampler.SetOutputPixelType(image.sitkUnknown)
    # resampler.SetUseNearestNeighborExtrapolator(False)
    # seg = Image(resampler.Execute(seg))

    # s2 = seg2surf(Image((seg>324))) # tissue
    # s1 = seg2surf(Image((seg>1250))) # bone
    pass


def chest_sym_plane(s):

    s1 = s.copy()
    # find initial plane
    s1_mirrored = s1.copy()
    pln_yz = Plane.get(normal=np.asarray([1,0,0]), point=s1.V.centroid.flatten())
    s1_mirrored.V = pln_yz.reflect(s1_mirrored.V)

    src = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(s1_mirrored.V))
    tar = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(s1.V))
    evaluation = o3d.pipelines.registration.evaluate_registration(
        src, tar, 100, np.eye(4))
    print(evaluation)
    print(evaluation.transformation)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        src, tar, 3, evaluation.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    print(reg_p2p.transformation)


    # reg_p2p = o3d.pipelines.registration.registration_icp( src, tar, 15, evaluation.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # reg_p2p = o3d.pipelines.registration.registration_icp( src, tar, 10, reg_p2p.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # reg_p2p = o3d.pipelines.registration.registration_icp( src, tar, 5, reg_p2p.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # plane
    s1_mirrored.V = s1_mirrored.V.transform(reg_p2p.transformation.T)
    avg = s1.V/2 + s1_mirrored.V/2
    pln = Plane.pca_fit(avg)
    return pln


if __name__=='__main__':

    if len(sys.argv) == 1:

        cage = Poly.read(r"c:\data\chest\p4_cage.stl")
        skin = Poly.read(r"c:\data\chest\p4_skin.stl")
        pln = chest_sym_plane(cage)
        pln_o3d = clip_yz_plane_with_bounding_box(pln, skin)
        cage_mirror = Poly(V=pln.reflect(cage.V), F=cage.F[:,[0,2,1]])
        skin_mirror = Poly(V=pln.reflect(skin.V), F=skin.F[:,[0,2,1]])

        cage = cage.to_o3d()
        skin = skin.to_o3d()
        pln.segment(cage)
        
        o3d.visualization.draw_geometries([cage, cage_mirror.to_o3d(), pln_o3d])
        o3d.visualization.draw_geometries([skin.paint_uniform_color([.2,.5,.5]), skin_mirror.to_o3d().paint_uniform_color([.5,.2,.2]), pln_o3d])

    else:

        if isinstance (sys.argv[1], str) and sys.argv[1] == 'test' :
            stl = o3d.io.read_triangle_mesh(r".\test\cage.stl")
            s1 = Poly(V=stl.vertices, F=stl.triangles)
            pln = Plane.pca_fit(s1.V)
            s2 = s1.copy()
            s2.V = pln.reflect(s2.V)
            o1 = s1.to_o3d()
            o2 = s2.to_o3d()
            pln.segment(o1)
            pln.segment(o2)
            o3d.visualization.draw_geometries([o1,o2])
            sys.exit(0)
        else:
            ct = sys.argv[1]
            chest_sym_plane(ct)
        
    pass