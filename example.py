import os
import open3d as o3d
import SimpleITK as sitk
from scipy.ndimage import binary_closing
import pyvista
import numpy as np


def remove_hidden_points(mesh_o3d):

    # use a total of 26 sources from various angles on expanded bounding box to cast rays
    # keep the hit triangles

    triangles = np.asarray(mesh_o3d.triangles)
    points = np.asarray(mesh_o3d.vertices)
    points = points[triangles,:].mean(axis=1)

    aabb = mesh_o3d.get_axis_aligned_bounding_box()
    bb_cen = aabb.get_center()
    bb_ext = aabb.get_extent()
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bb_cen-bb_ext,
        max_bound=bb_cen+bb_ext) # enlarge the bound box 2 fold

    # add 8 corners, centers of 6 faces, and midpoints of 12 sides of the bounding box
    # where rays are cast

    srcs = np.asarray(aabb.get_box_points())
    srcs = { tuple((srcs[i]/2 + srcs[j]/2).round(6).tolist())\
        for i in range(8) for j in range(8) if i <= j }
    srcs.remove(tuple(aabb.get_center().round(6).tolist()))

    rays = np.array(()).reshape(0,6)
    for s in srcs:
        ray_dir = points - s
        ray_dir = ray_dir / np.sum(ray_dir**2, axis=1, keepdims=True)**.5
        rays = np.vstack((rays, 
            np.hstack((np.tile(s, (ray_dir.shape[0],1)), ray_dir))
        ))

    # cast rays

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d))
    ids_hit = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))['primitive_ids'].numpy()
    mesh_o3d.remove_triangles_by_index(np.setdiff1d(np.arange(len(mesh_o3d.triangles)), ids_hit))
    indices, hist, _ = mesh_o3d.cluster_connected_triangles()
    mesh_o3d.remove_triangles_by_index((np.asarray(indices)!=np.argmax(hist)).nonzero()[0])
    mesh_o3d.remove_unreferenced_vertices()

    return mesh_o3d



if __name__=='__main__':

    os.chdir(r'Z:\pre-post-paired-40-send-1122')
    img = sitk.ReadImage(r'.\n0001\20110425-pre.nii.gz')
    seg = sitk.ReadImage(r'.\n0001\pre-seg.nii.gz')
    gd = pyvista.UniformGrid(
        dims=img.GetSize(),
        spacing=img.GetSpacing(),
        origin=img.GetOrigin(),
    )

    arr = sitk.GetArrayFromImage(img)
    arr = (arr>=324) & (arr<=4095)
    skin = gd.contour([.5], arr.flatten(), method='marching_cubes')
    skin = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(skin.points),
            triangles=o3d.utility.Vector3iVector(skin.faces.reshape(-1,4)[:,1:])
        )

    arr = sitk.GetArrayFromImage(seg==1)
    maxi = gd.contour([.5], arr.flatten(), method='marching_cubes')
    maxi = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(maxi.points),
            triangles=o3d.utility.Vector3iVector(maxi.faces.reshape(-1,4)[:,1:])
        )

    arr = sitk.GetArrayFromImage(seg==2)
    mand = gd.contour([.5], arr.flatten(), method='marching_cubes')
    mand = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mand.points),
            triangles=o3d.utility.Vector3iVector(mand.faces.reshape(-1,4)[:,1:])
        )

    skin = remove_hidden_points(skin)

    # skin.filter_smooth_laplacian(number_of_iterations=5,lambda_filter=.5)
    # maxi.filter_smooth_laplacian(number_of_iterations=5,lambda_filter=.5)
    # mand.filter_smooth_laplacian(number_of_iterations=5,lambda_filter=.5)

    # skin.simplify_quadric_decimation(target_number_of_triangles=100_000)
    # maxi.simplify_quadric_decimation(target_number_of_triangles=100_000)
    # mand.simplify_quadric_decimation(target_number_of_triangles=100_000)

    skin.compute_triangle_normals()
    skin.compute_vertex_normals()
    maxi.compute_triangle_normals()
    maxi.compute_vertex_normals()
    mand.compute_triangle_normals()
    mand.compute_vertex_normals()

    o3d.visualization.draw_geometries([skin, mand, maxi])

