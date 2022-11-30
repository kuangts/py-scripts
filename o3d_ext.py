import sys, glob, os, csv
import numpy as np
import open3d as o3d
import trimesh
from CASS import CASS

def remove_hidden_points(mesh_o3d):

    # use a total of 26 points from various angles on bounding box to cast rays
    # keep the hit triangles

    m = mesh_o3d
    triangles = np.asarray(m.triangles)
    points = np.asarray(m.vertices)
    points = points[triangles,:].mean(axis=1)

    aabb = m.get_axis_aligned_bounding_box()
    bb_cen = aabb.get_center()
    bb_ext = aabb.get_extent()
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bb_cen-bb_ext,
        max_bound=bb_cen+bb_ext) # enlarge the bound box 2 fold

    # add eight corners, centers of six faces, and midpoints of 12 sides of the bounding box
    # where rays are cast

    srcs = np.asarray(aabb.get_box_points())
    srcs = { tuple((srcs[i]/2 + srcs[j]/2).round(6).tolist())\
        for i in range(8) for j in range(8) if i <= j }
    srcs.remove(tuple(aabb.get_center().round(6).tolist()))

    # cast rays from these srcs to each triangle

    rays = np.array(()).reshape(0,6)
    for s in srcs:
        ray_dir = points - s
        ray_dir = ray_dir / np.sum(ray_dir**2, axis=1, keepdims=True)**.5
        rays = np.vstack((rays, 
            np.hstack((np.tile(s, (ray_dir.shape[0],1)), ray_dir))
        ))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    ids_hit = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))['primitive_ids'].numpy()
    m.remove_triangles_by_index(np.setdiff1d(np.arange(len(m.triangles)), ids_hit))

    return m

def cut_with_planes(mesh_o3d, normals, origins):
    vtx, fcs = np.asarray(mesh_o3d.vertices), np.asarray(mesh_o3d.triangles)
    for n,o in zip(normals, origins):
        vtx, fcs = trimesh.intersections.slice_faces_plane(vertices=vtx, faces=fcs, plane_normal=n, plane_origin=o)
    m = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vtx),
        triangles=o3d.utility.Vector3iVector(fcs),
    )
    return m


def job_20221129():
    
    # on 2022/11/18 and on 2022/11/29
    # export models and landmarks from cass file
    # then find skin surface and cut by planes
    
    for i in glob.glob(r'C:\data\nct\SH*\SH*.CASS'):
        id = os.path.basename(os.path.dirname(i))
        print(id)
        dest_dir = rf'c:\data\nct\{id}'
        with CASS(i) as f:
            f.models(['Skull (whole)', 'Mandible (whole)'], dest_dir, override=False)
            skin = f.models(['CT Soft Tissue'])[0]
            lmk = f.landmarks()
            with open(os.path.join(dest_dir, 'lmk.csv'),'w', newline='') as f:
                csv.writer(f).writerows([[k,*map(lambda x:round(x,3),v)] for k,v in lmk.items()])


        lmk = {k:np.array(v) for k,v in lmk.items()}
        remove_hidden_points(skin)
        # cut by three plane plus one slightly above the floor
        normals = [
            np.cross(lmk['Po-R'] - lmk['Or-L'], lmk['Po-L'] - lmk['Or-R']),
            np.cross(np.cross(lmk['Po-R'] - lmk['Or-L'], lmk['Po-L'] - lmk['Or-R']), np.array(lmk['Go-L']) - np.array(lmk['Go-R'])),
            np.cross(lmk['Go-L'] - lmk['C'], lmk['Go-R'] - lmk['C']),
            np.array((0,0,1)),
        ]
        origins = [
            (lmk['Po-R']+lmk['Or-R']+lmk['Po-L']+lmk['Or-L'])/4,
            (np.array(lmk['Go-L']) + np.array(lmk['Go-R']))/2,
            np.array(lmk['C']),
            np.asarray(skin.vertices).min(axis=0) + np.array((0,0,.1))
        ]
        skin = cut_with_planes(skin, normals, origins)

        indices, hist, _ = skin.remove_duplicated_vertices().cluster_connected_triangles()
        skin.remove_triangles_by_index((np.asarray(indices)!=np.argmax(hist)).nonzero()[0])
        skin.compute_vertex_normals().compute_triangle_normals()
        # o3d.visualization.draw_geometries([skin], mesh_show_back_face=True)
        o3d.io.write_triangle_mesh(os.path.join(dest_dir, 'skin.stl'), skin)

