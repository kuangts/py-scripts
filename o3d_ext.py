import numpy as np
import open3d as o3d

def remove_hidden_points(self):

    # use a total of 26 sources from various angles on expanded bounding box to cast rays
    # keep the hit triangles

    triangles = np.asarray(self.triangles)
    points = np.asarray(self.vertices)
    points = points[triangles,:].mean(axis=1)

    aabb = self.get_axis_aligned_bounding_box()
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
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self))
    ids_hit = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))['primitive_ids'].numpy()
    self.remove_triangles_by_index(np.setdiff1d(np.arange(len(self.triangles)), ids_hit))
    indices, hist, _ = self.cluster_connected_triangles()
    self.remove_triangles_by_index((np.asarray(indices)!=np.argmax(hist)).nonzero()[0])
    self.remove_unreferenced_vertices()

    return self

def cut_with_planes(self, normals, origins):
    vtx, fcs = np.asarray(self.vertices), np.asarray(self.triangles)
    
    for n,o in zip(normals, origins):
        d = (vtx - o).dot(n)
        fcs = fcs[np.any(d[fcs]>0, axis=1),:] # faces involved
        fcs_kept = fcs[np.all(d[fcs]>0, axis=1),:] # faces kept
        fcs_intersect = fcs[np.any(d[fcs]<0, axis=1),:] # faces to modify
        edg_intersect = np.vstack((fcs_intersect[:,[0,1]],fcs_intersect[:,[1,2]],fcs_intersect[:,[2,0]]))
        edg_intersect.sort(axis=1)
        edg_intersect, ind_inv = np.unique(edg_intersect, axis=0, return_inverse=True)
        vtx_new = np.empty((edg_intersect.shape[0],3))
        ind_kept = []
        for i,(e0,e1) in enumerate(edg_intersect):
            if np.sign(d[e0]) == np.sign(d[e1]):
                vtx_new[i,:] = vtx[e0,:]/2 + vtx[e1,:]/2
                if d[e0]>0:
                    ind_kept.append(i)
            else:
                vtx_new[i,:] = (vtx[e1,:]*d[e0] - vtx[e0,:]*d[e1])/(d[e0] - d[e1])
        ind_kept = np.union1d(np.unique(fcs_kept), np.asarray(ind_kept)+vtx.shape[0])
        ind_new = (ind_inv + vtx.shape[0]).reshape(3,-1).T

        f0 = np.hstack((ind_new[:,[2]],fcs_intersect[:,[0]],ind_new[:,[0]]))
        f1 = np.hstack((ind_new[:,[0]],fcs_intersect[:,[1]],ind_new[:,[1]]))
        f2 = np.hstack((ind_new[:,[1]],fcs_intersect[:,[2]],ind_new[:,[2]]))
        fc = np.hstack((ind_new[:,[2]],ind_new[:,[0]],ind_new[:,[1]]))
        vtx = np.vstack((vtx, vtx_new))
        fcs = np.vstack((fcs_kept, f0, f1, f2, fc))
        fcs = fcs[np.any(np.isin(fcs, ind_kept), axis=1),:]

    m = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vtx),
        triangles=o3d.utility.Vector3iVector(fcs)
    )
    m.remove_unreferenced_vertices()

    return m

def job_20221129():
    
    # on 2022/11/18 and on 2022/11/29
    # export models and landmarks from cass file
    # then find skin surface and cut by planes

    from CASS import CASS
    import glob, os, csv
    
    MyMesh = o3d.cpu.pybind.geometry.TriangleMesh
    setattr(MyMesh, 'remove_hidden_points', remove_hidden_points)
    setattr(MyMesh, 'cut_with_planes', cut_with_planes)


    for i in glob.glob(r'C:\data\nct\SH*\SH*.CASS')[0:1]:
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
        skin.remove_hidden_points().compute_vertex_normals()

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


        skin = skin.cut_with_planes(normals, origins)

        indices, hist, _ = skin.cluster_connected_triangles()
        skin.remove_triangles_by_index((np.asarray(indices)!=np.argmax(hist)).nonzero()[0])
        skin.compute_vertex_normals().compute_triangle_normals()
        o3d.visualization.draw_geometries([skin], mesh_show_back_face=True)
        o3d.io.write_triangle_mesh(os.path.join(dest_dir, 'skin.stl'), skin)



if __name__=='__main__':

    job_20221129()