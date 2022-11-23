import numpy as np
import open3d as o3d
import os

p4_surf_to_mirror = r'C:\Users\tmhtxk25\Downloads\skin_smooth_3mm_copy.stl'

# load surfaces
p4 = o3d.io.read_triangle_mesh(r'C:\Users\tmhtxk25\Downloads\skin_smooth_3mm.stl').remove_duplicated_vertices().compute_vertex_normals() # original
p4m = o3d.io.read_triangle_mesh(r'C:\Users\tmhtxk25\Downloads\P4_skin_smooth_3mm_Geo_Mirrored.stl').remove_duplicated_vertices().compute_vertex_normals().paint_uniform_color([.5,.2,.2]) # mirrored

# recreate the transformation
## first flip x coordinates of the original surface
flip_trans = np.eye(4)
flip_trans[0,0] = -1
p4.transform(flip_trans)
## Initial alignment
init_trans = np.eye(4)
init_trans[0,3] = -np.asarray(p4.vertices)[:,0].mean()*2
src = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(p4.vertices))
dst = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(p4m.vertices))
evaluation = o3d.pipelines.registration.evaluate_registration(
    src, dst, 5, init_trans)
## then icp to the mirrored position
icp_result = o3d.pipelines.registration.registration_icp(
    src, dst, 5, evaluation.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
p4.transform(icp_result.transformation)
## validate transformation
transformation = icp_result.transformation @ flip_trans
surface_to_mirror = o3d.io.read_triangle_mesh(p4_surf_to_mirror).remove_duplicated_vertices().compute_vertex_normals() # to mirror
surface_to_mirror.paint_uniform_color([.2,.5,.5])
surface_to_mirror.transform(transformation).compute_triangle_normals()
o3d.visualization.draw_geometries([surface_to_mirror, p4m], mesh_show_back_face=True)

# write result
np.savetxt(r'C:\Users\tmhtxk25\Downloads\p4_reflection_.txt', transformation)
x,y = os.path.splitext(p4_surf_to_mirror)
p4_surf_to_mirror_out = x+'_mirrored'+y
o3d.io.write_triangle_mesh(p4_surf_to_mirror_out, surface_to_mirror) # to mirror

