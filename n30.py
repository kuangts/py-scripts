from tools.mesh import *
from tools.ui import Window
from scipy.interpolate import RBFInterpolator
from vtkmodules.vtkIOExport import vtkVRMLExporter

file = r'C:\data\meshes\n0030\test\hexmesh_open_test.inp'
N,E = read_inp(file)
vol = local_volume(N,E)
elem_id = np.any(vol<0, axis=1)

G = calculate_grid(N,E)
G3D = grid_3d_from_flat(G)
IND = np.mgrid[0:G3D.shape[0],0:G3D.shape[1],0:G3D.shape[2]]

# delete upper left 10x10 and use neighboring 10 elements to regenerate

IND_find = np.zeros(IND.shape[1:], dtype=bool)
IND_keep = np.zeros(IND.shape[1:], dtype=bool)

IND_find[:10,:,-10:] = True
IND_keep[:20,:,-20:] = True
IND_keep = IND_keep & ~IND_find

N[G3D[IND_find],:] = RBFInterpolator(IND[:,IND_keep].T, N[G3D[IND_keep]])(IND[:,IND_find].T)

with open(r'C:\data\meshes\n0046\test\hexmesh_open_fix_n30.inp', 'w') as f:
    write_inp(f, N, E)

N,E = read_inp(r'C:\data\meshes\n0046\test\hexmesh_open_fix_n30.inp')

w = Window()

F = boundary_faces(E[elem_id])
polyd = polydata_from_numpy(N,F)
polyd = clean_polydata(polyd)
actor = polydata_actor(polyd, Color=[.8,.2,.2])
w.renderer.AddActor(actor)

F = boundary_faces(np.delete(E, elem_id, axis=0))
polyd = polydata_from_numpy(N,F, lines=True)
polyd = clean_polydata(polyd)
actor = polydata_actor(polyd, Color=[.8,.6,.6], Opacity=.8)
w.renderer.AddActor(actor)

w.render_window.Render()
exporter = vtkVRMLExporter()
exporter.SetInput(w.render_window)
exporter.SetFileName(r'C:\data\meshes\n0030\test\hexmesh_open_fix.wrl')
exporter.Update()
exporter.Write()

w.start()

