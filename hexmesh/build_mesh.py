import os, sqlite3, csv, subprocess, shutil
from tempfile import gettempdir
import numpy as np
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from scipy.interpolate import RBFInterpolator

case_name = 'kx01.cass'
d_ = os.getcwd()

# os.chdir(os.path.dirname(os.path.realpath(__file__)))
location = r'C:\AA.Release\CASS_POST.db'

with sqlite3.connect(location) as con:
    lib = con.execute(f'SELECT * FROM Landmark_Library').fetchall()
    id2label = {int(x[0]):x[4] for x in lib}
    id2group = {int(x[0]):x[3] for x in lib}

# d = gettempdir()
d = r'C:\py-scripts\hexmesh\temp'
os.makedirs(d, exist_ok=True)
os.chdir(d)

subprocess.run(['unrar', 'e', os.path.join(d_, 'hexmesh','cases', case_name), 'Measure_Data_Info_new.bin', 'Three_Data_Info.bin'])
with open('Measure_Data_Info_new.bin') as f:
    ldmk = f.read().strip(';').split(';')
    ldmk = {int(x[0]):x[6:9] for x in [x.split(',') for x in ldmk]}

os.remove('Measure_Data_Info_new.bin')

landmarks = dict(maxilla={}, mandible={}, skin={})
for k,v in ldmk.items():
    if id2group[k] in ['Cranium','Maxilla','Midface','Upper Tooth']:
        landmarks['maxilla'][id2label[k]] = v
    elif id2group[k] in ['Mandible','Lower Tooth']:
        landmarks['mandible'][id2label[k]] = v
    elif id2group[k] == 'Face':
        landmarks['skin'][id2label[k]] = v

'''
# for kx01.CASS
len(landmarks['mandible'])
60
len(landmarks['maxilla'])
76
len(landmarks['skin'])
72
len(ldmk)
208'''

# with open('Three_Data_Info.bin') as f:
#     info = f.read().strip(';').split(';')
#     object_names = [x.split(',')[0] for x in info[1:]]

# os.remove('Three_Data_Info.bin')

# maxilla_id = None
# mandible_id = None
# skin_id = None
# skin_post_id = None

# for i,name in enumerate(object_names):
#     if name.lower() == 'mandible (whole)':
#         mandible_id = i
#         continue
#     elif 'mandible' in name.lower() and 'whole' in name.lower():
#         mandible_id = i
#         continue
#     elif mandible_id is None and 'mandible' in name.lower():
#         mandible_id = i
#         continue

#     if name.lower() == 'skull (whole)':
#         maxilla_id = i
#         continue
#     elif 'skull' in name.lower() and 'whole' in name.lower():
#         maxilla_id = i
#         continue
#     elif maxilla_id is None and 'skull' in name.lower():
#         maxilla_id = i
#         continue

#     if name.lower() == 'ct soft tissue':
#         skin_id = i
#         continue
    
#     elif name.lower() == 'post ct soft tissue':
#         skin_post_id = i
#         continue

#     elif 'soft' in name.lower() and 'tissue' in name.lower():
#         if 'post' in name.lower():
#             skin_post_id = i
#         else:
#             skin_id = i
#         continue
    

# if any((mandible_id is None, maxilla_id is None, skin_id is None, skin_post_id is None)):
#     print(f'check {case_name}')
# else:
#     subprocess.run(['unrar', 'e', os.path.join(d_,'hexmesh', 'cases',case_name), f'{mandible_id}.stl', f'{maxilla_id}.stl', f'{skin_id}.stl', f'{skin_post_id}.stl'])

# shutil.move(f'{mandible_id}.stl', 'mandible.stl')
# shutil.move(f'{maxilla_id}.stl', 'maxilla.stl')
# shutil.move(f'{skin_id}.stl', 'skin.stl')
# shutil.move(f'{skin_post_id}.stl', 'skin_post.stl')

os.chdir(d_)


template_landmarks = {}
with open(r'.\hexmesh\template\maxilla.csv') as f:
    template_landmarks['maxilla'] = {x[0]:x[1:] for x in csv.reader(f)}
with open(r'.\hexmesh\template\mandible.csv') as f:
    template_landmarks['mandible'] = {x[0]:x[1:] for x in csv.reader(f)}
with open(r'.\hexmesh\template\skin.csv') as f:
    template_landmarks['skin'] = {x[0]:x[1:] for x in csv.reader(f)}

print(template_landmarks)

mandible_labels = np.intersect1d(
    list(landmarks['mandible'].keys()),
    list(template_landmarks['mandible'].keys())
)
maxilla_labels = np.intersect1d(
    list(landmarks['maxilla'].keys()),
    list(template_landmarks['maxilla'].keys())
)
skin_labels = np.intersect1d(
    list(landmarks['skin'].keys()),
    list(template_landmarks['skin'].keys())
)

landmark_mandible = np.array([landmarks['mandible'][k] for k in mandible_labels], dtype=float)
landmark_maxilla = np.array([landmarks['maxilla'][k] for k in maxilla_labels], dtype=float)
landmark_skin = np.array([landmarks['skin'][k] for k in skin_labels], dtype=float)

landmark_mandible_t = np.array([template_landmarks['mandible'][k] for k in mandible_labels], dtype=float)
landmark_maxilla_t = np.array([template_landmarks['maxilla'][k] for k in maxilla_labels], dtype=float)
landmark_skin_t = np.array([template_landmarks['skin'][k] for k in skin_labels], dtype=float)

interp_mandible = RBFInterpolator(landmark_mandible_t, landmark_mandible)
interp_maxilla = RBFInterpolator(landmark_maxilla_t, landmark_maxilla)
interp_skin = RBFInterpolator(landmark_skin_t, landmark_skin)

reader = vtkSTLReader()
reader.SetFileName(r'.\hexmesh\template\mandible.stl')
reader.Update()
stl_mandible = reader.GetOutput()
vertices_mandible_t = vtk_to_numpy(stl_mandible.GetPoints().GetData())
reader = vtkSTLReader()
reader.SetFileName(r'.\hexmesh\template\maxilla.stl')
reader.Update()
stl_maxilla = reader.GetOutput()
vertices_maxilla_t = vtk_to_numpy(stl_maxilla.GetPoints().GetData())
reader = vtkSTLReader()
reader.SetFileName(r'.\hexmesh\template\skin.stl')
reader.Update()
stl_skin = reader.GetOutput()
vertices_skin_t = vtk_to_numpy(stl_skin.GetPoints().GetData())

vertices_mandible_t = interp_mandible(vertices_mandible_t)
vertices_maxilla_t = interp_maxilla(vertices_maxilla_t)
vertices_skin_t = interp_skin(vertices_skin_t)



stl_mandible.GetPoints().SetData(
    numpy_to_vtk(vertices_mandible_t, True)
)
stl_maxilla.GetPoints().SetData(
    numpy_to_vtk(vertices_maxilla_t, True)
)
stl_skin.GetPoints().SetData(
    numpy_to_vtk(vertices_skin_t, True)
)

writer = vtkSTLWriter()
writer.SetFileName(r'.\hexmesh\temp\mandible_tps.stl')
writer.SetInputData(stl_mandible)
writer.Update()
writer.Write()
writer = vtkSTLWriter()
writer.SetFileName(r'.\hexmesh\temp\maxilla_tps.stl')
writer.SetInputData(stl_maxilla)
writer.Update()
writer.Write()
writer = vtkSTLWriter()
writer.SetFileName(r'.\hexmesh\temp\skin_tps.stl')
writer.SetInputData(stl_skin)
writer.Update()
writer.Write()




os.chdir(d_)
# shutil.rmtree(d)
