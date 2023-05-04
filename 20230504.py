from soft_tissue_prediction_copy_data import *
import vtk

root = r'P:\20230501'
cases = os.listdir(root)

for c in cases:
    print(c)
    case_dir = os.path.join(root, c)
    file = lambda x: os.path.join(case_dir, x)
    segs = ['di', 'diL', 'diR', 'le']
    if os.path.exists(file('pre_gen.stl')):
         segs.append('gen')

    for seg in segs:
        seg = 'pre_' + seg + '.stl'
        s = read_polydata(file(seg))
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(s)
        cleaner.Update()
        write_polydata(cleaner.GetOutput(), file(seg))


