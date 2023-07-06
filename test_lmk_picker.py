import os, glob
from tools import *


if __name__ == '__main__':

    root = r'C:\data\pre-post-paired-40-send-1122'
    ncase = 'n0001'
    nifti_pre = glob.glob(os.path.join(root, ncase, '*-pre.nii.gz'))
    assert len(nifti_pre) == 1, 'check patient dir'
    nifti_pre = nifti_pre[0]
    img = imagedata_from_nifti(nifti_pre)
    mask = threshold_imagedata(img, foreground_threshold)
    polyd_pre = polydata_from_mask(mask)
    lmk_pre = os.path.join(root, ncase, 'skin_landmark.txt')
    lmk_coords = np.genfromtxt(lmk_pre)
    lmk_color = numpy_to_vtk_(np.array([[1,.5,.5]]*lmk_coords.shape[0]))
    lmk_color.SetName('colors')
    glyph_filter = vtkGlyph3D()
    src = vtkSphereSource()
    src.Update()
    glyph_filter.SetSourceConnection(src.GetOutputPort())
    lmk = numpy_to_vtkpoints_(lmk_coords)
    inputd = vtkPointSet()
    inputd.SetPoints(lmk)
    inputd.GetPointData().SetVectors(lmk_color)
    inputd.GetPointData().SetActiveVectors('colors')
    glyph_filter.SetInputData(inputd)

    sel = PolygonalSurfaceGlyphSelector(glyph_filter)
    sel.add_pick_polydata(polyd_pre)
    sel.start()