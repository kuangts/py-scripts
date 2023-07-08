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

    glyph_filter = vtkGlyph3D()
    src = vtkSphereSource()
    src.SetRadius(1.0)
    src.Update()
    glyph_filter.SetSourceConnection(src.GetOutputPort())
    inputd = vtkPointSet()

    lmk_pre = os.path.join(root, ncase, 'skin_landmark.txt')
    lmk_coords = np.genfromtxt(lmk_pre)
    lmk = numpy_to_vtkpoints_(lmk_coords)
    inputd.SetPoints(lmk)
    glyph_filter.SetInputData(inputd)

    sel = PolygonalSurfaceGlyphSelector(glyph_filter)
    sel.add_pick_polydata(polyd_pre, Color=[.5,.5,.5])

    sel.start()