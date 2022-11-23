import SimpleITK as sitk
import open3d as o3d
import pyvista
from scipy.ndimage import binary_closing

def img2stl(img, threshold=None, new_spacing=None, reset_origin=False,
        smoothing_lambda=lambda m:m.filter_smooth_laplacian(number_of_iterations=5,lambda_filter=.5),
        decimation_lambda=lambda m:m.simplify_quadric_decimation(target_number_of_triangles=100_000)
        ):

    threshold_preset = {
        'bone':(1250,4095),
        'soft tissue':(324,1249),
        'all':(324,4095)
    }

    print(img.GetOrigin()) # delete later
    if new_spacing is not None:
        new_size = [int(a*b/c) for a,b,c in zip(img.GetSize(),img.GetSpacing(),new_spacing)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetTransform(sitk.Transform())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetDefaultPixelValue(-1024.)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputPixelType(sitk.sitkUnknown)
        resampler.SetUseNearestNeighborExtrapolator(False)
        img = resampler.Execute(img)

    if reset_origin:
        img.SetOrigin((0,0,0))

    models = []
    for v in threshold.items():
        if isinstance(v, str):
            if v in threshold_preset:
                v = threshold_preset[v]
            else:
                print(f'threshold for {v} is not found')
                continue

        arr = sitk.GetArrayFromImage((img>v[0])*(img<v[1]))
        seg_smooth = sitk.GetImageFromArray(binary_closing(arr)*1)
        seg_smooth.CopyInformation(seg)
        seg = seg_smooth

        gd = pyvista.UniformGrid(
            dims=seg.GetSize(),
            spacing=seg.GetSpacing(),
            origin=seg.GetOrigin(),
        )
        m = gd.contour([.5], sitk.GetArrayFromImage(seg).flatten(), method='marching_cubes')
        model = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(m.points),
                triangles=o3d.utility.Vector3iVector(m.faces.reshape(-1,4)[:,1:])
            )

        if smoothing_lambda is not None:
            model = smoothing_lambda(model)

        if decimation_lambda is not None:
            model = decimation_lambda(model)

        model.compute_triangle_normals()
        model.compute_vertex_normals()
        models.append(model)
    
    return models

