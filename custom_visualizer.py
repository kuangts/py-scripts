import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import sys


class Plane_App:

    def __init__(self, file=r"c:\data\chest\p4_cage.stl"):
        self.model = o3d.io.read_triangle_mesh(file)
        self.model.compute_vertex_normals()
        self.center = np.asarray(self.model.vertices).mean(axis=0)
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"


        app = o3d.visualization.gui.Application.instance
        self.window = app.create_window("Example", 1024, 768)
        self.scnwid = o3d.visualization.gui.SceneWidget()
        self.scnwid.scene = o3d.visualization.rendering.Open3DScene(self.window.renderer)
        self.scnwid.scene.add_geometry('obj', self.model, mtl)
        self.scnwid.set_on_mouse(self._on_mouse)
        self.window.add_child(self.scnwid)
        self.scnwid.scene.camera.look_at(self.center, self.center-[0,100,0], np.array((0.,0.,1.)))

        vertical_field_of_view = 15.0  # between 5 and 90 degrees
        near_plane = 0.1
        far_plane = 50.0
        fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
        self.scnwid.scene.camera.set_projection(vertical_field_of_view, 1., near_plane, far_plane, fov_type)
        self.scnwid.center_of_rotation
    def _on_mouse(self, event):
        try:
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
                x = event.x - self.scnwid.frame.x
                y = event.y - self.scnwid.frame.y
                print(x,y)
                self.model.vertices = o3d.utility.Vector3dVector(self.model.vertices + np.asarray([[10,20,30]]))
                # self.scnwid.scene.scene.update_geometry()
                return gui.Widget.EventCallbackResult.HANDLED
            elif event.type == gui.MouseEvent.Type.DRAG and event.is_modifier_down(gui.KeyModifier.CTRL):
                x = event.x
                y = event.y
                print(x,y)
            return gui.Widget.EventCallbackResult.IGNORED
            
        except Exception as e:
            print(e)
            return gui.Widget.EventCallbackResult.HANDLED


def main():
    app = gui.Application.instance
    try:
        app.initialize()
        Plane_App()
        app.run()
    except Exception as e:
        print(e)
        app.quit()
        sys.exit()

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit()

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.set_full_screen(True)
# vis.add_geometry(m)
# view = vis.get_view_control()
# ren = vis.get_render_option()
# ren.show_coordinate_frame = True
# ren.mesh_show_back_face = True
# view.set_up(np.asarray([0.,1.,0.]))
# view.set_front(np.asarray([0.,0.,1.]))
# # view.set_lookat(np.asarray(m.vertices).mean(axis=0)+np.asarray([0.,50.,0.]))
# vis.run()
# vis.destroy_window()