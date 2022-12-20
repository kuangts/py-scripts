import os
import sys
import glob
from enum import Enum
from tkinter import messagebox as msgbox
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import pyvista
import pyperclip

def get_material(mat_name):

    mat_params = {
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    mat = o3d.visualization.rendering.MaterialRecord()
    for key, val in mat_params[mat_name].items():
        setattr(mat, "base_" + key, val)
    mat.shader = "defaultLit"
    return mat


np.set_printoptions(precision=6)


class AppWindow:
    
    MENU_OPEN = 1
    MENU_SAVE = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11

    class Mode(Enum): 
        View = 0
        Plane = 1
        def switch(self):
            return self.__class__(1 - self.value)

    
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_val):
        assert isinstance(new_val, self.Mode)
        if hasattr(self, '_mode') and self._mode == new_val:
            return
        else:
            setattr(self, '_mode', new_val)
            if new_val == self.Mode.View:
                self._set_mouse_mode_view()
            elif new_val == self.Mode.Plane:
                self._set_mouse_mode_plane()

    def _set_mouse_mode_view(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self._transformation_panel.visible = False
        # self.color_mesh_distance()
        if self._scene.scene.has_geometry('__MIRRORED__'):
            self._scene.scene.show_geometry('__MIRRORED__', True)

    def _set_mouse_mode_plane(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)
        self._transformation_panel.visible = True
        self.mesh.paint_uniform_color([.5,.5,.5])
        if self._scene.scene.has_geometry('__MIRRORED__'):
            self._scene.scene.show_geometry('__MIRRORED__', True)

    def _on_key(self, event):
        if event.type == gui.KeyEvent.Type.DOWN and event.key == 32:
            self.mode = self.mode.switch()
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse(self, event):
        if event.is_button_down(o3d.visualization.gui.MouseButton.LEFT):
            view_mat = self._scene.scene.camera.get_view_matrix()
            proj_mat = self._scene.scene.camera.get_projection_matrix()
            modl_mat = self._scene.scene.camera.get_model_matrix()

            # print(view_mat.round(6))
            # print(proj_mat.round(6))
            # print(modl_mat.round(6))

            # print((view_mat@modl_mat).round(6))
            # print((modl_mat@view_mat).round(6))

        # pass on viewing mode
        if self.mode==self.Mode.View:
            return gui.Widget.EventCallbackResult.IGNORED
        # pass on model not loaded
        if not self._scene.scene.has_geometry('__PLANE__'):
            for l in self._trans_list:
                l.text = ' '.join(['...n/a...']*4)
            return gui.Widget.EventCallbackResult.HANDLED

        # t = self._scene.center_of_rotation
        # T0 = np.eye(4)
        # T1 = np.eye(4)
        # T0[:-1,-1] = -t[:]
        # T1[:-1,-1] = t[:]
        # T = T1 @ T @ T0

        # for l,x in zip(self._trans_list,T):
        #     l.text = ' '.join(f'{i:+0.2e}' for i in x)


        if self.mode == self.Mode.Plane and event.type == gui.MouseEvent.Type.DRAG:
            pass




        # layout window once button is up
        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            pln = self.get_plane_equation()
            T = np.eye(4) - 2 * np.array([[*pln[:-1],0]]).T @ pln[None,:]
            self._scene.scene.set_geometry_transform('__MIRRORED__', T)
            self._scene.scene.set_geometry_transform('__MESH__', np.eye(4))
            self.window.set_needs_layout()
        return gui.Widget.EventCallbackResult.HANDLED


    def color_mesh_distance(self):
        if not self._scene.scene.has_geometry('__PLANE__'):
            return
        pln = self.get_plane_equation()
        T = np.eye(4) - 2 * np.array([[*pln[:-1],0]]).T @ pln[None,:]
        vtk_faces = lambda f: np.hstack((np.tile(3,(f.shape[0],1)), f.astype(int))).flatten()
        h1 = o3d.geometry.TriangleMesh(self.mesh_mirrored).transform(T)
        h1 = pyvista.PolyData(np.asarray(h1.vertices), vtk_faces(np.asarray(h1.triangles)))
        h0 = pyvista.PolyData(np.asarray(self.mesh.vertices), vtk_faces(np.asarray(self.mesh.triangles)))
        _, closest_points = h1.find_closest_cell(h0.points, return_closest_point=True)
        d = closest_points - h0.points
        r = np.arange(100)[:,None]/99
        cmap = r @ np.asarray([[1,0,0]]) + (1-r) @ np.asarray([[0,0,1]])
        d = np.sum(d**2, axis=1)**.5
        d[d>10] = 10
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(cmap[(d/10*99).astype(int)])
        self._scene.scene.scene.update_geometry('__MESH__', self.mesh, o3d.visualization.rendering.Scene.UPDATE_COLORS_FLAG)

    def get_plane_equation(self):
        if self._scene.scene.has_geometry('__PLANE__'):
            T = self._scene.scene.get_geometry_transform('__PLANE__')
            center = np.asarray(self.plane.vertices).mean(axis=0) 
            center = (T @ np.asarray([[*center, 1]]).T).flat[:-1]
            vtx = np.asarray(self.plane.vertices)
            normal = (T @ np.asarray([[*(vtx[1]-vtx[0]),0.]]).T).flat[:-1]
            pln = np.array([ *normal, -center.dot(normal) ])
            return pln
        return None


    def load(self, path):
        self._scene.scene.clear_geometry()
        # triangle mesh
        self.mesh = o3d.io.read_triangle_mesh(path)
        self.mesh.compute_vertex_normals()
        self.mesh_mirrored = o3d.geometry.TriangleMesh(self.mesh).paint_uniform_color((.2,.5,.5))
        # initial plane of symmetry
        src = o3d.geometry.PointCloud(points=self.mesh_mirrored.vertices)
        mat_refl = np.eye(4)
        mat_refl[0,0] = -1
        src = src.transform(mat_refl)
        tar = o3d.geometry.PointCloud(points=self.mesh.vertices)
        trans_init = np.eye(4)
        trans_init[0,-1] = (np.asarray(tar.points)-np.asarray(src.points))[:,0].mean()
        reg = o3d.pipelines.registration.evaluate_registration(
            src, tar, 100, trans_init)

        reg = o3d.pipelines.registration.registration_icp(
            src, tar, 20, reg.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

        reg = o3d.pipelines.registration.registration_icp(
            src, tar, 10, reg.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

        # plane
        src_t = o3d.geometry.PointCloud(src)
        src_t = src_t.transform(reg.transformation)
        avg = np.asarray(tar.points)/2 + np.asarray(src_t.points)/2
        cen = avg.mean(axis=0, keepdims=True)
        avg = avg - cen
        _, _, W = np.linalg.svd(avg.T@avg)
        T_plane =  np.block([[W[::-1,:],cen.T],[0,0,0,1]])

        T_reflect = np.eye(4) - 2 * np.array([[*W[-1],0]]).T @ np.asarray([[ *W[-1], -W[-1].dot(cen.flatten()) ]])

        # initial plane of symmetry
        vtx = np.asarray(self.mesh.vertices)
        bd_max, bd_min = vtx.max(axis=0, keepdims=True), vtx.min(axis=0, keepdims=True)
        extent = bd_max - bd_min
        plane_size = (1., extent.flat[1]*1.2, extent.flat[2]*1.2)
        self.plane = o3d.geometry.TriangleMesh.create_box(*plane_size)
        self.plane.vertices = o3d.utility.Vector3dVector(np.asarray(self.plane.vertices) - .5*np.asarray(plane_size))
        self.plane.paint_uniform_color([.5,.5,.2])
        self.plane.compute_vertex_normals()
        
        
        # add geometry to the scene
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("__PLANE__", self.plane, get_material('Metal (smoother)'))
        self._scene.scene.add_geometry("__MESH__", self.mesh, get_material('Plastic'))
        self._scene.scene.add_geometry("__MIRRORED__", self.mesh_mirrored, get_material('Plastic'))

        self._scene.center_of_rotation = cen.flatten()
        self._scene.scene.set_geometry_transform('__PLANE__', T_plane)
        self._scene.scene.set_geometry_transform('__MIRRORED__', T_reflect)
        self._scene.scene.set_geometry_transform('__MESH__', np.eye(4))

        if self.mode == self.Mode.View:
            self._scene.scene.show_geometry("__MIRRORED__", False)
        # camera
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())
        self._save_button.enabled = True


    def __init__(self, width, height):

        self.window = gui.Application.instance.create_window(
            "Open3D", width, height)
        w = self.window  # to make the code more concise

        lighting = {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        }

        for k,v in lighting.items():
            setattr(self, k, v)

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_mouse(self._on_mouse)
        self._scene.set_on_key(self._on_key)
        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._arcball_button = gui.Button("View")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_view)
        self._plane_button = gui.Button("Plane")
        self._plane_button.horizontal_padding_em = 0.5
        self._plane_button.vertical_padding_em = 0
        self._plane_button.set_on_clicked(self._set_mouse_mode_plane)
        view_ctrls.add_child(gui.Label("Mouse controls"))

        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._plane_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._mesh_color = gui.ColorEdit()
        self._mesh_color.set_on_value_changed(self._on_mesh_color)
        self._mesh_color.color_value = o3d.visualization.gui.Color(.5,.5,.5, 1)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._mesh_color)
        view_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axes)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        self._transformation_panel = gui.Vert(
                    0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        self._trans_list = [gui.Label(' ') for _ in range(4)]
        for l in self._trans_list:
            l.background_color = gui.Color(0,0,0,0)
            l.text_color = gui.Color(.1,.1,.1,1)
            l.font_id = mono_id
            l.text = ' '.join(['.'*9]*4)

        self._save_button = gui.Button("Save")
        self._save_button.horizontal_padding_em = 0.5
        self._save_button.vertical_padding_em = 0.5
        self._save_button.set_on_clicked(self._on_menu_save)
        self._save_button.enabled = False
        _trans_title = gui.Label('Transformation')
        self._transformation_panel.add_child(_trans_title)
        for i in self._trans_list:
            self._transformation_panel.add_child(i)
        self._transformation_panel.add_fixed(separation_height)
        self._transformation_panel.add_child(self._save_button)
        self._transformation_panel.background_color = o3d.cpu.pybind.visualization.gui.Color(0,0,0,0)
        self._transformation_panel.visible = False
        _trans_title.text_color = o3d.cpu.pybind.visualization.gui.Color(0,0,0,1)
        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.add_child(self._transformation_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Save Transformation Matrix...", AppWindow.MENU_SAVE)
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Panel", AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, False)
            self._settings_panel.visible = False
            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Settings", settings_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(AppWindow.MENU_SAVE, self._on_menu_save)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS, self._on_menu_toggle_settings_panel)
        # ----

        # self.materials = {}
        # self.materials['geometry'] = o3d.visualization.rendering.MaterialRecord()
        # c = self._mesh_color.color_value
        # self.materials['geometry'].base_color = (c.red, c.green, c.blue, c.alpha)
        # self.materials['plane'] = o3d.visualization.rendering.MaterialRecord()
        # self.materials['plane'].base_color = [.5,.5,.2,1]

        self.mode = self.Mode.View 

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 20 * layout_context.theme.font_size
        preferred_size = self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
        height1 = min(r.height, preferred_size.height)
        preferred_size = self._transformation_panel.calc_preferred_size(layout_context, gui.Widget.Constraints())
        height2 = min(r.height, preferred_size.height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height1)
        self._transformation_panel.frame = gui.Rect(r.get_right() - width, r.get_bottom()-height2, width,
                                              height2)

    def _on_mesh_color(self, new_color):
        self.materials['geometry'].base_color = [new_color.red, new_color.green, new_color.blue, new_color.alpha]
        self._scene.scene.modify_geometry_material('__MESH__', self.materials['geometry'])

    def _on_show_axes(self, show):
        self._show_axes.checked = show
        self._scene.scene.show_axes(self._show_axes.checked)

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        geometry_type = o3d.io.read_file_geometry_type(filename)
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            self.load(filename)
        else:
            print('change to diglog window later. this file does not contain mesh.')

    def _on_menu_save(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".tfm", "Transformation")
        dlg.set_on_cancel(self._on_dialog_cancel)
        dlg.set_on_done(self._on_save_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog_done(self, filename):
        self.window.close_dialog()
        pyperclip.copy('The text to be copied to the clipboard.')
        spam = pyperclip.paste()
    
        try:
            T = self._scene.scene.get_geometry_transform('__MIRRORED__')
            with open(filename, 'w') as f:
                f.write('\n'.join(','.join(f'{i:+1.8e}' for i in x) for x in T))
        except Exception as e:
            self.show_message_box('Error', e.__str__)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)


if __name__ == "__main__":

    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()
    global mono_id
    mono_id = gui.Application.instance.add_font(gui.FontDescription(typeface='monospace'))
    w = AppWindow(2048, 1536)
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    w.load(r'C:\data\chest\soft tissue_low.stl')

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()

