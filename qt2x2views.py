
class ObjectView(QVTK, EventHandler):
    def __init__(self, parent=None, **kw):
        super().__init__(parent=parent, **kw)
        style = vtkInteractorStyleTrackballCamera()
        style.AutoAdjustCameraClippingRangeOn()
        style.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(style)
        style.lbp = style.AddObserver('LeftButtonPressEvent', self.left_button_press)
        style.lbr = style.AddObserver('LeftButtonReleaseEvent', self.left_button_release)
        style.lbr = style.AddObserver('MouseMoveEvent', self.mouse_move_event)
        style.kp = style.AddObserver('KeyPressEvent', self.key_press_event)
        style.kr = style.AddObserver('KeyReleaseEvent', self.key_release_event)

        self.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(style)
        return None


    @property
    def current_ui_task(self):
        if hasattr(self, '_current_ui_task'):
            return self._current_ui_task
        else:
            return None
    
    @current_ui_task.setter
    def current_ui_task(self, t):
        setattr(self, '_current_ui_task', t)
        return None


    def pick(self, obj, debug=True):
        pos = obj.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
        # record picked point position
        self.current_point = [float('nan'),]*3
        self.current_point_ijk = [float('nan'),]*3
        if self.picker.GetCellId() != -1:
            self.current_point = self.picker.GetPickPosition()
            self.img_data.TransformPhysicalPointToContinuousIndex(self.current_point, self.current_point_ijk)
        if debug:
            try:
                name = self.picker.GetProp3D().GetObjectName()
                assert name
            except:
                name = self.objectName()
            print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.current_point), end=' ')
        return None


    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is pressed')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state | InteractionState.CTRL
        elif key.startswith('Alt'):
            self.interaction_state = self.interaction_state | InteractionState.ALT
        else:
            self.parent().key_press_event(obj, event)


    def key_release_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is released')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state & ~InteractionState.CTRL
        if key.startswith('Alt'):
            self.interaction_state = self.interaction_state & ~InteractionState.ALT
        else:
            self.parent().key_release_event(obj, event)


    # button press is relatively expensive
    # mouse move could abort
    def left_button_press(self, obj, event):
        self.interaction_state = self.interaction_state.left_button_pressed()
        self.pick(obj)
        if self.app_window.mode == Mode.LANDMARK:
            print('& LMK', end=' ')
        elif self.app_window.mode == Mode.VIEW:
            if self.interaction_state & InteractionState.ALT & ~InteractionState.CTRL:
                obj.OnLeftButtonDown()
    
        print('**')
        return None


    def mouse_move_event(self, obj, event):
        self.interaction_state = self.interaction_state.mouse_moved()
        # if self.current_ui_task is not None:
        #     self.current_ui_task.cancel()
        # self.current_ui_task = asyncio.create_task(self.pick(obj))
        # await self.current_ui_task
        self.pick(obj)
        if self.interaction_state & InteractionState.CTRL:
            print('**')
            return None

        elif self.interaction_state & InteractionState.LEFT_BUTTON:
            if self.app_window.mode == Mode.LANDMARK:
                print('& LMK', end=' ')
            elif self.app_window.mode == Mode.VIEW:
                print('& VIEW', end=' ')
                obj.OnMouseMove() # might do nothing if OnLeftButtonDown is not properly called
                    
        print('**')
        return None


    def left_button_release(self, obj, event):
        # do not rely on clean up
        # handle stuff on the fly using move event and implement abort
        obj.OnLeftButtonUp()
        self.interaction_state = self.interaction_state.left_button_released()

        print('**')
        return None

class ImageView(QVTK, EventHandler):

    slice_change_singal = Signal(int, int) # orientation, slice#
    all_slice_change_singal = Signal(int, int, int) # YZ slice, XZ slice, ZY slice

    def __init__(self, parent=None, orientation=2, **kw):
        super().__init__(parent=parent, **kw)

        self.orientation = orientation

        style = vtkInteractorStyleImage()
        style.AutoAdjustCameraClippingRangeOn()
        style.lbp = style.AddObserver('LeftButtonPressEvent', self.left_button_press)
        style.lbr = style.AddObserver('LeftButtonReleaseEvent', self.left_button_release)
        style.lbr = style.AddObserver('MouseMoveEvent', self.mouse_move_event)
        style.kp = style.AddObserver('KeyPressEvent', self.key_press_event)
        style.kr = style.AddObserver('KeyReleaseEvent', self.key_release_event)
        style.mwf = style.AddObserver('MouseWheelForwardEvent', self.scroll_forward)
        style.mwb = style.AddObserver('MouseWheelBackwardEvent', self.scroll_backward)

        style.SetDefaultRenderer(self.renderer)
        self.SetInteractorStyle(style)
        self.viewer = vtkImageViewer2()
        self.viewer.SetRenderWindow(self.window)
        self.viewer.SetRenderer(self.renderer)
        self.viewer.SetSliceOrientation(orientation)
        self.renderer.SetBackground(0.2, 0.2, 0.2)

        return None

    
    def scroll_forward(self, obj, event):
        self.slice_change_singal.emit(self.orientation, self.viewer.GetSlice() + 1)
        return None


    def scroll_backward(self, obj, event):
        self.slice_change_singal.emit(self.orientation, self.viewer.GetSlice() - 1)
        return None
        

    def show(self):
        self.viewer.SetInputData(self.img_data)
        self.viewer.SetSlice((self.viewer.GetSliceMin() + self.viewer.GetSliceMax())//2)
        self.viewer.Render()
        super().show()

    def pick(self, obj, debug=True):
        pos = obj.GetInteractor().GetEventPosition()
        self.picker.Pick(pos[0], pos[1], 0, obj.GetDefaultRenderer())
        # record picked point position
        self.current_point = [float('nan'),]*3
        self.current_point_ijk = [float('nan'),]*3
        if self.picker.GetCellId() != -1:
            self.current_point = self.picker.GetPickPosition()
            self.img_data.TransformPhysicalPointToContinuousIndex(self.current_point, self.current_point_ijk)
        if debug:
            try:
                name = self.picker.GetProp3D().GetObjectName()
                assert name
            except:
                name = self.objectName()
            print('on "{}" ({:4d},{:4d}) at ({:6.2f},{:6.2f},{:6.2f})'.format(name, *pos, *self.current_point), end=' ')
        return None


    def key_press_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is pressed')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state | InteractionState.CTRL
        elif key.startswith('Alt'):
            self.interaction_state = self.interaction_state | InteractionState.ALT
        else:
            self.parent().key_press_event(obj, event)


    def key_release_event(self, obj, event):
        key = obj.GetInteractor().GetKeySym()
        print(f'key {key} is released')
        if key.startswith('Control'):
            self.interaction_state = self.interaction_state & ~InteractionState.CTRL
        if key.startswith('Alt'):
            self.interaction_state = self.interaction_state & ~InteractionState.ALT
        else:
            self.parent().key_release_event(obj, event)



    def left_button_press(self, obj, event):
        self.app_window.interaction_state = self.interaction_state.left_button_pressed()
        self.pick(obj)
        if self.app_window.param.mode == Mode.LANDMARK:
            print('& LMK', end=' ')
        elif self.app_window.mode == Mode.VIEW:
            if self.interaction_state & InteractionState.ALT & ~InteractionState.CTRL:
                obj.OnLeftButtonDown()
    
        print('**')
        return None


    def mouse_move_event(self, obj, event):
        self.interaction_state = self.interaction_state.mouse_moved()
        # if self.current_ui_task is not None:
        #     self.current_ui_task.cancel()
        # self.current_ui_task = asyncio.create_task(self.pick(obj))
        # await self.current_ui_task
        self.pick(obj)
        if self.interaction_state & InteractionState.CTRL:
            ijk = [int(round(x)) for x in self.current_point_ijk]
            self.all_slice_change_singal.emit(self)
            print('**')
            return None

        elif self.interaction_state & InteractionState.LEFT_BUTTON:
            if self.app_window.mode == Mode.LANDMARK:
                print('& LMK', end=' ')
            elif self.app_window.mode == Mode.VIEW:
                print('& VIEW', end=' ')
                obj.OnMouseMove() # might do nothing if OnLeftButtonDown is not properly called
                    
        print('**')
        return None


    def left_button_release(self, obj, event):
        # do not rely on clean up
        # handle stuff on the fly using move event and implement abort
        obj.OnLeftButtonUp()
        self.interaction_state = self.interaction_state.left_button_released()

        print('**')
        return None

