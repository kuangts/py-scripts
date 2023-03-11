#!/usr/bin/env python3
# Contributed by Eric E Monson

import sys, os 
import numpy
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
import PyQt5
import PyQt6
from PyQt5 import QtGui
from PySide6.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QMdiSubWindow
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera, vtkInteractorStyleTrackballActor, vtkInteractorStyleSwitch

from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderer,
    vtkCamera
)

# # did not work, too old
# class Example:

#     class VtkQtFrame(QtGui.QWidget):
#         # http://www.ifnamemain.com/posts/2013/Dec/08/python_qt_vtk/
#         def __init__(self):
#             QtGui.QWidget.__init__(self)    
#             self.layout = QtGui.QVBoxLayout(self)
#             self.layout.setContentsMargins(0, 0, 0, 0)
#             self.setLayout(self.layout)

#             self.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)

#             self.VTKRenderer = vtkRenderer()   
#             self.VTKRenderer.SetBackground(1, 1, 1)
#             self.VTKRenderer.SetViewport( 0, 0, 1, 1)

#             self.VTKRenderWindow = vtkRenderWindow()
#             self.VTKRenderWindow.AddRenderer(self.VTKRenderer)    
#             self.VTKRenderWindowInteractor = QVTKRenderWindowInteractor(self,rw=self.VTKRenderWindow)
#             self.layout.addWidget(self.VTKRenderWindowInteractor)

#             self.VTKCamera = vtkCamera()
#             self.VTKCamera.SetClippingRange(0.1,1000)
#             self.VTKRenderer.SetActiveCamera(self.VTKCamera)

#             self.VTKInteractorStyleSwitch = vtkInteractorStyleSwitch()
#             self.VTKInteractorStyleSwitch.SetCurrentStyleToTrackballCamera()
#             self.VTKRenderWindowInteractor.SetInteractorStyle(self.VTKInteractorStyleSwitch)
#             self.VTKRenderWindowInteractor.Initialize()
#             self.VTKRenderWindowInteractor.Start()
#             self.VTKRenderWindowInteractor.ReInitialize()


#     class SphereActor(vtkActor):
#         def __init__(self,rad,res,r,c):
#             self.pos = numpy.array(r)
#             self.source = vtkSphereSource()
#             self.source.SetRadius(rad)
#             self.source.SetPhiResolution(res)
#             self.source.SetThetaResolution(res)
#             self.source.SetCenter(r[0],r[1],r[2])
#             self.Mapper = vtkPolyDataMapper()
#             self.Mapper.SetInput(self.source.GetOutput())
#             self.SetMapper(self.Mapper)
#             self.GetProperty().SetColor((c[0],c[1],c[2]))
#         def move_to(self,r):
#             self.pos = numpy.array(r)
#             self.source.SetCenter(r[0],r[1],r[2])    
#         def set_color(self,color):
#             self.GetProperty().SetColor(color)
#         def set_rad(self,rad):
#             self.source.SetRadius(rad)    
#         def get_pos(self):
#             return self.pos


#     def __init__(self, *args):
#         app = QtGui.QApplication(sys.argv)

#         #create our new Qt MainWindow
#         window = QtGui.QMainWindow()

#         #create our new custom VTK Qt widget
#         render_widget = self.__class__.VtkQtFrame()

#         for i in range(0,10):
#             # random 3D position between 0,10
#             r = numpy.random.rand(3)*10.0
#             # random RGB color between 0,1
#             c = numpy.random.rand(3)
#             # create new sphere actor
#             my_sphere = self.__class__.SphereActor(1.0,20,r,c)
#             # add to renderer
#             render_widget.VTKRenderer.AddActor(my_sphere)

#         # reset the camera and set anti-aliasing to 2x
#         render_widget.VTKRenderer.ResetCamera()
#         render_widget.VTKRenderWindow.SetAAFrames(2)

#         # add and show
#         window.setCentralWidget(render_widget)
#         window.show()

#         # start the event loop
#         try:
#             sys.exit(app.exec_())
#         except SystemExit as e:
#             if e.code != 0:
#                 raise()
#             os._exit(0)




class vtkSubView(QMdiSubWindow):

    def __init__(self, parent=None, size=None, title=None):

        super().__init__(parent)

        self.setObjectName("MainWindow")
        # self.resize(603, 553)
        wdgt = QWidget(self)
        self.vtkWidget = QVTKRenderWindowInteractor(wdgt)
        self.gridlayout = QGridLayout(wdgt)
        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 1, 1)
        self.setWidget(wdgt)
        self.renderer = vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        # Create source
        source = vtkSphereSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(5.0)

        # Create a mapper
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        # Create an actor
        actor = vtkActor()
        actor.SetMapper(mapper)

        self.renderer.AddActor(actor)


if __name__ == "__main__":


    app = QApplication(sys.argv)
    window = vtkSubView()
    window.show()
    window.interactor.Initialize()  # Need this line to actually show the render inside Qt
    sys.exit(app.exec())