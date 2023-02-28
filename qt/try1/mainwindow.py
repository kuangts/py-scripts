# This Python file uses the following encoding: utf-8
import sys
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTreeWidget,
    QFrame,
    QGridLayout,
    QGraphicsView,
    QWidget,
    QTreeWidgetItem,
    QLabel,
    QVBoxLayout
)
from ui_mainwindow import Ui_MainWindow
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk import (
    vtkRenderer,
    vtkActor,
    vtkSphereSource,
    vtkPolyDataMapper
)
import landmark

class LandmarkSideWindow(QMainWindow):
    lmk_lib = landmark.Library()

    def __init__(self, parent=None):
        
        # super().__init__(parent)

        # self.setObjectName("LandmarkSideUI")
        # self.resize(960, 640)
        # self.centralWidget = QWidget(self)
        # self.vtkWidget = QVTKRenderWindowInteractor(self.centralWidget)

        # self.layout = QVBoxLayout(self.centralWidget)

        # self.tree = QTreeWidget(self)
        # self.tree.resize(240, 600)
        # self.name = QLabel(self)
        # self.name.setText('???????????')
        # self.name.resize(48, 600)
        # self.detail = QLabel(self)
        # self.detail.setText('???????????\n???????????\n???????????')
        # self.detail.resize(144, 600)

        # self.layout.addWidget(self.tree,1)
        # self.layout.addWidget(self.name,0)
        # self.layout.addWidget(self.detail,0)
        # self.layout.addWidget(self.vtkWidget,1)

        # self.setCentralWidget(self.centralWidget)


        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.tree = self.findChild(QTreeWidget, 'tree')
        self.name = self.findChild(QLabel, 'name_label')
        self.detail = self.findChild(QLabel, 'detail_label')
        self.bottom = self.findChild(QWidget, 'widget')
        self.gridlayout = QGridLayout(self.bottom)
        self.vtkWidget = QVTKRenderWindowInteractor(self.bottom)
        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 1, 1)

        # self.setCentralWidget(self.bottom)
        self.ren = vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

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
        self.ren.AddActor(actor)
        self.ren.ResetCamera()
        

    def load_lmk_tree(self):
        lib = self.lmk_lib.sorted()
        groups = {}
        for d in lib:
            if d.Group not in groups:
                item = QTreeWidgetItem()
                item.setText(0, d.Group)
                self.tree.addTopLevelItem(item)
                groups[d.Group] = item

            item = QTreeWidgetItem()
            item.setText(1, d.Name)
            item.setText(2, d.print_str()['Definition'])
            groups[d.Group].addChild(item)

        for x in groups.values():
            self.tree.expandItem(x)

        self.tree.itemClicked.connect(self.lmk_changed)


    def load_stl():
        pass

    def lmk_changed(self, item):
        if len(item.text(1)):
            item = self.lmk_lib.find(Name=item.text(1))
            if not item:
                return
            self.name.setText(item.Name)
            self.detail.setText(item.print_str()['Fullname'] + '\n' + item.print_str()['Description'])


    def show(self, *args, **kwargs):
        self.load_lmk_tree()
        
        super().show(*args, **kwargs)
        self.iren.Initialize()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LandmarkSideWindow()
    window.show()
    sys.exit(app.exec())
