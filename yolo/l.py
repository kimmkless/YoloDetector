import os, PyQt5
import sys
from PyQt5.QtWidgets import QApplication, QLabel
dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir


app = QApplication(sys.argv)
label = QLabel("Qt is working!")
label.show()
sys.exit(app.exec_())

