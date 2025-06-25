from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self.plot_empty()

    def plot_empty(self):
        self.ax.clear()
        self.ax.set_title("训练指标")
        self.ax.set_xlabel("轮数")
        self.ax.set_ylabel("Loss")
        self.draw()

    def update_loss_curve(self, loss_list):
        self.ax.clear()
        self.ax.plot(loss_list, label="训练Loss", color='blue')
        self.ax.set_title("训练Loss曲线")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.legend()
        self.draw()

    def update_map_curve(self, map50_list, map5095_list):
        self.ax.clear()
        self.ax.plot(map50_list, label="mAP@0.5", color='green')
        self.ax.plot(map5095_list, label="mAP@0.5:0.95", color='orange')
        self.ax.set_title("mAP 指标曲线")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("mAP 值")
        self.ax.legend()
        self.draw()
