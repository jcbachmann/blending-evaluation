import random
import sys

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        figure = plt.figure()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(toolbar)
        self.setLayout(layout)

        for _ in range(3):
            data = [random.random() for _ in range(25)]
            ax = figure.add_subplot(111)
            ax.plot(data, '*-')
            canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main = Window()
    main.setWindowTitle('Interactive Plot')
    main.show()

    sys.exit(app.exec_())
