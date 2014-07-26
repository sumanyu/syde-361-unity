from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#pyqtgraph.examples.run()

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.setWindowTitle('Unity - MDI')
mw.resize(100,100)
cw = QtGui.QWidget()
mw.setCentralWidget(cw)

#Create layout to manage widgets size and positions
layout = QtGui.QGridLayout()
cw.setLayout(layout)

btn_start = QtGui.QPushButton('Start Meditation')
btn_stop = QtGui.QPushButton('End Session')
layout.addWidget(btn_start,0,0)
layout.addWidget(btn_stop,1,0)

plot = pg.PlotWidget()

mw.show()

def updateUI():
    print "updateUI cliked"

## Start a timer to rapidly update the plot in widget
# t = QtCore.QTimer()
# t.timeout.connect(updateData)
# t.start(1000)

def btnclick_start():
    #add function here to start calibration
    updateUI()

def btnclick_stop():
    data = random_generator(100)
    report_plot(1,data)

btn_start.clicked.connect(btnclick_start)
btn_stop.clicked.connect(btnclick_stop)

def report_plot(nr, data):
    y = np.array(random_generator(1000))
    #y_max = np.hstack(y).max(axis = 0)
    #x_max = y.size//512
    y_max = 1
    x_max = 120

    pw_report = pg.PlotWidget()
    layout.addWidget(pw_report, 2, 0)
    pw_report.setLabel('left', 'Magnitude', units='dB')
    pw_report.setLabel('bottom', 'Time', units='s')
    # pw_report.setXRange(0, x_max)
    # pw_report.setYRange(0, y_max)
    pw_report.plot(y)

    #fiddle with this for additional plots
    mw.resize(800,800)

def random_generator(n):
    data_x = np.random.random(n)
    print data_x
    return data_x

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



