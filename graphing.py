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
    #do stuff for ui if needed
    print "updateUI clicked"

## Start a timer to rapidly update the plot in widget
# t = QtCore.QTimer()
# t.timeout.connect(updateData)
# t.start(1000)

def btnclick_start():
    nr = layout.count()
    for i in reversed(range(nr, 1, -1)):
        layout.itemAt(i).widget().deleteLater()

    #TODO: resizing doesn't work as expected
    mw.resize(100,100)
    mw.show()

    btn_stop.setDisabled(True)
    btn_stop.setDisabled(False)
    #add function here to start calibration
    updateUI()

def btnclick_stop():
    btn_stop.setDisabled(True)
    btn_start.setDisabled(False)
    data = np.array(random_generator(1000))

    #get data here
    for i in range(3):
        report_plot(i,data)

btn_start.clicked.connect(btnclick_start)
btn_stop.clicked.connect(btnclick_stop)

def report_plot(nr, data):
    y = data
    #y_max = np.hstack(y).max(axis = 0)
    #x_max = y.size//512
    y_max = 1
    x_max = 120

    pw_report = pg.PlotWidget()
    layout.addWidget(pw_report, nr + 3, 0)
    pw_report.setLabel('left', 'Magnitude', units='dB')
    pw_report.setLabel('bottom', 'Time', units='s')
    # pw_report.setXRange(0, x_max)
    # pw_report.setYRange(0, y_max)
    pw_report.plot(y)

    #fiddle with this for additional plots
    mw.resize(500,500)

def random_generator(n):
    data_x = np.random.random(n)
    print data_x
    return data_x

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



