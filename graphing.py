import pyqtgraph.examples
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import json
import os.path
import time

#pyqtgraph.examples.run()

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.setWindowTitle('Unity - MDI')
mw.resize(800,800)
cw = QtGui.QWidget()
mw.setCentralWidget(cw)

#Create layout to manage widgets size and positions
layout = QtGui.QGridLayout()
cw.setLayout(layout)

btn_start = QtGui.QPushButton('Start')
btn_stop = QtGui.QPushButton('End')
plot = pg.PlotWidget()

layout.addWidget(btn_start,0,0)
layout.addWidget(btn_stop,1,0)
pw = pg.PlotWidget(name='Plot1')
layout.addWidget(pw,2,0)
pw2 = pg.PlotWidget(name='Plot2')
layout.addWidget(pw2,3,0)
pw3 = pg.PlotWidget(name='Plot3')
layout.addWidget(pw3,4,0)

mw.show()

## Create an empty plot curve to be filled later, set its pen
p1 = pw.plot()
p1.setPen((200,200,100))


pw.setLabel('left', 'Magnitude', units='dB')
pw.setLabel('bottom', 'Time', units='s')
pw.setXRange(0, 50)
pw.setYRange(0, 1e-10)

def rand(n):
    data = np.random.random(n)
    data[int(n*0.1):int(n*0.13)] += .5
    data[int(n*0.18)] += 2
    data[int(n*0.1):int(n*0.13)] *= 5
    data[int(n*0.18)] *= 20
    data *= 1e-12
    return data, np.arange(n, n+len(data)) / float(n)

def update_pw1(pw1_x,pw1_y):
    data = np.random.normal(size=(10000,50))
    p1.setData(data)


def updateData():
    yd, xd = rand(10000)
    p1.setData(y=yd, x=xd)

## Start a timer to rapidly update the plot in pw
t = QtCore.QTimer()
t.timeout.connect(updateData)
t.start(100)
#updateData()

## Multiple parameterized plots--we can autogenerate averages for these.
# for i in range(0, 5):
#     for j in range(0, 3):
#         yd, xd = rand(10000)
#         pw2.plot(y=yd*(j+1), x=xd, params={'iter': i, 'val': j})

## Test large numbers
# curve = pw3.plot(np.random.normal(size=100)*1e0, clickable=True)
# curve.curve.setClickable(True)
# curve.setPen('w')  ## white pen
# curve.setShadowPen(pg.mkPen((70,70,30), width=6, cosmetic=True))

# def clicked():
#     print("curve clicked")
# curve.sigClicked.connect(clicked)

# lr = pg.LinearRegionItem([1, 30], bounds=[0,100], movable=True)
# pw3.addItem(lr)
# line = pg.InfiniteLine(angle=90, movable=True)
# pw3.addItem(line)
# line.setBounds([0,200])

## Start Qt event loop unless running in interactive mode or using pyside.



#Plot1 Report

def report_plot(data):

    #






if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



