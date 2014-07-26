# -*- coding: utf-8 -*-

import sys

from scipy.fftpack import fft
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import os
import threading
import pygame
import datetime
import json
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#matplotlib.rcParams['backend'] = "GTKAgg"
q = deque([])

modelling_noise = True
calibrating = True
start_session = True
exit_app = False

theta_start = 3
theta_end = 7

alpha_start = 7
alpha_end = 15

beta_start = 15
beta_end = 32

gamma_start = 32
gamma_end = 64

app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
#Create layout to manage widgets size and positions
layout = QtGui.QGridLayout()

btn_start = QtGui.QPushButton('Start Meditation')
btn_stop = QtGui.QPushButton('End Session')

def updateUI():
    #do stuff for ui if needed
    print "updateUI clicked"

def btnclick_start():
    global modelling_noise
    global start_session
    global calibrating

    modelling_noise = False
    start_session = True
    calibrating = True

    nr = layout.count()
    for i in reversed(range(nr, 1, -1)):
      if layout.itemAt(i):
        layout.itemAt(i).widget().deleteLater()

    #TODO: resizing doesn't work as expected
    mw.resize(100,100)
    mw.show()

    btn_stop.setDisabled(True)
    btn_stop.setDisabled(False)
    #add function here to start calibration
    updateUI()

def btnclick_stop():
    global modelling_noise
    global start_session
    start_session = False
    modelling_noise = True

    btn_stop.setDisabled(True)
    btn_start.setDisabled(False)
    data = np.array(random_generator(1000))

    #get data here
    for i in range(3):
        report_plot(i,data)

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

def get_data(file_name):
  data = {
    'eeg': [],
    'x': [],
    'y': [],
    'z': []
  }

  with open(file_name) as f:
    # Ignore header
    header = f.readline()

    for input_data in f:
      # ignore time step
      eeg, x, y, z = [float(tmp.strip()) if tmp.strip() else 0.0 for tmp in input_data.split(",")[1:5]]

      d = {
        'eeg': eeg,
        'x': x,
        'y': y,
        'z': z
      }
      q.append(d)

      data['eeg'].append(eeg)
      data['x'].append(x)
      data['y'].append(y)
      data['z'].append(z)

  return data

def get_noise_model(q, debug=True):
  # Constructs a noise model that will be used to offset the actual signal from the sensor

  print "Entering modelling noise"

  N = 512

  eeg_avg = []
  x_avg = []
  y_avg = []
  z_avg = []

  window = []

  while modelling_noise:
    if q:
      print "noise modelling: collecting noise..."
      datum = q.popleft()

      if len(window) < N:
        window.append(datum)
      else:
        for array, label in zip([eeg_avg, x_avg, y_avg, z_avg], ['eeg', 'x', 'y', 'z']):
          avg = np.mean([item[label] for item in window])
          array.append(avg)
    else:
      print "noise modelling: queue is empty :("
      time.sleep(1)

  # Return average of the averaged values
  averages = dict()

  if eeg_avg:
    averages = {
      'eeg': np.mean(eeg_avg),
      'x': np.mean(x_avg),
      'y': np.mean(y_avg),
      'z': np.mean(z_avg)
    }

  if debug:
    print "noise modelling: "
    print averages

  return averages

def offset_g(original, offset):
  return abs(original - offset)

def bound_eeg(eeg):
    if eeg > 800:
        eeg = 800

    return eeg

def bound_output(output):
  # Bound output from 0.0 to 1.0
  if output < 0.0:
    output = 0.0
  elif output > 1.0:
    output = 1.0

  return output

def get_mean_pos(label, window):
  val = [abs(item[label]) for item in window if abs(item[label]) > 0.0]
  val = np.mean(val)

  return val

def get_mean_eeg_spectrums(eegf):
  # Theta 4 – 7
  # theta_avg = sum(eegf[theta_start:theta_end])/len(eegf[theta_start:theta_end])
  theta_avg = np.mean(eegf[theta_start:theta_end])
  
  # Alpha 8 – 15
  # alpha_avg = sum(eegf[alpha_start:alpha_end])/len(eegf[alpha_start:alpha_end])
  alpha_avg = np.mean(eegf[alpha_start:alpha_end])      

  # Beta 16 – 31
  # beta_avg = sum(eegf[beta_start:beta_end])/len(eegf[beta_start:beta_end])
  beta_avg = np.mean(eegf[beta_start:beta_end])            

  # Gamma 32 - 64
  # gamma_avg = sum(eegf[gamma_start:gamma_end])/len(eegf[gamma_start:gamma_end])
  gamma_avg = np.mean(eegf[gamma_start:gamma_end])

  return theta_avg, alpha_avg, beta_avg, gamma_avg

def windowed_fft(q, slide=False, debug=False, noise_model=None):
  print "Entering windowed_fft"

  # Given there are about 550 samples in one second let's use that size for window
  T = 1.0/512
  N = 512

  # t = []
  theta = []
  alpha = []
  beta = []
  gamma = []
  X = []
  Y = []
  Z = []
  O = []

  window = []

  # Gaussian noise
  mu = 0
  sigma = 0.05
  size = 1
  
  # Normal volume
  baseline = -0.3

  # Time
  t = 0.0
  time_scale = 500.0

  WARM_UP_TIME = 5.0

  # Default offsets
  eeg_offset = 0
  x_offset = 3
  y_offset = 6
  z_offset = 60

  eeg_warm = []
  theta_warm = []
  alpha_warm = []
  beta_warm = []
  gamma_warm = []
  X_warm = []
  Y_warm = []
  Z_warm = []

  file_name = "output_%s" % str(datetime.datetime.now().date())
  with open(file_name, 'w') as f:
    while start_session:
      if q:
        datum = q.popleft()

        if len(window) < N:
          window.append(datum)
        else:
          # Warm up. Get some initial readings on the person to evaluate their starting state.
          if calibrating:
            print "Calibrating..."
            # print "Entering warm up time: %d" % t

            # EEG, x, y, z
            eeg = np.array([item['eeg'] for item in window])
            eeg_warm.append(np.mean(eeg))

            x = get_mean_pos('x', window)
            X_warm.append(x)

            y = get_mean_pos('y', window)
            Y_warm.append(y)

            z = get_mean_pos('z', window)
            Z_warm.append(z)

            # Compute offsets
            eeg_offset = np.mean(eeg_warm)
            x_offset = np.mean(X_warm)
            y_offset = np.mean(Y_warm)
            z_offset = np.mean(Z_warm)

            #*** For Freq domain averaging; not using right now ***
            # Frequency domain for EEG
            eegf = fft(eeg)
            eegf = 2.0/N * np.abs(eegf[0:N/2])

            # Get the theta, alpha, beta, gamma average magnitudes
            theta_avg, alpha_avg, beta_avg, gamma_avg = get_mean_eeg_spectrums(eegf)

            theta_warm.append(theta_avg)    
            alpha_warm.append(alpha_avg)      
            beta_warm.append(beta_avg)
            gamma_warm.append(gamma_avg)
            #*** End optional code ***

            if slide:
              # Slide the window
              window = window[1:N]
              window.append(datum)
            else:
              # Empty the window completely for copy-paste style processing
              window = []
          else:
            print "Meditating..."
            # EEG, x, y, z
            eeg = np.array([item['eeg'] for item in window])

            # Offset noise
            if noise_model:
              eeg = [data - noise_model['eeg'] for data in eeg]

            # Offset human bias
            eeg = [data - eeg_offset for data in eeg]

            x = get_mean_pos('x', window)
            x = offset_g(x, x_offset)
            X.append(x)

            y = get_mean_pos('y', window)
            y = offset_g(y, y_offset)
            Y.append(y)

            z = get_mean_pos('z', window)
            z = offset_g(z, z_offset)
            Z.append(z)

            # Frequency domain for EEG
            eegf = fft(eeg)
            eegf = 2.0/N * np.abs(eegf[0:N/2])

            # Get the theta, alpha, beta, gamma average magnitudes
            theta_avg, alpha_avg, beta_avg, gamma_avg = get_mean_eeg_spectrums(eegf)

            # Compute output
            noise = np.random.normal(mu, sigma, size)[0]
            expo = np.exp([-t/time_scale])[0]

            output = 0.0
            output += noise
            output += baseline
            output += expo
            output += x/30.0
            output += y/30.0
            output += z/30.0

            # Comment this for the PI
            theta.append(theta_avg)
            alpha.append(alpha_avg)
            beta.append(beta_avg)
            gamma.append(gamma_avg)
            output = bound_output(output)

            if pygame.mixer.music.get_busy():
              adjustVol(output)
            else:
              playMusic()

            O.append(output)

            # For outputing to a file
            # data_packet = {
            #   't': t,
            #   'theta': theta_avg,
            #   'alpha': alpha_avg,
            #   'gamma': gamma_avg,
            #   'beta': beta_avg,
            #   'x': x,
            #   'y': y,
            #   'z': z,
            #   'output': output,
            #   'noise': noise,
            #   'exp': expo
            # }

            # print "Outputting data to file"
            # print data_packet

            # json_packet = json.dumps(data_packet)
            # json.dump(json_packet, f)
            # f.write("\n")

            # Debug
            if debug:
              print 'T: ', t
              print 'Theta: ', theta_avg
              print 'Alpha: ', alpha_avg
              print 'Beta: ', beta_avg
              print 'Gamma: ', gamma_avg
              print 'X: ', x
              print 'Y: ', y
              print 'Z: ', z
              print 'Noise: ', noise
              print "Exponential: ", expo
              print 'Output: ', output

            if slide:
              # Slide the window
              window = window[1:N]
              window.append(datum)
              t += T
            else:
              # Empty the window completely for copy-paste style processing
              window = []
              t += T*N
      else:
        #print "queue is empty :("
        time.sleep(1)

def playMusic():
    #check for user input of starting session
    pygame.mixer.music.load("enya.mp3")
    pygame.mixer.music.play(0)

    if pygame.mixer.music.get_busy():
        print "music playing...."
        time.sleep(5)


def adjustVol(vol):
    pygame.mixer.music.set_volume(vol)
    print "Adjusted volume to " + str(vol)

def readPipe(pipe):
    while True:
        if not os.path.exists(pipe):
            os.mkfifo(pipe, 0666)

        fd = os.open(pipe, os.O_RDONLY)
        response = os.read(fd, 2000)
        #print response + "\n"

        vals = response.split(",")
        for val in vals:
            if val.strip():
                eeg, x, y, z = [float(tmp.strip()) if tmp.strip() else 0.0 for tmp in val.split("/")]

                d = {
                    'eeg': eeg,
                    'x': x,
                    'y': y,
                    'z': z
                }

                q.append(d)

        os.close(fd)

def stopNoise(timeout):
    global modelling_noise
    while True:
        if q:
            time.sleep(timeout)
            print "Stopping modelling noise"
            modelling_noise = False
            break

def stop_calibration(timeout):
    global calibrating
    while True:
        if q:
            time.sleep(timeout)
            print "Stopping calibration. Starting meditation session"
            calibrating = False
            break

def runner(q):
  while not exit_app:
    noise_model = get_noise_model(q)

    # Set warm up for 5 seconds. We can hook this up to an actual function later.
    thread_stop_calibration = threading.Thread(target=stop_calibration, args=(10,))
    thread_stop_calibration.start()

    windowed_fft(q, slide=False, debug=True, noise_model=noise_model)

def ui_init():
  mw.setWindowTitle('Unity - MDI')
  mw.resize(100,100)
  cw = QtGui.QWidget()
  mw.setCentralWidget(cw)
  cw.setLayout(layout)
  layout.addWidget(btn_start,0,0)
  layout.addWidget(btn_stop,1,0)

  mw.show()
  btn_start.clicked.connect(btnclick_start)
  btn_stop.clicked.connect(btnclick_stop)

def main():
  if '--data' in sys.argv:
    file_name = sys.argv[2]
    data = get_data(file_name)

    # interactive_plot(q)
    # data = restrict_data(data, 10)

    windowed_fft(q, slide=False, debug=True)

    # process_data(data)
    # interactive_plot(data)

  else:
    pipe = '/tmp/opi_pipe'
    threads = []
    for i in range(1, 11):
        thread = threading.Thread(target=readPipe, args=((pipe + str(i)),))
        threads.append(thread)
        thread.start()

    # thread_music = threading.Thread(target=playMusic, args=())
    # threads.append(thread_music)
    # thread_music.start()

    pygame.init()

    thread_runner = threading.Thread(target=runner, args=(q,))
    threads.append(thread_runner)
    thread_runner.start()

    # thread_stopNoise = threading.Thread(target=stopNoise, args=(20,))
    # threads.append(thread_stopNoise)
    # thread_stopNoise.start()

    # noise_model = get_noise_model(q)

    # Set warm up for 5 seconds. We can hook this up to an actual function later.
    # thread_stop_calibration = threading.Thread(target=stop_calibration, args=(10,))
    # threads.append(thread_stop_calibration)
    # thread_stop_calibration.start()

    # thread_fft = threading.Thread(target=windowed_fft, args=(q, False, True, noise_model,))
    # threads.append(thread_fft)
    # thread_fft.start()
    #windowed_fft(q, slide=False, debug=True, noise_model=noise_model)

    #UI INITIALIZATIONS
    ui_init()
    QtGui.QApplication.instance().exec_()

    # Close application
    global exit_app
    global modelling_noise
    global start_session

    exit_app = True
    modelling_noise = False
    start_session = False

    for t in threads:
        t.join()

    for num in range(1, 11):
        os.remove(pipe + str(num))
    print "main exits"

if __name__ == "__main__":
  main()
