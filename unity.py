# -*- coding: utf-8 -*-

import sys

from scipy.fftpack import fft
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
import os
import threading

#matplotlib.rcParams['backend'] = "GTKAgg"
q = deque([])

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

def interactive_plot(q):
  # Interactive on
  plt.ion()

  plt.xlabel('Time (s)')
  plt.ylabel('EEG (mV)')
  plt.title('EEG vs. Time')
  plt.axis([0, 40000, -1000, 1000])

  plt.show()

  # x = data['x']
  # y = data['y']

  xs = np.array([])
  ys = np.array([])

  ax = plt.figure().gca()

  line, = ax.plot(xs, ys)

  # for x, y in zip(data['x'], data['y'])[0::100]:
  rate = 512
  i = 0
  while not q.empty():
    if i%rate:
      i = 0
      continue
    i += 1

    d = q.get()
    x = d['x']
    y = d['y']
    xs = np.append(xs, x)
    ys = np.append(ys, y)
    line.set_data(xs, ys)
    plt.scatter(x, y, s=1)
    plt.draw()
    # time.sleep(0.000001)

  plt.ioff()

def windowed_fft(q, slide=False, debug=False):
  # Given there are about 550 samples in one second let's use that size for window
  T = 1.0/512
  N = 512

  theta_start = 3
  theta_end = 7

  alpha_start = 7
  alpha_end = 15

  beta_start = 15
  beta_end = 32

  gamma_start = 32
  gamma_end = 64

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
  baseline = -0.5

  # Time
  t = 0.0
  time_scale = 500.0

  while True:
    if q:
      datum = q.popleft()

      # # Grab the leftmost data points
      # for i in range(step_size):
      #   datum.append(q.popleft())

      if len(window) < N:
        # Not enough items in the window to do anything interesting
        #   for i in range(step_size):
        # datum.append(q.popleft())
        window.append(datum)
      else:
        # window already contains N items, let's process it

        # EEG, x, y, z
        eeg = np.array([item['eeg'] for item in window])

        x = [abs(item['x']) for item in window if abs(item['x']) > 0.0]
        x = np.mean(x)
        X.append(x)

        y = [abs(item['y']) for item in window if abs(item['x']) > 0.0]
        y = np.mean(y)      
        Y.append(y)

        z = [abs(item['z']) for item in window if abs(item['x']) > 0.0]
        z = np.mean(z)
        Z.append(z)

        # Frequency domain for EEG
        eegf = fft(eeg)
        eegf = 2.0/N * np.abs(eegf[0:N/2])

        # Get the theta, alpha, beta, gamma average magnitudes

        # Theta 4 – 7
        # theta_avg = sum(eegf[theta_start:theta_end])/len(eegf[theta_start:theta_end])
        theta_avg = np.mean(eegf[theta_start:theta_end])
        theta.append(theta_avg)
        
        # Alpha 8 – 15
        # alpha_avg = sum(eegf[alpha_start:alpha_end])/len(eegf[alpha_start:alpha_end])
        alpha_avg = np.mean(eegf[alpha_start:alpha_end])      
        alpha.append(alpha_avg)

        # Beta 16 – 31
        # beta_avg = sum(eegf[beta_start:beta_end])/len(eegf[beta_start:beta_end])
        beta_avg = np.mean(eegf[beta_start:beta_end])            
        beta.append(beta_avg)

        # Gamma 32 - 64
        # gamma_avg = sum(eegf[gamma_start:gamma_end])/len(eegf[gamma_start:gamma_end])
        gamma_avg = np.mean(eegf[gamma_start:gamma_end])
        gamma.append(gamma_avg)

        # Compute output
        noise = np.random.normal(mu, sigma, size)[0]
        expo = np.exp([-t/time_scale])[0]

        output = 0.0
        output += noise
        output += baseline
        output += expo
        output += (x + y + z)/3.0

        # Bound output from 0.0 to 1.0
        if output < 0.0:
          output = 0.0
        elif output > 1.0:
          output = 1.0

        O.append(output)

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
      print "queue is empty :("
      time.sleep(1)

  # Temp disabling
  # if debug:
  #   print "****** Statistics ******"

  #   print 'min Theta: ', min(theta)
  #   print 'max Theta: ', max(theta)

  #   print 'min Alpha: ', min(alpha)
  #   print 'max Alpha: ', max(alpha)

  #   print 'min Beta: ', min(beta)
  #   print 'max Beta: ', max(beta)

  #   print 'min Gamma: ', min(gamma)
  #   print 'max Gamma: ', max(gamma)

  #   print 'min X: ', min(X)
  #   print 'max X: ', max(X)

  #   print 'min Y: ', min(Y)
  #   print 'max Y: ', max(Y)

  #   print 'min Z: ', min(Z)
  #   print 'max Z: ', max(Z)

  #   print 'min O: ', min(O)
  #   print 'max O: ', max(O)

  #   # Plot T vs. Output

  #   # First figure
  #   plt.figure(1)

  #   time_array = np.linspace(0.0, float(t), t)
  #   # print len(time_array)
  #   output_array = np.array(O)
  #   # print len(output_array)

  #   # Theta
  #   plt.subplot(211)
  #   plt.plot(time_array, output_array)
  #   plt.xlabel('Time (s)')
  #   plt.ylabel('Output (vol)')
  #   plt.title('Output vs. Time')

  #   plt.grid()
  #   plt.show()





  # L = len(data['eeg'])-N

  # # For each window take FFT, determine the magnitude for ranges
  # for i in range(L):
  #   x = data['x'][i:N+i]
  #   y = np.array(data['y'][i:N+i])

  #   # Frequency domain
  #   yf = fft(y)
  #   yf = 2.0/N * np.abs(yf[0:N/2])

  #   # Get the theta, alpha, beta, gamma average magnitudes

  #   # Theta 4 – 7
  #   theta_avg = sum(yf[theta_start:theta_end])/len(yf[theta_start:theta_end])
  #   theta.append(theta_avg)
    
  #   # Alpha 8 – 15
  #   alpha_avg = sum(yf[alpha_start:alpha_end])/len(yf[alpha_start:alpha_end])
  #   alpha.append(alpha_avg)

  #   # Beta 16 – 31
  #   beta_avg = sum(yf[beta_start:beta_end])/len(yf[beta_start:beta_end])
  #   beta.append(beta_avg)

  #   # Gamma 32 - 64
  #   gamma_avg = sum(yf[gamma_start:gamma_end])/len(yf[gamma_start:gamma_end])
  #   gamma.append(gamma_avg)

  #   t.append(x[-1])

  # # Convert to np.array
  # for ar in [t, theta, alpha, beta, gamma]:
  #   ar = np.array(ar)

  # # First figure
  # plt.figure(1)

  # # Theta
  # plt.subplot(221)
  # plt.plot(t, theta)
  # plt.xlabel('Time (s)')
  # plt.ylabel('Magnitude (dB)')
  # plt.title('Theta vs. Time')

  # # Alpha
  # plt.subplot(222)
  # plt.plot(t, alpha)
  # plt.xlabel('Time (s)')
  # plt.ylabel('Magnitude (dB)')
  # plt.title('Alpha vs. Time')

  # # Beta
  # plt.subplot(223)
  # plt.plot(t, beta)
  # plt.xlabel('Time (s)')
  # plt.ylabel('Magnitude (dB)')
  # plt.title('Beta vs. Time')

  # # Gamma
  # plt.subplot(224)
  # plt.plot(t, gamma)
  # plt.xlabel('Time (s)')
  # plt.ylabel('Magnitude (dB)')
  # plt.title('Gamma vs. Time')

  # plt.grid()
  # plt.show(block=True)

def process_data(data):
  # Number of samplepoints
  N = len(data['x'])

  # sample spacing - Play around with this
  # This is approximately the spacing of the actual data
  T = 1.0 / 550.0
  # x = np.linspace(0.0, N*T, N)
  
  x = np.array(data['x'])
  y = np.array(data['y'])

  # Frequency domain
  yf = fft(y)

  # Use simplifying assumption that x are evenly spaced
  xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

  # First figure
  plt.figure(1)

  # Time domain
  plt.subplot(211)
  plt.plot(data['x'], data['y'])
  plt.xlabel('Time (s)')
  plt.ylabel('EEG (mV)')
  plt.title('EEG vs. Time')

  # Frequency domain
  plt.subplot(212)
  plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Magnitude (dB)')
  plt.title('Magnitude vs. Frequency')

  # xmin xmax ymin ymax
  plt.axis([1, 40, 0, 6])

  # plt.plot(data['x'], data['y'])
  plt.grid()
  plt.show()

def restrict_data(data, factor):
  for key in data:
    to = len(data[key])/factor
    data[key] = data[key][0: to]

  return data

def readPipe(pipe):
    if not os.path.exists(pipe):
        os.mkfifo(pipe, 0666)

    fd = os.open(pipe, os.O_RDONLY)
    response = os.read(fd, 2000)
    print response + "\n"

    # Tmp until we get actual values for xyz
    eeg, x, y, z = [float(tmp.strip()) if tmp.strip() else 0.0 for tmp in response.split(",") + [""]*3]

    # x = 0.0
    # y = 0.0
    # z = 0.0

    d = {
      'eeg': eeg,
      'x': x,
      'y': y,
      'z': z
    }

    q.append(d)

    # for val in vals:
    #     if val:
    #         q.append(val)

    os.close(fd)

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

    windowed_fft(q, slide=False, debug=True)

    for t in threads:
        t.join()

    for num in range(1, 11):
        os.remove(pipe + str(num))
    print "main exits"

if __name__ == "__main__":
  main()
