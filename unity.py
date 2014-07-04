# -*- coding: utf-8 -*-

import sys

from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import time

def get_data(file_name):
  data = {
    'x': [],
    'y': []
  }

  with open(file_name) as f:
    # Ignore header
    header = f.readline()

    for input_data in f:
      x, y = [float(tmp.strip()) for tmp in input_data.split(",")[:2]]
      # print x, y
      data['x'].append(x)
      data['y'].append(y)

  return data

def interactive_plot(data):
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

  for x, y in zip(data['x'], data['y'])[0::100]:
    xs = np.append(xs, x)
    ys = np.append(ys, y)
    line.set_data(xs, ys)
    plt.scatter(x, y, s=1)
    plt.draw()
    # time.sleep(0.000001)

  plt.ioff()

def windowed_fft(data):
  # Given there are about 550 samples in one second let's use that size for window
  T = 1.0/550.0
  N = 550

  theta_start = 3
  theta_end = 7

  alpha_start = 7
  alpha_end = 15

  beta_start = 15
  beta_end = 32

  gamma_start = 32
  gamma_end = 64

  t = []
  theta = []
  alpha = []
  beta = []
  gamma = []

  L = len(data['x'])-N

  # For each window take FFT, determine the magnitude for ranges
  for i in range(L):
    x = data['x'][i:N+i]
    y = np.array(data['y'][i:N+i])

    # Frequency domain
    yf = fft(y)
    yf = 2.0/N * np.abs(yf[0:N/2])

    # Get the theta, alpha, beta, gamma average magnitudes

    # Theta 4 – 7
    theta_avg = sum(yf[theta_start:theta_end])/len(yf[theta_start:theta_end])
    theta.append(theta_avg)
    
    # Alpha 8 – 15
    alpha_avg = sum(yf[alpha_start:alpha_end])/len(yf[alpha_start:alpha_end])
    alpha.append(alpha_avg)

    # Beta 16 – 31
    beta_avg = sum(yf[beta_start:beta_end])/len(yf[beta_start:beta_end])
    beta.append(beta_avg)

    # Gamma 32 - 64
    gamma_avg = sum(yf[gamma_start:gamma_end])/len(yf[gamma_start:gamma_end])
    gamma.append(gamma_avg)

    t.append(x[-1])

  # Convert to np.array
  for ar in [t, theta, alpha, beta, gamma]:
    ar = np.array(ar)

  # First figure
  plt.figure(1)

  # Theta
  plt.subplot(221)
  plt.plot(t, theta)
  plt.xlabel('Time (s)')
  plt.ylabel('Magnitude (dB)')
  plt.title('Theta vs. Time')

  # Alpha
  plt.subplot(222)
  plt.plot(t, alpha)
  plt.xlabel('Time (s)')
  plt.ylabel('Magnitude (dB)')
  plt.title('Alpha vs. Time')

  # Beta
  plt.subplot(223)
  plt.plot(t, beta)
  plt.xlabel('Time (s)')
  plt.ylabel('Magnitude (dB)')
  plt.title('Beta vs. Time')

  # Gamma
  plt.subplot(224)
  plt.plot(t, gamma)
  plt.xlabel('Time (s)')
  plt.ylabel('Magnitude (dB)')
  plt.title('Gamma vs. Time')

  plt.grid()
  plt.show(block=True)

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
  data['x'] = data['x'][0:len(data['x'])/factor]
  data['y'] = data['y'][0:len(data['y'])/factor]

  return data

def main():
  if '--data' in sys.argv:
    file_name = sys.argv[2]
    data = get_data(file_name)
    data = restrict_data(data, 4)
    windowed_fft(data)
    # process_data(data)
    # interactive_plot(data)

if __name__ == "__main__":
  main()
