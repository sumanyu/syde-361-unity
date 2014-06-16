import sys

from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np

def get_data(file_name):
  data = {
    'x': [],
    'y': []
  }

  with open(file_name) as f:
    for input_data in f:
      x, y = [int(tmp.strip()) for tmp in input_data.split(",")]  
      data['x'].append(x)
      data['y'].append(y)

  print data

  return data

def process_data(data):
  # Number of samplepoints
  N = len(data['x'])

  # sample spacing - Play around with this
  # T = 1.0 / 800.0
  # x = np.linspace(0.0, N*T, N)
  # y = 

  # Frequency domain
  # yf = fft(y)
  # xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

  # Plotting
  # plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))

  plt.plot(data['x'], data['y'])
  plt.grid()
  plt.show()

def main():
  if '--data' in sys.argv:
    file_name = sys.argv[2]
    data = get_data(file_name)
    process_data(data)

if __name__ == "__main__":
  main()