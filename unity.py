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
    # Ignore header
    header = f.readline()

    for input_data in f:
      x, y = [float(tmp.strip()) for tmp in input_data.split(",")[:2]]
      # print x, y
      data['x'].append(x)
      data['y'].append(y)

  return data

def process_data(data):
  # Number of samplepoints
  N = len(data['x'])

  # sample spacing - Play around with this
  T = 1.0 / 800.0
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

def main():
  if '--data' in sys.argv:
    file_name = sys.argv[2]
    data = get_data(file_name)
    process_data(data)

if __name__ == "__main__":
  main()
