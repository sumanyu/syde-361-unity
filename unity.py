import sys

def get_data(file_name):
  data = []
  with open(file_name) as f:
    pass

  return data

def main():
  if '--data' in sys.argv:
    file_name = sys.argv[1]
    data = get_data(file_name)

if __name__ == "__main__":
  main()