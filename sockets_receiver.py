import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

def main():
	sock = socket.socket(socket.AF_INET, # Internet
	                     socket.SOCK_DGRAM) # UDP
	sock.bind((UDP_IP, UDP_PORT))

	while True:
		# We can play around with buffer sizes
	    data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
	    print "received message:", data
	
if __name__ == "__main__":
  main()