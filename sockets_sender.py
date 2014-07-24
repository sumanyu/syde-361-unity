import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
MESSAGE = "Hello, World!"

def main():
	print "UDP target IP:", UDP_IP
	print "UDP target port:", UDP_PORT

	# We're using UDP because we don't need ack's and it should be faster than TCP
	sock = socket.socket(socket.AF_INET, # Internet
	                     socket.SOCK_DGRAM) # UDP

	while True:
		message = raw_input('Message to send: ')
		sock.sendto(message, (UDP_IP, UDP_PORT))
	
if __name__ == "__main__":
  main()