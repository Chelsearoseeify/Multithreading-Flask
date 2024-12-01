# producer.py
import socket
import random
import time

host = '127.0.0.1'
port = 12345
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def wait_for_connection(host, port, server_socket):
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Producer: Waiting for connection...")

def accept_connectionn(server_socket):
    conn, addr = server_socket.accept()
    print(f"Producer: Connected to {addr}")
    return conn

def close_connection(conn, server_socket):
    conn.close()
    server_socket.close()

def main():
    global server_socket, host, port    
    wait_for_connection(host, port, server_socket)
    conn = accept_connectionn(server_socket)

    try:
        while True:
            # Generate random data
            number = random.randint(1, 100)
            conn.sendall(str(number).encode('utf-8'))
            print(f"Producer: Sent {number}")
            time.sleep(1)  # Wait 1 second before sending the next number
    except KeyboardInterrupt:
        print("Producer: Shutting down.")
    finally:
        close_connection(conn, server_socket)

if __name__ == "__main__":
    main()
