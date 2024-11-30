# producer.py
import socket
import random
import time

def main():
    host = '127.0.0.1'
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Producer: Waiting for connection...")

    conn, addr = server_socket.accept()
    print(f"Producer: Connected to {addr}")

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
        conn.close()
        server_socket.close()

if __name__ == "__main__":
    main()
