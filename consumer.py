# consumer.py
import socket

def main():
    host = '127.0.0.1'
    port = 12345
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    print("Consumer: Connecting to producer...")
    client_socket.connect((host, port))
    print("Consumer: Connected.")

    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            print(f"Consumer: Received {data.decode('utf-8')}")
    except KeyboardInterrupt:
        print("Consumer: Shutting down.")
    finally:
        client_socket.close()

if __name__ == "__main__":
    main()
