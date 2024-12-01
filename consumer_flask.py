from flask import Flask, render_template, jsonify
import socket
import threading
import time
from multiprocessing import Manager
import json

def create_app():
    app = Flask(__name__)

    # Use a multiprocessing manager to handle shared state
    manager = Manager()
    shared_state = manager.dict()
    shared_state["latest_data"] = "No data yet."

    def consume_data():
        """Connect to the producer and receive data."""
        host = '127.0.0.1'
        port = 12345
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        print("Consumer: Connecting to producer...")
        while True:
            try:
                client_socket.connect((host, port))
                break
            except ConnectionRefusedError:
                print("Consumer: Connection refused. Retrying in 1 second...")
                time.sleep(1)

        print("Consumer: Connected to producer.")

        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                # Update the shared dictionary with the new data
                shared_state["latest_data"] = json.loads(data.decode('utf-8'))
                print(f"Consumer: Received {shared_state['latest_data']}")
        except KeyboardInterrupt:
            print("Consumer: Shutting down.")
        finally:
            client_socket.close()

    @app.route('/')
    def index():
        """Serve the HTML page."""
        return render_template('index.html')

    @app.route('/data')
    def get_data():
        """Serve the latest data."""
        data_to_return = shared_state["latest_data"]
        print(f"get_data called. Latest data: {data_to_return}")
        return jsonify({"data": data_to_return})

    def start_consumer():
        """Start the consumer in a separate thread."""
        consumer_thread = threading.Thread(target=consume_data)
        consumer_thread.daemon = True
        consumer_thread.start()

    start_consumer()
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=False)
