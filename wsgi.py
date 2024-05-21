import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial Object Detection")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()