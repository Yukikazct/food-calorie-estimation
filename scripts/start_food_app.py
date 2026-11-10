"""Start the local food-calorie demo, without downloading or opening a camera."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--checkpoint", default="models/food10/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not 0 <= args.port <= 65535:
        parser.error("Invalid port")
    url = f"http://127.0.0.1:{args.port}/"
    if args.port:
        try:
            with urllib.request.urlopen(url + "api/info", timeout=1) as response:
                running = json.load(response)
            if running.get("app") == "food-calorie":
                print("Food app already running: " + url, flush=True)
                if args.open_browser:
                    webbrowser.open(url)
                return
        except (OSError, ValueError):
            pass
    from foodcal.web import FoodApp, create_server
    server = create_server(FoodApp(checkpoint=args.checkpoint, device=args.device), args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print("Food calorie app: " + url, flush=True)
    if args.open_browser:
        timer = threading.Timer(.3, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
