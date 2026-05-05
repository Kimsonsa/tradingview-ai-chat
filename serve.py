"""
TradeAI PWA Shell Server
Streamlit 앱(8502)을 감싸는 PWA 셸을 서빙하는 경량 서버.
API 로직은 Streamlit이 처리하므로 여기서는 정적 파일만 서빙합니다.
"""

import http.server
import os
import sys
from pathlib import Path

PORT = 3000
PWA_DIR = Path(__file__).parent / "pwa"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".webmanifest": "application/manifest+json",
}


class PWAHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"  {args[0]}")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"

        file_path = PWA_DIR / path.lstrip("/")

        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(PWA_DIR.resolve())):
                self.send_error(403)
                return
        except Exception:
            self.send_error(400)
            return

        if not file_path.is_file():
            file_path = PWA_DIR / "index.html"

        ext = file_path.suffix.lower()
        content_type = MIME_TYPES.get(ext, "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            if path == "/sw.js":
                self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))


def main():
    if not PWA_DIR.is_dir():
        print(f"ERROR: {PWA_DIR} not found")
        sys.exit(1)

    server = http.server.HTTPServer(("0.0.0.0", PORT), PWAHandler)
    print(f"\n  TradeAI PWA Shell Server")
    print(f"  http://localhost:{PORT}")
    print(f"  Streamlit -> http://localhost:8502")
    print(f"  Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped")
        server.server_close()


if __name__ == "__main__":
    main()
