"""Loopback-only style picker; writes only the shared style selection config."""

from __future__ import annotations

import argparse
import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from agentic.app.style_catalog import SelectionConflict, StyleLibrary


WEB_ROOT = Path(__file__).with_name("style_picker_web")


def create_server(*, library: StyleLibrary | None = None, port: int = 8766) -> ThreadingHTTPServer:
    library = library or StyleLibrary()
    library.snapshot()
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' https://matplinta.github.io; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)

        def _json(self, status: int, data: dict) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _valid_host(self) -> bool:
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}",
            }

        def do_GET(self) -> None:
            if not self._valid_host():
                self._json(403, {"error": "只接受本機頁面的請求。"})
                return
            path = urlsplit(self.path).path
            if path == "/api/styles":
                try:
                    self._json(200, {**library.snapshot(), "token": token})
                except (ValueError, OSError) as exc:
                    self._json(500, {"error": str(exc)})
                return
            files = {"/": ("index.html", "text/html"), "/app.css": ("app.css", "text/css"),
                     "/app.js": ("app.js", "text/javascript")}
            if path not in files:
                self._json(404, {"error": "找不到頁面。"})
                return
            filename, content_type = files[path]
            self._send(200, (WEB_ROOT / filename).read_bytes(), content_type + "; charset=utf-8")

        def do_PUT(self) -> None:
            origin = self.headers.get("Origin")
            allowed_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if not self._valid_host() or (origin and origin not in allowed_origins):
                self._json(403, {"error": "只接受本機頁面的儲存請求。"})
                return
            if urlsplit(self.path).path != "/api/selection":
                self._json(404, {"error": "找不到儲存入口。"})
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("請使用 JSON 儲存選擇。")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("儲存內容的大小不正確。")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("儲存內容必須是 JSON 物件。")
                supplied_token = data.get("token")
                if not isinstance(supplied_token, str) or not hmac.compare_digest(supplied_token, token):
                    self._json(403, {"error": "頁面驗證已失效，請重新載入。"})
                    return
                self._json(200, library.save(data.get("enabled_ids"), revision=data.get("revision")))
            except SelectionConflict as exc:
                self._json(409, {"error": str(exc)})
            except (ValueError, UnicodeDecodeError) as exc:
                self._json(400, {"error": str(exc)})
            except OSError:
                self._json(500, {"error": "無法寫入設定，選擇尚未儲存。請檢查設定檔權限後重試。"})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="勾選 Krea 2 隨機生成風格")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    with create_server(port=args.port) as server:
        print(f"Style picker: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
