#!/usr/bin/env python3
"""Minimal zero-dependency HTTP server for the v9m1 interactive demo.

    python demo/server.py --port 4174
"""
import argparse
import functools
import http.server
import os
import socketserver


class DemoHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml",
        ".json": "application/json; charset=utf-8",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description="Serve the v9m1 interactive demo.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "4174")))
    args = ap.parse_args()

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(DemoHandler, directory=demo_dir)
    with ReusableTCPServer((args.host, args.port), handler) as httpd:
        print(f"v9m1 demo listening on http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
