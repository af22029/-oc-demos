"""
その場で UST を歌わせるためのローカルサーバ。

    python engine/serve.py

を実行すると http://localhost:8000 でデモが開き、
ページ上の「USTファイルを読み込む」から選んだ楽譜をその場で合成できる。
（事前レンダ済みの4曲は Python なしでも聴けるので、これは追加機能）

UTAU で作った UST をそのまま渡せる。Shift_JIS / UTF-8 どちらでも可。
"""

import hashlib
import http.server
import json
import socketserver
import threading
import webbrowser
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = ROOT / "web"
CACHE = WEB / "cache"
PORT = 8000

import sys
sys.path.insert(0, str(HERE))
from prepare import SR                                   # noqa: E402
from synth import (parse_ust, notes_to_sequence, load_model, render,   # noqa: E402
                   auto_transpose, transpose)
from rule_synth import render_rule                       # noqa: E402
from prerender_web import save_spec, save_wav16, peaks    # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL, CKPT = None, None
LOCK = threading.Lock()


def synth_ust(ust_bytes: bytes, name: str):
    global MODEL, CKPT
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(ust_bytes).hexdigest()[:10]
    tmp = CACHE / f"{tag}.ust"
    tmp.write_bytes(ust_bytes)

    notes = parse_ust(tmp)
    if not notes:
        raise ValueError("音符が1つも読み取れませんでした")
    k = auto_transpose(notes)
    if k:
        notes = transpose(notes, k)
    pid, prv, nxt, pos, dur, lf0, vuv, hz = notes_to_sequence(notes)

    with LOCK:
        if MODEL is None:
            MODEL, CKPT = load_model(DEV)
        wav_model = render(MODEL, CKPT, pid, prv, nxt, pos, dur, lf0, vuv, hz, DEV)
    wav_rule = render_rule(pid, hz, vuv)

    out = {}
    for method, wav in [("model", wav_model), ("rule", wav_rule)]:
        raw = CACHE / f"{tag}_{method}_raw.wav"
        sf.write(str(raw), wav, SR)
        dst = CACHE / f"{tag}_{method}.wav"
        sec = save_wav16(raw, dst)
        raw.unlink(missing_ok=True)
        png = CACHE / f"{tag}_{method}.png"
        save_spec(dst, png, sec)
        out[method] = {"wav": f"cache/{tag}_{method}.wav",
                       "png": f"cache/{tag}_{method}.png",
                       "peaks": peaks(dst),
                       "label": "学習モデル" if method == "model" else "ルールベース（学習なし）",
                       "sec": round(sec, 2)}
    return {"key": tag, "title": name,
            "note": f"読み込んだ楽譜／{len(notes)}音符" + (f"／自動で {k:+d} 半音移調" if k else ""),
            "notes": len(notes), "kind": "ust", "tracks": out}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):
        if "POST" in fmt % args:
            print("[serve]", fmt % args)

    def do_POST(self):
        if self.path != "/render":
            self.send_error(404); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            name = self.headers.get("X-Filename", "song.ust")
            data = self.rfile.read(n)
            print(f"[serve] 合成中: {name}（{n}バイト）")
            res = synth_ust(data, name)
            body = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f"[serve] 完了: {res['notes']}音符 / {res['tracks']['model']['sec']}秒")
        except Exception as e:
            msg = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(msg)
            print(f"[serve] エラー: {e}")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/index.html"
        print(f"[serve] {url} を開きます（終了は Ctrl+C）  device={DEV}")
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] 終了")


if __name__ == "__main__":
    main()
