"""
デモに載せる曲をまとめて合成する。

対象:
    songs/*.ust   … make_songs.py が作るデモ用の曲（パブリックドメイン）
    ust/*.ust     … 自分で用意した UST（ここに置くだけで自動で追加される）

それぞれを「学習モデル」と「ルールベース」の2通りで合成し、
output/ に書き出したうえで output/index.json に一覧を残す。
prerender_web.py はこの一覧を読んでブラウザ用の素材を作る。

使い方:
    python engine/render_all.py                # 未生成のものだけ
    python engine/render_all.py --force        # すべて作り直す
    python engine/render_all.py --max_sec 45   # 長い曲は先頭45秒だけにする
"""

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "output"

import sys
sys.path.insert(0, str(HERE))
from prepare import SR                                              # noqa: E402
from synth import (parse_ust, notes_to_sequence, load_model, render,   # noqa: E402
                   auto_transpose, transpose)
from rule_synth import render_rule                                  # noqa: E402

TITLES = {"kaeru": "かえるのうた", "kirakira": "きらきら星",
          "chouchou": "ちょうちょ", "scale": "音階とあいうえお"}
NOTES = {"kaeru": "ドイツ民謡", "kirakira": "フランス民謡",
         "chouchou": "ドイツ民謡", "scale": "母音の違いを見る用"}


def ascii_key(stem: str, fallback: str) -> str:
    """ファイル名からURLに使えるキーを作る（日本語名なら連番にする）"""
    if any(not c.isascii() for c in stem):      # 日本語名は連番キーにする
        return fallback
    s = unicodedata.normalize("NFKC", stem)
    s = "".join(c for c in s if c.isalnum() or c in "-_")
    return s.strip("-_").lower() or fallback


def trim(notes, max_sec: float):
    if not max_sec:
        return notes
    out, t = [], 0.0
    for n in notes:
        if t + n[0] > max_sec:
            break
        out.append(n); t += n[0]
    return out or notes[:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max_sec", type=float, default=0.0,
                    help="この秒数を超える曲は先頭だけ使う（0＝全部）")
    ap.add_argument("--transpose", type=float, default=None,
                    help="移調（半音）。省略時は曲ごとに自動。0で移調なし")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(dev)
    print(f"[render] device={dev}")

    targets = []
    for i, f in enumerate(sorted((ROOT / "songs").glob("*.ust"))):
        targets.append((ascii_key(f.stem, f"song{i}"), f, TITLES.get(f.stem, f.stem),
                        NOTES.get(f.stem, "デモ用"), "demo"))
    for i, f in enumerate(sorted((ROOT / "ust").glob("*.ust")), 1):
        targets.append((ascii_key(f.stem, f"user{i}"), f, f.stem, "持ち込みの UST", "user"))

    index = []
    for key, path, title, note, group in targets:
        notes = parse_ust(path)
        if not notes:
            print(f"[render] {path.name}: 音符が読めません。スキップ")
            continue
        notes = trim(notes, args.max_sec)
        k = auto_transpose(notes) if args.transpose is None else args.transpose
        if k:
            notes = transpose(notes, k)
        seq = notes_to_sequence(notes)
        sec = len(seq[0]) * 5 / 1000

        for method in ("model", "rule"):
            dst = OUT / f"{key}_{method}.wav"
            if dst.exists() and not args.force and dst.stat().st_mtime > path.stat().st_mtime:
                continue
            if method == "model":
                wav = render(model, ckpt, *seq[:7], seq[7], dev)
            else:
                wav = render_rule(seq[0], seq[7], seq[6])
            sf.write(str(dst), wav, SR)
        note_txt = note + (f"／自動で {k:+.0f} 半音移調" if k else "")
        index.append({"key": key, "title": title, "note": note_txt, "group": group,
                      "notes": len(notes), "sec": round(sec, 2),
                      "transpose": k, "ust": str(path)})
        print(f"[render] {title}: {len(notes)}音符 / {sec:.1f}秒 / 移調 {k:+.0f} -> {key}_*.wav")

    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"[render] 一覧 -> {OUT / 'index.json'}（{len(index)}曲）")


if __name__ == "__main__":
    main()
