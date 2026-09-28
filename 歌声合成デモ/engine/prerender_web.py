"""
ブラウザデモ用に音とスペクトログラムを書き出す。

    web/audio/<曲>_rule.wav    ルールベース（学習なし）
    web/audio/<曲>_model.wav   学習モデル
    web/audio/<本物>_real.wav  PJS の実際の歌声（比較用）
    web/data/spec_*.png        スペクトログラム画像
    web/data/songs.js          曲リストと長さ

使い方:
    python engine/make_songs.py      # 先に UST を用意
    python engine/prerender_web.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare import SR
from synth import parse_ust, OUT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = ROOT / "web"
AUD = WEB / "audio"
DAT = WEB / "data"
CORPUS = ROOT / "data" / "PJS_corpus_ver1.1"

REALS = ["pjs001", "pjs100"]


def load_index():
    """render_all.py が書いた一覧を読む（無ければデモ用4曲だけ）"""
    idx = OUT / "index.json"
    if idx.exists():
        return json.loads(idx.read_text(encoding="utf-8"))
    return [{"key": k, "title": k, "note": "", "notes": 0}
            for k in ("kaeru", "kirakira", "chouchou", "scale")]


def save_wav16(src: Path, dst: Path):
    x, sr = sf.read(str(src), dtype="float32")
    peak = float(np.abs(x).max())
    if peak > 0:
        x = x / peak * 0.92
    sf.write(str(dst), x, sr, subtype="PCM_16")
    return len(x) / sr


def peaks(wav: Path, n: int = 900):
    """波形表示用に、区間ごとの最大振幅を n 点に間引く（0-255）"""
    x, _ = sf.read(str(wav), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    m = len(x) // n
    if m < 1:
        m, n = 1, len(x)
    v = np.abs(x[: m * n]).reshape(n, m).max(1)
    v = v / (v.max() + 1e-9)
    return [int(round(float(a) * 255)) for a in v]


def save_spec(wav: Path, png: Path, seconds: float):
    x, sr = sf.read(str(wav), dtype="float32")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=1024, hop_length=128)), ref=np.max)
    S = S[: int(1024 * 6000 / sr)]                      # 6kHz まで表示
    fig = plt.figure(figsize=(10, 2.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=0)
    fig.savefig(png, facecolor="#0b0e13"); plt.close(fig)


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    DAT.mkdir(parents=True, exist_ok=True)
    entries = []

    for item in load_index():
        key, title, note = item["key"], item["title"], item["note"]
        tracks = {}
        for method, suffix, label in [("rule", "_rule", "ルールベース（学習なし）"),
                                      ("model", "_model", "学習モデル")]:
            src = OUT / f"{key}{suffix}.wav"
            if not src.exists():
                print(f"[warn] {src.name} がありません。synth.py / rule_synth.py を先に実行してください")
                continue
            dur = save_wav16(src, AUD / f"{key}{suffix}.wav")
            save_spec(AUD / f"{key}{suffix}.wav", DAT / f"spec_{key}{suffix}.png", dur)
            tracks[method] = {"wav": f"audio/{key}{suffix}.wav",
                              "png": f"data/spec_{key}{suffix}.png",
                              "peaks": peaks(AUD / f"{key}{suffix}.wav"),
                              "label": label, "sec": round(dur, 2)}
        entries.append({"key": key, "title": title, "note": note,
                        "notes": item.get("notes", 0), "kind": "ust",
                        "group": item.get("group", "demo"), "tracks": tracks})
        print(f"[web] {title}: {item.get('notes', 0)}音符 / {len(tracks)}種類")

    reals = []
    for name in REALS:
        src_real = CORPUS / name / f"{name}_song.wav"
        src_model = OUT / f"{name}_model.wav"
        if not src_real.exists() or not src_model.exists():
            print(f"[warn] {name}: 実データまたはモデル出力がありません（synth.py --copy {name}）")
            continue
        x, sr = sf.read(str(src_real), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x = librosa.resample(x.astype(np.float64), orig_sr=sr, target_sr=SR).astype(np.float32)
        tmp = OUT / f"{name}_real24k.wav"; sf.write(str(tmp), x, SR)
        d1 = save_wav16(tmp, AUD / f"{name}_real.wav")
        d2 = save_wav16(src_model, AUD / f"{name}_model.wav")
        save_spec(AUD / f"{name}_real.wav", DAT / f"spec_{name}_real.png", d1)
        save_spec(AUD / f"{name}_model.wav", DAT / f"spec_{name}_model.png", d2)
        lyric = (CORPUS / name / f"{name}.txt").read_text(encoding="utf-8").strip().splitlines()
        reals.append({"key": name, "title": f"PJS {name[-3:]}番",
                      "note": " / ".join(lyric[:2]),
                      "kind": "real",
                      "tracks": {
                          "real":  {"wav": f"audio/{name}_real.wav",
                                    "png": f"data/spec_{name}_real.png",
                                    "peaks": peaks(AUD / f"{name}_real.wav"),
                                    "label": "本物の歌声（PJS）", "sec": round(d1, 2)},
                          "model": {"wav": f"audio/{name}_model.wav",
                                    "png": f"data/spec_{name}_model.png",
                                    "peaks": peaks(AUD / f"{name}_model.wav"),
                                    "label": "学習モデル（同じ楽譜・同じF0）", "sec": round(d2, 2)}}})
        print(f"[web] {name}: 本物 {d1:.1f}秒 / モデル {d2:.1f}秒")

    meta = {"songs": entries, "reals": reals, "sr": SR,
            "credit": "学習データ：PJS corpus (Junya Koguchi, Shinnosuke Takamichi) / CC BY-SA 4.0"}
    (DAT / "songs.js").write_text("window.SONGS=" + json.dumps(meta, ensure_ascii=False) + ";",
                                  encoding="utf-8")
    print(f"[web] 完了 -> {WEB}")


if __name__ == "__main__":
    main()
