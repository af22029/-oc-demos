"""
PJS corpus の歌唱データを WORLD 特徴量＋音素ラベルに変換する。

出力: features/pjsXXX.npz
    mcep   (T, 40)  スペクトル包絡（メルケプストラム）
    apc    (T, 4)   非周期性
    lf0    (T,)     log F0（無声区間は補間済み）
    vuv    (T,)     有声/無声フラグ
    pid    (T,)     音素ID
    pos    (T,)     その音素の中での位置 0〜1
    dur    (T,)     その音素の長さ（秒）
    prev / next (T,) 前後の音素ID

使い方:
    python engine/prepare.py             # 全100曲
    python engine/prepare.py --check 1   # 1曲だけ分析→再合成して音質を確認
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
import librosa

from dsp import MelWarp, sp_to_mcep, mcep_to_sp, ap_to_code, code_to_ap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPUS = ROOT / "data" / "PJS_corpus_ver1.1"
FEAT = ROOT / "features"

SR = 24000          # WORLD 処理用（元は48kHz）
FRAME_MS = 5.0
MCEP_ORDER = 60
AP_ORDER = 4
F0_FLOOR, F0_CEIL = 100.0, 800.0     # 女性歌手の音域

# PJS に出てくる36音素（頻度順は問わない、IDは固定）
PHONES = ["pau", "cl", "xx",
          "a", "i", "u", "e", "o", "N",
          "k", "ky", "g", "gy", "s", "sh", "z", "j", "t", "ts", "ch", "d",
          "n", "ny", "h", "hy", "f", "b", "by", "p", "py", "m", "my",
          "y", "r", "ry", "w"]
PID = {p: i for i, p in enumerate(PHONES)}
VOWELS = {"a", "i", "u", "e", "o", "N"}


def read_lab(path: Path):
    """HTK ラベル（100ns単位）→ [(開始秒, 終了秒, 音素)]"""
    out = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        s, e, p = line.split()
        out.append((int(s) / 1e7, int(e) / 1e7, p))
    return out


def analyze(wav_path: Path):
    x, sr = sf.read(str(wav_path), dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    x = np.ascontiguousarray(x)
    f0, t = pw.harvest(x, SR, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=FRAME_MS)
    f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    return x, f0, sp, ap


def interp_lf0(f0):
    """無声区間の log F0 を線形補間（モデルが学習しやすくなる）"""
    vuv = (f0 > 0).astype(np.float32)
    lf0 = np.zeros_like(f0)
    idx = np.where(f0 > 0)[0]
    if len(idx) == 0:
        return lf0.astype(np.float32), vuv
    lf0[idx] = np.log(f0[idx])
    lf0 = np.interp(np.arange(len(f0)), idx, lf0[idx])
    return lf0.astype(np.float32), vuv


def frame_labels(lab, n_frames):
    """フレームごとの音素ID・音素内位置・長さ・前後音素"""
    pid = np.zeros(n_frames, np.int64)
    pos = np.zeros(n_frames, np.float32)
    dur = np.zeros(n_frames, np.float32)
    prv = np.zeros(n_frames, np.int64)
    nxt = np.zeros(n_frames, np.int64)
    for k, (s, e, p) in enumerate(lab):
        i0 = int(round(s * 1000 / FRAME_MS))
        i1 = min(n_frames, int(round(e * 1000 / FRAME_MS)))
        if i1 <= i0:
            continue
        pid[i0:i1] = PID.get(p, PID["xx"])
        dur[i0:i1] = e - s
        pos[i0:i1] = np.linspace(0, 1, i1 - i0, endpoint=False)
        prv[i0:i1] = PID.get(lab[k - 1][2], PID["pau"]) if k > 0 else PID["pau"]
        nxt[i0:i1] = PID.get(lab[k + 1][2], PID["pau"]) if k + 1 < len(lab) else PID["pau"]
    return pid, pos, dur, prv, nxt


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--check", type=int, default=0,
                     help="N曲だけ分析→再合成して check/ に書き出す（音質確認用）")
    args = ap_.parse_args()

    FEAT.mkdir(exist_ok=True)
    warp = MelWarp(n_bins=pw.get_cheaptrick_fft_size(SR) // 2 + 1, sr=SR)
    print(f"[prepare] SR={SR} フレーム={FRAME_MS}ms スペクトル次元={warp.n_bins}")

    songs = sorted(glob.glob(str(CORPUS / "pjs*" / "pjs*_song.wav")))
    if not songs:
        raise SystemExit(f"[ERROR] コーパスが見つかりません: {CORPUS}")

    if args.check:
        out = ROOT / "check"; out.mkdir(exist_ok=True)
        for wav in songs[: args.check]:
            name = Path(wav).stem
            x, f0, sp, apx = analyze(Path(wav))
            mc = sp_to_mcep(sp, warp, MCEP_ORDER)
            ac = ap_to_code(apx, warp, AP_ORDER)
            y = pw.synthesize(f0, mcep_to_sp(mc, warp), code_to_ap(ac, warp), SR, FRAME_MS)
            sf.write(str(out / f"{name}_orig.wav"), x, SR)
            sf.write(str(out / f"{name}_resynth.wav"), y, SR)
            err = np.mean((np.log(sp + 1e-10) - np.log(mcep_to_sp(mc, warp) + 1e-10)) ** 2)
            print(f"[check] {name}: 対数スペクトル二乗誤差={err:.4f} -> {out}")
        return

    total = 0.0
    for i, wav in enumerate(songs, 1):
        name = Path(wav).stem.replace("_song", "")
        lab = read_lab(Path(wav).parent / f"{name}.lab")
        x, f0, sp, apx = analyze(Path(wav))
        n = len(f0)
        mc = sp_to_mcep(sp, warp, MCEP_ORDER).astype(np.float32)
        ac = ap_to_code(apx, warp, AP_ORDER).astype(np.float32)
        lf0, vuv = interp_lf0(f0)
        pid, pos, dur, prv, nxt = frame_labels(lab, n)
        np.savez_compressed(FEAT / f"{name}.npz", mcep=mc, apc=ac, lf0=lf0, vuv=vuv,
                            pid=pid, pos=pos, dur=dur, prev=prv, next=nxt)
        total += n * FRAME_MS / 1000
        if i % 10 == 0 or i == len(songs):
            print(f"[prepare] {i}/{len(songs)} 完了 ({total / 60:.1f}分)")

    meta = {"sr": SR, "frame_ms": FRAME_MS, "mcep_order": MCEP_ORDER, "ap_order": AP_ORDER,
            "n_bins": warp.n_bins, "phones": PHONES, "vowels": sorted(VOWELS),
            "n_songs": len(songs), "total_min": round(total / 60, 2)}
    (FEAT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare] 完了: {len(songs)}曲 / {total / 60:.1f}分 -> {FEAT}")


if __name__ == "__main__":
    main()
