"""
蝉ミキサー デモ用 事前レンダリングスクリプト（研究フォルダとは完全独立）

engine/snapshot/ にコピー済みの学習済みモデルだけを読み、
web/audio/ と web/data/ に静的ファイルを書き出す。
D:\GAN V3 や振幅スペクトログラム側には一切書き込まない。

出力:
    web/audio/grid_XX_YY.wav   潜在空間マップの格子点の音（GRID x GRID 個）
    web/audio/morph_NN.wav     2点間モーフィングの音（MORPH_STEPS 個）
    web/data/map.json          格子座標・実データ散布図・特徴量・ファイル名
    web/data/spec_XX_YY.png    各格子点のスペクトログラム画像

使い方:
    python engine/prerender.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

HERE     = Path(__file__).resolve().parent
SNAP     = HERE / "snapshot"
WEB      = HERE.parent / "web"
AUD      = WEB / "audio"
DAT      = WEB / "data"

GRID         = 11      # 格子の一辺（11x11 = 121 音）
MORPH_STEPS  = 21      # モーフィング段数
EQ_GAIN      = 0.76    # 提案手法の最終設定（4-8kHz 補正）
PCTL         = (5, 95) # 格子が覆う範囲（実データ z の分位点）

sys.path.insert(0, str(SNAP))
from model import ComplexSpecVAE_V3   # snapshot 内のコピーを読む


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    DAT.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((SNAP / "config.json").read_text(encoding="utf-8"))
    sr         = cfg["audio"]["target_sr"]
    n_fft      = cfg["stft"]["n_fft"]
    hop_length = cfg["stft"]["hop_length"]
    win_length = cfg["stft"]["win_length"]
    seg_samples = int(sr * cfg["stft"]["segment_sec"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prerender] device={device}")

    window = torch.hann_window(win_length).to(device)
    with torch.no_grad():
        T_fixed = torch.stft(torch.zeros(seg_samples), n_fft, hop_length, win_length,
                             window=torch.hann_window(win_length),
                             return_complex=True).shape[-1]

    ckpt = torch.load(SNAP / "model_4000.pt", map_location=device)
    real_mean, real_std = float(ckpt["real_mean"]), float(ckpt["real_std"])
    imag_mean, imag_std = float(ckpt["imag_mean"]), float(ckpt["imag_std"])

    vae = ComplexSpecVAE_V3(cfg, T_fixed).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    vae.eval()

    z_lib = np.load(SNAP / "z_vectors.npy")           # (N, 8) 実データの潜在ベクトル
    target_rms = float(np.load(SNAP / "eval_rms_stats.npy").mean())
    print(f"[prerender] z_lib={z_lib.shape}, target_rms={target_rms:.4f}")

    # ── PCA: 8次元 → 2次元の地図 ────────────────────────────────
    pca = PCA(n_components=2).fit(z_lib)
    z2d = pca.transform(z_lib)
    print(f"[prerender] 寄与率: PC1={pca.explained_variance_ratio_[0]:.3f} "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    lo = np.percentile(z2d, PCTL[0], axis=0)
    hi = np.percentile(z2d, PCTL[1], axis=0)
    xs = np.linspace(lo[0], hi[0], GRID)
    ys = np.linspace(lo[1], hi[1], GRID)

    eq_lo = round(4000 * n_fft / sr)
    eq_hi = round(8000 * n_fft / sr)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    def synth(z8: np.ndarray):
        """潜在ベクトル(8次元) → 波形, 対数振幅スペクトログラム, 特徴量"""
        z = torch.from_numpy(z8.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            spec = vae.decode(z)
            re = spec[0, 0] * (real_std + 1e-8) + real_mean
            im = spec[0, 1] * (imag_std + 1e-8) + imag_mean
            cs = torch.complex(re.float(), im.float())
            cs[eq_lo:eq_hi, :] *= EQ_GAIN
            wav = torch.istft(cs, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, window=window, length=seg_samples)
            spec_eval = torch.stft(wav, n_fft=n_fft, hop_length=hop_length,
                                   win_length=n_fft, window=window, return_complex=True)
            mag = spec_eval.abs()
            cur = float(mag.T.pow(2).mean(dim=1).sqrt().mean().cpu())
        wav_np = wav.cpu().numpy()
        if cur > 1e-8:
            wav_np = wav_np * (target_rms / cur)
        mag_np = mag.cpu().numpy()
        centroid = float((mag_np * freqs[:, None]).sum() / (mag_np.sum() + 1e-8))
        return wav_np, np.log10(mag_np + 1e-5), centroid

    def save_wav(path: Path, wav_np: np.ndarray):
        peak = float(np.abs(wav_np).max())
        w = wav_np / peak * 0.9 if peak > 0.9 else wav_np   # 再生用にクリップ回避
        sf.write(str(path), w.astype(np.float32), sr, subtype="PCM_16")

    def save_spec(path: Path, logmag: np.ndarray):
        fig = plt.figure(figsize=(2.4, 1.6), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.imshow(logmag, origin="lower", aspect="auto", cmap="magma")
        fig.savefig(path, transparent=False); plt.close(fig)

    # ── 格子点の生成 ────────────────────────────────────────────
    cells = []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            z8 = pca.inverse_transform(np.array([[x, y]]))[0]
            wav_np, logmag, centroid = synth(z8)
            name = f"grid_{ix:02d}_{iy:02d}"
            save_wav(AUD / f"{name}.wav", wav_np)
            save_spec(DAT / f"spec_{ix:02d}_{iy:02d}.png", logmag)
            cells.append({"ix": ix, "iy": iy, "x": float(x), "y": float(y),
                          "centroid": round(centroid, 1),
                          "wav": f"audio/{name}.wav",
                          "png": f"data/spec_{ix:02d}_{iy:02d}.png"})
        print(f"[prerender] grid row {iy + 1}/{GRID} 完了")

    # ── モーフィング（実データの両端 2 点を結ぶ）────────────────
    a_idx = int(np.argmin(z2d[:, 0]))
    b_idx = int(np.argmax(z2d[:, 0]))
    morph = []
    for k in range(MORPH_STEPS):
        t = k / (MORPH_STEPS - 1)
        z8 = (1 - t) * z_lib[a_idx] + t * z_lib[b_idx]
        wav_np, _, centroid = synth(z8)
        name = f"morph_{k:02d}"
        save_wav(AUD / f"{name}.wav", wav_np)
        morph.append({"t": round(t, 3), "centroid": round(centroid, 1),
                      "wav": f"audio/{name}.wav"})
    print("[prerender] morph 完了")

    meta = {
        "grid": GRID,
        "xs": [float(v) for v in xs],
        "ys": [float(v) for v in ys],
        "cells": cells,
        "morph": morph,
        "real_points": [[float(p[0]), float(p[1])] for p in z2d],
        "explained": [float(v) for v in pca.explained_variance_ratio_[:2]],
        "sr": sr,
        "note": "model_4000.pt (ComplexSpecVAE-GAN V3, EQ=0.76) のスナップショットから生成",
    }
    (DAT / "map.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"[prerender] 完了: {len(cells)} 格子点 + {len(morph)} モーフ -> {WEB}")


if __name__ == "__main__":
    main()
