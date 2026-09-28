"""
比較用：ルールベースのフォルマント合成（学習を一切使わない）

学習モデル版（synth.py）と条件を揃えるため、
    ・同じ UST パーサ
    ・同じ F0 軌跡（ポルタメント・ビブラート）
    ・同じ WORLD ボコーダ
を使い、**スペクトル包絡の作り方だけ**を変える。

    学習モデル版 : 音素とF0 → ニューラルネット → 包絡
    ルールベース : 音素 → 表引きしたフォルマント3本 → 数式で包絡

つまり2つの音の違いは、まるごと「包絡をどう決めたか」の違いになる。

使い方:
    python engine/rule_synth.py --ust songs/kaeru.ust
"""

import argparse
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

from prepare import PHONES, SR, FRAME_MS
from synth import parse_ust, notes_to_sequence, OUT

# 母音のフォルマント（F1, F2, F3）[Hz]：成人女性のおおよその値
FORMANT = {
    "a": (850, 1200, 2800), "i": (300, 2700, 3300), "u": (380, 1150, 2600),
    "e": (500, 2300, 2900), "o": (450,  800, 2700), "N": (250,  1000, 2200),
}
BW = (80, 90, 130)              # 各共鳴の帯域幅
AMP = (1.0, 0.55, 0.28)         # 高い共鳴ほど弱い
TILT_DB_PER_OCT = -10.0         # 声帯音源の高域減衰

# 子音の扱い：摩擦音は雑音、破裂音は無音＋短い雑音、鼻音は低域のみ
FRICATIVE = {"s", "sh", "h", "f", "z", "j", "ts", "ch"}
PLOSIVE = {"k", "ky", "t", "p", "py", "b", "by", "d", "g", "gy"}
NASAL = {"n", "ny", "m", "my", "N"}


def envelope_for(phone: str, freqs: np.ndarray):
    """1フレーム分のスペクトル包絡を数式で作る"""
    if phone in ("pau", "cl"):
        return np.full_like(freqs, 1e-8)

    if phone in FRICATIVE:                       # 高域中心の雑音スペクトル
        env = 1e-6 + 1e-4 * (freqs / 8000.0) ** 2
        return env
    if phone in PLOSIVE:                         # ごく弱い広帯域
        return np.full_like(freqs, 3e-6)

    base = FORMANT.get(phone)
    if base is None:                             # 半母音・流音は隣の母音に寄せた既定値
        base = FORMANT["a"] if phone in ("y", "w", "r", "ry") else FORMANT["e"]
    if phone in NASAL:
        base = FORMANT["N"]

    env = np.zeros_like(freqs)
    for f, bw, a in zip(base, BW, AMP):
        env += a * (bw / 2) ** 2 / ((freqs - f) ** 2 + (bw / 2) ** 2)
    tilt = 10 ** (TILT_DB_PER_OCT * np.log2(np.maximum(freqs, 50) / 500) / 20)
    return (env * tilt) ** 2 * 1e-3 + 1e-9


def render_rule(pid, hz, vuv):
    n_bins = pw.get_cheaptrick_fft_size(SR) // 2 + 1
    freqs = np.linspace(0, SR / 2, n_bins)
    cache = {p: envelope_for(p, freqs) for p in PHONES}

    sp = np.stack([cache[PHONES[i]] for i in pid])
    # 音素の切り替わりを 30ms でなめらかに繋ぐ（そうしないとブツ切りになる）
    k = int(30 / FRAME_MS)
    ker = np.hanning(k * 2 + 1); ker /= ker.sum()
    sp = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, sp)

    ap = np.zeros_like(sp) + 1e-3
    for i, p in enumerate(pid):
        ph = PHONES[p]
        if ph in FRICATIVE or ph in PLOSIVE or ph == "pau":
            ap[i] = 0.999                        # 完全に雑音として鳴らす
    f0 = np.ascontiguousarray(np.where(vuv > 0.5, hz, 0.0).astype(np.float64))
    wav = pw.synthesize(f0, np.ascontiguousarray(sp), np.ascontiguousarray(ap), SR, FRAME_MS)
    peak = np.abs(wav).max()
    return wav / peak * 0.9 if peak > 0 else wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ust", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    notes = parse_ust(Path(args.ust))
    pid, prv, nxt, pos, dur, lf0, vuv, hz = notes_to_sequence(notes)
    wav = render_rule(pid, hz, vuv)
    out = Path(args.out or OUT / (Path(args.ust).stem + "_rule.wav"))
    sf.write(str(out), wav, SR)
    print(f"[rule] {Path(args.ust).name} -> {out}（{len(wav) / SR:.1f}秒）")


if __name__ == "__main__":
    main()
