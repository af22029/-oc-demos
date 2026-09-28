"""
学習済みモデルで歌わせる。

2つのモード:
    --copy pjs100    実データの音素ラベルとF0をそのまま与えて再合成（品質確認用）
    --ust song.ust   UTAU の UST ファイルを読んで歌わせる

どちらも「音素列 + F0」→ モデル → スペクトル包絡 → WORLD → wav という同じ道筋。

使い方:
    python engine/synth.py --copy pjs100
    python engine/synth.py --ust songs/sample.ust --out out.wav
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
import torch

from dsp import MelWarp, mcep_to_sp, code_to_ap
from train import AcousticModel
from prepare import PID, PHONES, SR, FRAME_MS, MCEP_ORDER, read_lab, interp_lf0

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FEAT = ROOT / "features"
OUT = ROOT / "output"

# かな → 音素（UTAU の歌詞用。拗音・撥音・促音まで対応）
KANA = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "k a", "き": "k i", "く": "k u", "け": "k e", "こ": "k o",
    "が": "g a", "ぎ": "g i", "ぐ": "g u", "げ": "g e", "ご": "g o",
    "さ": "s a", "し": "sh i", "す": "s u", "せ": "s e", "そ": "s o",
    "ざ": "z a", "じ": "j i", "ず": "z u", "ぜ": "z e", "ぞ": "z o",
    "た": "t a", "ち": "ch i", "つ": "ts u", "て": "t e", "と": "t o",
    "だ": "d a", "ぢ": "j i", "づ": "z u", "で": "d e", "ど": "d o",
    "な": "n a", "に": "n i", "ぬ": "n u", "ね": "n e", "の": "n o",
    "は": "h a", "ひ": "h i", "ふ": "f u", "へ": "h e", "ほ": "h o",
    "ば": "b a", "び": "b i", "ぶ": "b u", "べ": "b e", "ぼ": "b o",
    "ぱ": "p a", "ぴ": "p i", "ぷ": "p u", "ぺ": "p e", "ぽ": "p o",
    "ま": "m a", "み": "m i", "む": "m u", "め": "m e", "も": "m o",
    "や": "y a", "ゆ": "y u", "よ": "y o",
    "ら": "r a", "り": "r i", "る": "r u", "れ": "r e", "ろ": "r o",
    "わ": "w a", "を": "o", "ん": "N", "っ": "cl", "ー": "-",
    "きゃ": "ky a", "きゅ": "ky u", "きょ": "ky o",
    "ぎゃ": "gy a", "ぎゅ": "gy u", "ぎょ": "gy o",
    "しゃ": "sh a", "しゅ": "sh u", "しょ": "sh o",
    "じゃ": "j a", "じゅ": "j u", "じょ": "j o",
    "ちゃ": "ch a", "ちゅ": "ch u", "ちょ": "ch o",
    "にゃ": "ny a", "にゅ": "ny u", "にょ": "ny o",
    "ひゃ": "hy a", "ひゅ": "hy u", "ひょ": "hy o",
    "びゃ": "by a", "びゅ": "by u", "びょ": "by o",
    "ぴゃ": "py a", "ぴゅ": "py u", "ぴょ": "py o",
    "みゃ": "my a", "みゅ": "my u", "みょ": "my o",
    "りゃ": "ry a", "りゅ": "ry u", "りょ": "ry o",
}
CONSONANT_SEC = 0.06        # 子音に割り当てる長さ
VOWELS = set("aiueoN")


def lyric_to_phones(lyric: str):
    """歌詞（かな）→ 音素リスト。UTAU の連続音表記 'a か' なども拾う。"""
    s = lyric.strip()
    if not s or s in ("R", "r", "休", "-"):
        return ["pau"]
    s = s.split()[-1]                                  # 'a か' → 'か'
    if re.fullmatch(r"[a-zA-Z]+( [a-zA-Z]+)*", s):     # すでに音素表記
        return [p for p in s.split() if p in PID]
    out = []
    i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two in KANA:
            out += KANA[two].split(); i += 2
        elif s[i] in KANA:
            v = KANA[s[i]]
            out += ([out[-1]] if v == "-" and out else v.split())
            i += 1
        else:
            i += 1
    return out or ["pau"]


def parse_ust(path: Path):
    """UST → [(音符長さ秒, 音高Hz, 歌詞)]。テンポは [#SETTING] または各音符の Tempo。"""
    raw = path.read_bytes()
    try:                                   # UTF-8 で読めれば UTF-8、駄目なら UTAU 標準の Shift_JIS
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("shift_jis", errors="ignore")
    tempo = 120.0
    m = re.search(r"^Tempo=([\d.]+)", txt, re.M)
    if m:
        tempo = float(m.group(1))
    notes = []
    for block in re.split(r"\[#", txt)[1:]:
        head, *rest = block.split("\n")
        if head.strip().upper() in ("SETTING", "VERSION", "TRACKEND", "PREV", "NEXT"):
            continue
        d = dict(re.findall(r"^([A-Za-z0-9]+)=(.*)$", "\n".join(rest), re.M))
        if "Length" not in d:
            continue
        if "Tempo" in d and d["Tempo"]:
            tempo = float(d["Tempo"])
        length = float(d["Length"]) / 480.0 * (60.0 / tempo)   # 480tick = 4分音符
        lyric = d.get("Lyric", "")
        rest_note = lyric.strip() in ("R", "r", "休", "")
        note = int(d.get("NoteNum", 60))
        hz = 440.0 * 2 ** ((note - 69) / 12)
        notes.append((length, 0.0 if rest_note else hz, lyric))
    return notes


# 学習データ（PJS）が得意な音域。ここに収まるほど発音がはっきりする
TRAIN_LO, TRAIN_HI = 48.0, 67.0


def hz_to_midi(hz):
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def transpose(notes, semitones: float):
    """音符の高さを半音単位でずらす（歌詞と長さはそのまま）"""
    if not semitones:
        return notes
    r = 2.0 ** (semitones / 12.0)
    return [(l, hz * r if hz > 0 else 0.0, ly) for l, hz, ly in notes]


def auto_transpose(notes, lo=TRAIN_LO, hi=TRAIN_HI, search=(-14, 5)):
    """学習音域からのはみ出し時間がいちばん短くなる移調量を選ぶ（半音単位）"""
    v = [(l, hz_to_midi(hz)) for l, hz, _ in notes if hz > 0]
    if not v:
        return 0
    best, best_cost = 0, None
    for k in range(search[0], search[1] + 1):
        cost = sum(l for l, m in v if not (lo <= m + k <= hi))
        # 同点なら移調量の小さいほうを選ぶ
        if best_cost is None or cost < best_cost - 1e-9 or (abs(cost - best_cost) < 1e-9 and abs(k) < abs(best)):
            best, best_cost = k, cost
    return best


def notes_to_sequence(notes, vibrato=True, portamento=0.06):
    """音符列 → フレーム単位の (音素ID列, 位置, 長さ, logF0, 有声フラグ)"""
    fps = 1000.0 / FRAME_MS
    seq_ph, seq_dur = [], []            # (音素, 長さ秒)
    f0_pts = []                         # (時刻, Hz)
    t = 0.0
    last_vowel = "a"
    for length, hz, lyric in notes:
        phs = lyric_to_phones(lyric) if hz > 0 else ["pau"]
        # 長音「ー」や未知の歌詞は、直前の母音を伸ばす扱いにする
        phs = [last_vowel if p == "-" else p for p in phs]
        phs = [p for p in phs if p in PID] or [last_vowel]
        for p in phs:
            if p in VOWELS:
                last_vowel = p
        cons = [p for p in phs if p not in VOWELS]
        vows = [p for p in phs if p in VOWELS] or ["a"]
        c_len = min(CONSONANT_SEC * len(cons), length * 0.4)
        v_len = max(length - c_len, 0.02)
        for p in cons:
            seq_ph.append(p); seq_dur.append(c_len / max(1, len(cons)))
        for p in vows:
            seq_ph.append(p); seq_dur.append(v_len / len(vows))
        f0_pts.append((t, hz)); f0_pts.append((t + length, hz))
        t += length

    total = sum(seq_dur)
    n = int(round(total * fps))
    pid = np.zeros(n, np.int64); prv = np.zeros(n, np.int64); nxt = np.zeros(n, np.int64)
    pos = np.zeros(n, np.float32); dur = np.zeros(n, np.float32)
    i = 0
    for k, (p, dsec) in enumerate(zip(seq_ph, seq_dur)):
        m = max(1, int(round(dsec * fps)))
        j = min(n, i + m)
        if j <= i:
            continue
        pid[i:j] = PID.get(p, PID["pau"]); dur[i:j] = dsec
        pos[i:j] = np.linspace(0, 1, j - i, endpoint=False)
        prv[i:j] = PID.get(seq_ph[k - 1], PID["pau"]) if k > 0 else PID["pau"]
        nxt[i:j] = PID.get(seq_ph[k + 1], PID["pau"]) if k + 1 < len(seq_ph) else PID["pau"]
        i = j

    # F0 軌跡：音符間をなめらかに繋ぎ、ビブラートを乗せる
    tt = np.arange(n) / fps
    xs = np.array([p[0] for p in f0_pts]); ys = np.array([p[1] for p in f0_pts])
    hz = np.interp(tt, xs, ys)
    voiced = hz > 0
    if portamento > 0 and voiced.any():
        k = max(1, int(portamento * fps))
        ker = np.hanning(k * 2 + 1); ker /= ker.sum()
        sm = np.convolve(np.where(voiced, hz, np.nan_to_num(hz)), ker, mode="same")
        hz = np.where(voiced, sm, hz)
    if vibrato:
        # 実際の歌手と同じく、音符が始まってすぐには掛けず 0.25秒 かけて深くする
        depth, rate, delay, rise = 0.022, 5.5, 0.25, 0.35
        env = np.zeros(n, np.float32)
        t0 = 0.0
        for length, note_hz, _ in notes:
            if note_hz > 0 and length > delay:
                i0 = int(t0 * fps); i1 = min(n, int((t0 + length) * fps))
                age = (np.arange(i1 - i0) / fps) - delay
                env[i0:i1] = np.clip(age / rise, 0, 1)
            t0 += length
        hz = hz * (1 + depth * env * np.sin(2 * np.pi * rate * tt) * voiced)
    lf0 = np.zeros(n, np.float32); lf0[voiced] = np.log(hz[voiced])
    if voiced.any():
        lf0 = np.interp(np.arange(n), np.where(voiced)[0], lf0[voiced]).astype(np.float32)
    vuv = voiced.astype(np.float32)
    return pid, prv, nxt, pos, dur, lf0, vuv, hz


def load_model(dev):
    ck = torch.load(ROOT / "checkpoints" / "acoustic.pt", map_location=dev, weights_only=False)
    model = AcousticModel(len(ck["meta"]["phones"]), ck["n_out"]).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck


# 学習データ（PJS）の音域は MIDI 48-65 が中心。これを大きく超えるとモデルが外挿になり、
# 母音の区別が消えて「何を歌っているか分からない」音になる。
# そこで **モデルに渡す音高だけ** を上限で頭打ちにする（実際に鳴らす音高は楽譜どおり）。
COND_MAX_MIDI = 67.0        # 0 にすると頭打ちなし


def render(model, ck, pid, prv, nxt, pos, dur, lf0, vuv, hz, dev, cond_max_midi=None):
    cap = COND_MAX_MIDI if cond_max_midi is None else cond_max_midi
    if cap:
        lf0 = np.minimum(lf0, np.log(440.0 * 2 ** ((cap - 69) / 12))).astype(np.float32)
    feat = np.stack([pos, np.clip(dur, 0, 1.5), lf0, vuv], 1).astype(np.float32)
    feat = (feat - ck["feat_mean"]) / ck["feat_std"]
    T = lambda a, dt: torch.from_numpy(a[None]).to(dev, dt)
    with torch.no_grad():
        y = model(T(pid, torch.long), T(prv, torch.long), T(nxt, torch.long),
                  T(feat, torch.float))[0].cpu().numpy()
    y = y * ck["tgt_std"] + ck["tgt_mean"]
    mc, ac = y[:, :MCEP_ORDER].astype(np.float64), y[:, MCEP_ORDER:].astype(np.float64)
    warp = MelWarp(ck["meta"]["n_bins"], SR)
    sp, apx = mcep_to_sp(mc, warp), code_to_ap(ac, warp)
    f0 = np.ascontiguousarray(np.where(vuv > 0.5, hz, 0.0).astype(np.float64))
    wav = pw.synthesize(f0, sp, apx, SR, FRAME_MS)
    peak = np.abs(wav).max()
    return (wav / peak * 0.9 if peak > 0 else wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", type=str, help="実データ名（例 pjs100）を再合成")
    ap.add_argument("--ust", type=str, help="UST ファイル")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no_vibrato", action="store_true")
    ap.add_argument("--cond_max_midi", type=float, default=None,
                    help="モデルに渡す音高の上限（既定67）。0で無効")
    ap.add_argument("--transpose", type=float, default=None,
                    help="移調（半音）。省略時は学習音域に合うよう自動で下げる。0で移調なし")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(dev)

    if args.copy:
        d = np.load(FEAT / f"{args.copy}.npz")
        hz = np.where(d["vuv"] > 0.5, np.exp(d["lf0"]), 0.0)
        wav = render(model, ck, d["pid"], d["prev"], d["next"], d["pos"], d["dur"],
                     d["lf0"], d["vuv"], hz, dev, args.cond_max_midi)
        out = Path(args.out or OUT / f"{args.copy}_model.wav")
        sf.write(str(out), wav, SR)
        print(f"[synth] {args.copy}: 実データのラベルとF0から再合成 -> {out}")
        return

    if args.ust:
        notes = parse_ust(Path(args.ust))
        k = auto_transpose(notes) if args.transpose is None else args.transpose
        if k:
            notes = transpose(notes, k)
        print(f"[synth] {Path(args.ust).name}: {len(notes)}音符 / 移調 {k:+.0f} 半音")
        seq = notes_to_sequence(notes, vibrato=not args.no_vibrato)
        wav = render(model, ck, *seq[:7], seq[7], dev, args.cond_max_midi)
        out = Path(args.out or OUT / (Path(args.ust).stem + "_model.wav"))
        sf.write(str(out), wav, SR)
        print(f"[synth] -> {out}（{len(wav) / SR:.1f}秒）")
        return

    ap.error("--copy か --ust のどちらかを指定してください")


if __name__ == "__main__":
    main()
