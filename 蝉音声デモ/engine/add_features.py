"""生成済み wav から追加の特徴量を計算して web/data/map.json に書き戻す。
（デコードし直さないので数十秒で終わる。prerender.py の後に実行）"""
import json
from pathlib import Path
import numpy as np
import soundfile as sf

WEB = Path(__file__).resolve().parent.parent / "web"
meta = json.loads((WEB / "data" / "map.json").read_text(encoding="utf-8"))
sr = meta["sr"]

def feats(path):
    w, _ = sf.read(str(WEB / path), dtype="float32")
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), 1 / sr)
    p = spec ** 2
    total = p.sum() + 1e-12
    band_hi = float(p[(freqs >= 8000)].sum() / total)          # 8kHz 以上の割合＝ざらつき
    x = w - w.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= (ac[0] + 1e-12)
    lo, hi = int(sr * 0.002), int(sr * 0.05)                    # 20-500Hz のパルス周期
    k = int(np.argmax(ac[lo:hi])) + lo
    return {"band_hi": round(band_hi, 4),
            "pulse_hz": round(sr / k, 1),
            "pulse_strength": round(float(ac[k]), 4)}

for c in meta["cells"]:
    c.update(feats(c["wav"]))
for m in meta["morph"]:
    m.update(feats(m["wav"]))
(WEB / "data" / "map.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
v = [c["pulse_hz"] for c in meta["cells"]]
b = [c["band_hi"] for c in meta["cells"]]
print(f"pulse_hz {min(v)}-{max(v)} / band_hi {min(b)}-{max(b)}")
