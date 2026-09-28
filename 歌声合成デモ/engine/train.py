"""
音響モデルの学習：音素と音高から「声の音色」を予測する。

入力（フレームごと）:
    現在・直前・直後の音素ID、音素内の位置(0-1)、音素の長さ、log F0、有声フラグ
出力（フレームごと）:
    メルケプストラム60次元 + 非周期性4次元

F0 は入力として与える（楽譜から決まるため）。モデルが学ぶのは
「その音素を、その高さで歌うと、どんな響きになるか」だけ。
位相もスペクトルの細部も扱わないので、少ないデータでも破綻しにくい。

使い方:
    python engine/train.py                  # 400 epoch
    python engine/train.py --epochs 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FEAT = ROOT / "features"
CKPT = ROOT / "checkpoints"

SEG = 400            # 学習に使う切り出し長（フレーム）＝ 2秒
EMB = 32
HID = 256


class AcousticModel(nn.Module):
    """音素列＋F0 → スペクトル包絡。膨張畳み込みで前後 ±0.6秒 を見る。"""

    def __init__(self, n_phones: int, n_out: int):
        super().__init__()
        self.emb = nn.Embedding(n_phones, EMB)
        in_dim = EMB * 3 + 4                      # cur/prev/next + pos,dur,lf0,vuv
        self.inp = nn.Conv1d(in_dim, HID, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(HID, HID, 5, padding=2 * d, dilation=d),
                nn.BatchNorm1d(HID),
                nn.GELU(),
                nn.Conv1d(HID, HID, 1),
            ) for d in (1, 2, 4, 8, 16, 32)
        ])
        self.act = nn.GELU()
        self.out = nn.Conv1d(HID, n_out, 1)

    def forward(self, pid, prv, nxt, feat):
        # pid…(B,T) / feat…(B,T,4)
        e = torch.cat([self.emb(pid), self.emb(prv), self.emb(nxt), feat], dim=-1)
        h = self.act(self.inp(e.transpose(1, 2)))
        for blk in self.blocks:
            h = self.act(h + blk(h))
        return self.out(h).transpose(1, 2)


def load_all():
    files = sorted(FEAT.glob("pjs*.npz"))
    data = []
    for f in files:
        d = np.load(f)
        n = len(d["lf0"])
        data.append({
            "pid": d["pid"].astype(np.int64), "prev": d["prev"].astype(np.int64),
            "next": d["next"].astype(np.int64),
            "feat": np.stack([d["pos"], np.clip(d["dur"], 0, 1.5), d["lf0"], d["vuv"]], 1).astype(np.float32),
            "tgt": np.concatenate([d["mcep"], d["apc"]], 1).astype(np.float32),
            "n": n,
        })
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    CKPT.mkdir(exist_ok=True)
    meta = json.loads((FEAT / "meta.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_all()
    tr, va = data[:95], data[95:]
    print(f"[train] device={dev} 学習{len(tr)}曲 / 検証{len(va)}曲")

    # 正規化統計（学習データのみから）
    allf = np.concatenate([d["feat"] for d in tr]); allt = np.concatenate([d["tgt"] for d in tr])
    fm, fs = allf.mean(0), allf.std(0) + 1e-6
    tm, ts = allt.mean(0), allt.std(0) + 1e-6
    for d in data:
        d["feat"] = (d["feat"] - fm) / fs
        d["tgt"] = (d["tgt"] - tm) / ts

    n_out = allt.shape[1]
    model = AcousticModel(len(meta["phones"]), n_out).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"[train] 出力次元={n_out} パラメータ数={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    rng = np.random.default_rng(0)

    def batch(src, bs):
        pid, prv, nxt, ft, tg = [], [], [], [], []
        for _ in range(bs):
            d = src[rng.integers(len(src))]
            if d["n"] <= SEG:
                s, pad = 0, SEG - d["n"]
            else:
                s, pad = int(rng.integers(0, d["n"] - SEG)), 0
            sl = slice(s, s + SEG - pad)
            def P(a, v=0):
                return np.pad(a[sl], [(0, pad)] + [(0, 0)] * (a.ndim - 1), constant_values=v)
            pid.append(P(d["pid"])); prv.append(P(d["prev"])); nxt.append(P(d["next"]))
            ft.append(P(d["feat"])); tg.append(P(d["tgt"]))
        t = lambda a, dt: torch.from_numpy(np.stack(a)).to(dev, dt)
        return (t(pid, torch.long), t(prv, torch.long), t(nxt, torch.long),
                t(ft, torch.float), t(tg, torch.float))

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for _ in range(20):
            pid, prv, nxt, ft, tg = batch(tr, args.batch)
            loss = nn.functional.l1_loss(model(pid, prv, nxt, ft), tg)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss)
        sched.step()

        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                pid, prv, nxt, ft, tg = batch(va, 16)
                vl = float(nn.functional.l1_loss(model(pid, prv, nxt, ft), tg))
            print(f"[train] epoch {ep:4d}/{args.epochs}  train={tot / 20:.4f}  val={vl:.4f}")
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "n_out": n_out,
                            "feat_mean": fm, "feat_std": fs, "tgt_mean": tm, "tgt_std": ts,
                            "meta": meta, "epoch": ep, "val": vl},
                           CKPT / "acoustic.pt")
    print(f"[train] 完了 best val={best:.4f} -> {CKPT / 'acoustic.pt'}")


if __name__ == "__main__":
    main()
