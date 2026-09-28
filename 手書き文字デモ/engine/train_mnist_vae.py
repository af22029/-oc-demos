"""
手書き数字（MNIST）VAE を学習し、ブラウザ用に重みを書き出す。

設計:
    潜在は 8 次元（精度のため）。地図表示だけ PCA で 2 次元に落とす。
    地図上の点 (a, b) は z = mean + a*pc1 + b*pc2 という 8 次元空間の「平面の切り口」なので、
    クリックした場所と生成される文字は厳密に一致する（見せかけの2次元化ではない）。

出力: web/data/mnist_vae.js
      エンコーダ／デコーダの重み + PCA 基底 + テストデータの潜在ベクトル（8次元, 判定用）
      ブラウザ側が JS で forward 計算するので、配布時に Python は不要。

使い方:
    python engine/train_mnist_vae.py                  # 潜在8次元・40 epoch
    python engine/train_mnist_vae.py --zdim 2         # 2次元（地図がそのまま潜在になるが精度は落ちる）
"""

import argparse
import base64
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE  = Path(__file__).resolve().parent
CACHE = HERE / "mnist_cache"
OUT   = HERE.parent / "web" / "data" / "mnist_vae.js"

BASE  = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x":  "t10k-images-idx3-ubyte.gz",  "test_y":  "t10k-labels-idx1-ubyte.gz"}

H1, H2 = 512, 256


def fetch(name: str) -> np.ndarray:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / FILES[name]
    if not path.exists():
        print(f"[mnist] downloading {FILES[name]} ...")
        urllib.request.urlretrieve(BASE + FILES[name], path)
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if "x" in name:
        return np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    return np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)


class VAE(nn.Module):
    def __init__(self, zdim: int):
        super().__init__()
        self.e1 = nn.Linear(784, H1); self.e2 = nn.Linear(H1, H2)
        self.mu = nn.Linear(H2, zdim); self.lv = nn.Linear(H2, zdim)
        self.d1 = nn.Linear(zdim, H2); self.d2 = nn.Linear(H2, H1)
        self.d3 = nn.Linear(H1, 784)

    def encode(self, x):
        h = F.relu(self.e2(F.relu(self.e1(x))))
        return self.mu(h), self.lv(h)

    def decode(self, z):
        return torch.sigmoid(self.d3(F.relu(self.d2(F.relu(self.d1(z))))))

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.decode(z), mu, lv


def b64(a) -> str:
    return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zdim", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n_scatter", type=int, default=4000, help="地図に描く実データ点の数")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mnist] device={dev} zdim={args.zdim}")

    xtr = torch.from_numpy(fetch("train_x")).to(dev)
    xte_np = fetch("test_x")
    xte = torch.from_numpy(xte_np).to(dev)
    yte = fetch("test_y")

    model = VAE(args.zdim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = xtr.shape[0]
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            xb = xtr[perm[i:i + args.batch]]
            recon, mu, lv = model(xb)
            bce = F.binary_cross_entropy(recon, xb, reduction="sum") / xb.shape[0]
            kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum()) / xb.shape[0]
            loss = bce + args.beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * xb.shape[0]
        if ep % 5 == 0 or ep == 1:
            print(f"[mnist] epoch {ep:3d}/{args.epochs}  loss={tot / n:.2f}")

    # ── 潜在ベクトル & PCA（地図用の2次元平面）──────────────────
    model.eval()
    with torch.no_grad():
        mu_te = model.encode(xte)[0].cpu().numpy()
        mu_tr = model.encode(xtr[:20000])[0].cpu().numpy()

    mean = mu_tr.mean(0)
    U, S, Vt = np.linalg.svd(mu_tr - mean, full_matrices=False)
    comps = Vt[:2]                                   # (2, zdim)
    proj_tr = (mu_tr - mean) @ comps.T
    lo = np.percentile(proj_tr, 1.5, axis=0)
    hi = np.percentile(proj_tr, 98.5, axis=0)
    var = (S ** 2 / (S ** 2).sum())[:2]
    print(f"[mnist] PCA寄与率 PC1={var[0]:.3f} PC2={var[1]:.3f}")

    # ── 判定精度の実測（デモと同じ k-NN 判定）────────────────
    rng = np.random.default_rng(0)
    idx = rng.choice(len(mu_te), size=min(args.n_scatter, len(mu_te)), replace=False)
    ref_z, ref_y = mu_te[idx], yte[idx]
    rest = np.setdiff1d(np.arange(len(mu_te)), idx)[:3000]
    d = ((mu_te[rest][:, None, :] - ref_z[None]) ** 2).sum(-1)
    knn = ref_y[np.argsort(d, axis=1)[:, :5]]
    pred = np.array([np.bincount(r, minlength=10).argmax() for r in knn])
    acc = float((pred == yte[rest]).mean())
    print(f"[mnist] 5-NN判定の正答率（{args.zdim}次元潜在）: {acc:.1%}")

    W = {}
    for name, layer in [("e1", model.e1), ("e2", model.e2), ("mu", model.mu),
                        ("d1", model.d1), ("d2", model.d2), ("d3", model.d3)]:
        W[name] = {"w": b64(layer.weight.detach().cpu().numpy()),
                   "b": b64(layer.bias.detach().cpu().numpy()),
                   "shape": list(layer.weight.shape)}

    meta = {
        "arch": {"h1": H1, "h2": H2, "zdim": args.zdim},
        "weights": W,
        "pca": {"mean": [round(float(v), 4) for v in mean],
                "comps": [[round(float(v), 4) for v in c] for c in comps],
                "var": [round(float(v), 4) for v in var]},
        "range": {"lo": [round(float(v), 3) for v in lo], "hi": [round(float(v), 3) for v in hi]},
        "scatter": {"z": b64(ref_z), "y": [int(v) for v in ref_y], "n": int(len(ref_z))},
        "accuracy": round(acc, 4),
        "epochs": args.epochs, "beta": args.beta,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.MNIST=" + json.dumps(meta) + ";", encoding="utf-8")
    print(f"[mnist] 書き出し完了: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
