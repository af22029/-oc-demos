# 蝉ミキサー（高校生向けデモ）

ComplexSpecVAE-GAN（提案手法）の学習済みモデルで、**潜在空間の地図**を体験させるデモ。

## 使い方（当日）

`デモを開く.bat` をダブルクリック → ブラウザが開く。Python も GPU もネットも不要。

- **① 声の地図**：11×11 の格子をクリック（ドラッグでもOK）すると、その場所の蝉の声が鳴る。白い点＝本物の録音の位置。
- **② モーフィング**：蝉A→蝉B へスライダーで連続変化。
- **③ しくみ**：VAE / GAN / ISTFT の説明パネル。

## 研究フォルダとの関係（重要）

このフォルダは **`D:\GAN V3` から完全に独立**しています。

- `engine/snapshot/` に `model.py` `model_4000.pt` `z_vectors.npy` `eval_rms_stats.npy` `config.json` を **コピー**して保持
- 実行時に `D:\GAN V3` を参照も書き込みもしない
- 研究側を改修してもデモは壊れず、デモを触っても研究側に影響しない

## 音を作り直したいとき

```bash
python engine/prerender.py     # 格子121音＋モーフ21音＋スペクトログラム画像を生成（GPUで数分）
python engine/add_features.py  # 生成音からリズム等の特徴量を計算し map.json に追記
python -c "from pathlib import Path; t=Path('web/data/map.json').read_text(encoding='utf-8'); Path('web/data/map.js').write_text('window.MAP='+t+';',encoding='utf-8')"
```

`prerender.py` の先頭定数で調整可能：`GRID`（格子の細かさ）、`MORPH_STEPS`、`EQ_GAIN=0.76`、`PCTL`（地図が覆う範囲）。

最後の map.js 変換は `file://` で開けるようにするため（fetch が使えないので JSON を JS に埋め込む）。

## 生成条件

model_4000.pt（V3 Global Latent, kl_beta=0.05）＋ EQ 0.76（4–8kHz）＋ ISTFT、
3秒 / 48kHz、スペクトル RMS を実データ平均に正規化（＝格子間で音量差が出ない）。
潜在 8 次元 → PCA 2 次元（寄与率 PC1 0.53 / PC2 0.12）で地図化し、逆変換して decode。

## 関連

手書き数字版のデモ（同じ VAE のしくみを目で見える文字で体験する）は `D:\手書き文字デモ_高校生向け\` に分離してあります。
