"""
WORLD ボコーダ用の特徴量変換（pysptk を使わない自前実装）

スペクトル包絡 (513次元) をそのまま学習するのは重いので、
    周波数軸をメル尺度に伸縮 → 対数 → DCT-II → 低次40係数だけ残す
という圧縮をかける。逆変換で WORLD に戻せる。

pysptk.sp2mc / mc2sp の代替。メル一般化ケプストラムではなく
「メル軸上の対数スペクトルのケプストラム」なので厳密には別物だが、
歌声のスペクトル包絡は滑らかなので実用上ほぼ同じ精度が出る。
"""

import numpy as np
from scipy.fftpack import dct, idct


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


class MelWarp:
    """線形周波数ビン ↔ メル等間隔ビンの相互変換（線形補間）"""

    def __init__(self, n_bins: int, sr: int, n_mel: int = 256):
        self.n_bins, self.sr, self.n_mel = n_bins, sr, n_mel
        self.f_lin = np.linspace(0, sr / 2, n_bins)
        m_max = hz_to_mel(sr / 2)
        self.f_mel = mel_to_hz(np.linspace(0, m_max, n_mel))

    def to_mel(self, spec):                 # (T, n_bins) -> (T, n_mel)
        return np.stack([np.interp(self.f_mel, self.f_lin, s) for s in spec])

    def to_lin(self, spec_mel):             # (T, n_mel) -> (T, n_bins)
        return np.stack([np.interp(self.f_lin, self.f_mel, s) for s in spec_mel])


def sp_to_mcep(sp, warp: MelWarp, order: int = 40):
    """WORLD のスペクトル包絡 -> メルケプストラム (T, order)"""
    log_mel = np.log(np.maximum(warp.to_mel(sp), 1e-10))
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :order]


def mcep_to_sp(mc, warp: MelWarp):
    """メルケプストラム -> WORLD のスペクトル包絡 (T, n_bins)"""
    pad = np.zeros((mc.shape[0], warp.n_mel), dtype=np.float64)
    pad[:, : mc.shape[1]] = mc
    log_mel = idct(pad, type=2, axis=1, norm="ortho")
    return np.ascontiguousarray(np.exp(warp.to_lin(log_mel)))


def ap_to_code(ap, warp: MelWarp, order: int = 4):
    """非周期性指標 -> 低次係数（同じ圧縮を使う）"""
    log_mel = np.log(np.maximum(warp.to_mel(ap), 1e-10))
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, :order]


def code_to_ap(code, warp: MelWarp):
    pad = np.zeros((code.shape[0], warp.n_mel), dtype=np.float64)
    pad[:, : code.shape[1]] = code
    log_mel = idct(pad, type=2, axis=1, norm="ortho")
    ap = np.exp(warp.to_lin(log_mel))
    return np.ascontiguousarray(np.clip(ap, 1e-6, 1.0 - 1e-6))
