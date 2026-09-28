"""
Phase 2: VAEモデル（PLAN B: ComplexSpecVAE-GAN V3.5 Sequence Latent）

クラス構成:
    Encoder        - (B, 2, 513, T_fixed) → (mu, log_var) 各 (B, latent_dim, T_seq)
    Decoder        - (B, latent_dim, T_seq) → (B, 2, 513, T_fixed)
    ComplexSpecVAE - Encoder + Decoder をまとめた VAE

V3.5変更点:
    Global Latent（1ベクトル）→ Sequence Latent（T_seq=9の時系列）
    AdaptiveAvgPool2d((8,8)) → AdaptiveAvgPool2d((8, T_seq)) で時間軸を保持
    fc層: Linear → Conv1d(kernel=1) でtime-step独立射影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    入力: (B, 2, 513, T_fixed)
    出力: mu (B, latent_dim, T_seq), log_var (B, latent_dim, T_seq)

    AdaptiveAvgPool2d((8, T_seq)) で周波数軸のみプールし時間軸を T_seq に保持。
    Conv1d(kernel=1) で各時間ステップを独立に latent_dim 次元へ射影。
    """

    def __init__(self, latent_dim: int, t_seq: int = 9, leaky_slope: float = 0.1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(  2,  32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d( 32,  64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d( 64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )
        self.pool       = nn.AdaptiveAvgPool2d((8, t_seq))          # 周波数→8, 時間→t_seq
        self.fc_mu      = nn.Conv1d(256 * 8, latent_dim, kernel_size=1)  # per-timestep 射影
        self.fc_log_var = nn.Conv1d(256 * 8, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h = self.convs(x)                           # (B, 256, H', W')
        h = self.pool(h)                             # (B, 256, 8, T_seq)
        B, C, freq, t = h.shape
        h = h.view(B, C * freq, t)                   # (B, 256*8, T_seq)
        return self.fc_mu(h), self.fc_log_var(h)     # (B, latent_dim, T_seq) each

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """学習時: z = mu + eps * exp(0.5 * log_var)  推論時: z = mu"""
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu


class Decoder(nn.Module):
    """
    入力: z (B, latent_dim, T_seq)
    出力: (B, 2, target_H, target_W)

    Conv1d(kernel=1) で各時間ステップを (256 × T_seq) チャネルへ射影し、
    (B, 256, T_seq, T_seq) の2D初期特徴マップに変換。
    6x ConvTranspose2d で (T_seq, T_seq)=(9,9) → (576,576) まで拡大し、
    crop/pad のみで目標サイズ (target_H, target_W) に揃える。
    bilinear 補間は使用しない。
    """

    def __init__(self, latent_dim: int, t_seq: int, target_H: int, target_W: int,
                 leaky_slope: float = 0.1):
        super().__init__()
        self.target_H = target_H  # 513 (= n_fft // 2 + 1)
        self.target_W = target_W  # T_fixed
        self.t_seq    = t_seq     # 9

        # 各時間ステップを (256 × t_seq) チャネルへ射影: (B, latent_dim, T_seq) → (B, 256*t_seq, T_seq)
        self.fc = nn.Conv1d(latent_dim, 256 * t_seq, kernel_size=1)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (9,9)→(18,18)
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),  # →(36,36)
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=1),  # →(72,72)
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d( 32,  16, kernel_size=4, stride=2, padding=1),  # →(144,144)
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d( 16,   8, kernel_size=4, stride=2, padding=1),  # →(288,288)
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d(  8,   2, kernel_size=4, stride=2, padding=1),  # →(576,576)
            # 活性化なし（Linear output）
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim, T_seq)
        B = z.size(0)
        h = self.fc(z)                                       # (B, 256*T_seq, T_seq)
        h = h.view(B, 256, self.t_seq, self.t_seq)          # (B, 256, T_seq, T_seq) = (B,256,9,9)
        h = self.deconvs(h)                                  # (B, 2, 576, 576)

        # crop または zero-pad のみで目標サイズに揃える（bilinear なし）
        H, W = h.shape[2], h.shape[3]
        if H > self.target_H:
            h = h[:, :, :self.target_H, :]
        elif H < self.target_H:
            h = F.pad(h, (0, 0, 0, self.target_H - H))

        if W > self.target_W:
            h = h[:, :, :, :self.target_W]
        elif W < self.target_W:
            h = F.pad(h, (0, self.target_W - W))

        return h  # (B, 2, target_H, target_W)

    def to_waveform(self, spec_2ch_raw: torch.Tensor, dataset,
                    original_length: int) -> torch.Tensor:
        """
        Decoder 出力（正規化済みスペクトル）を音声波形に変換する。

        Args:
            spec_2ch_raw  : (2, F, T) または (B, 2, F, T) — 正規化済み
            dataset       : ComplexSpectrogramDataset（逆正規化・STFTパラメータ用）
            original_length: Dataset.__getitem__ が返した元波形サンプル数
                             ※ 近似計算（T_fixed * hop_length 等）禁止

        Returns:
            wav: (original_length,) または (B, original_length)
        """
        # STFT演算は float32 必須（AMP使用時も変換）
        denorm = dataset.denormalize(spec_2ch_raw.float())
        device = spec_2ch_raw.device
        window = dataset.window.to(device)

        if denorm.dim() == 3:
            # single item: (2, F, T)
            complex_spec = torch.complex(denorm[0], denorm[1])  # (F, T)
        else:
            # batched: (B, 2, F, T)
            complex_spec = torch.complex(denorm[:, 0], denorm[:, 1])  # (B, F, T)

        wav = torch.istft(
            complex_spec,
            n_fft=dataset.n_fft,
            hop_length=dataset.hop_length,
            win_length=dataset.win_length,
            window=window,
            length=original_length,
        )
        return wav  # (original_length,) or (B, original_length)


class ComplexSpecVAE(nn.Module):
    """
    ComplexSpectrogramVAE: Encoder + Decoder。

    forward(spec_2ch) → (spec_recon, mu, log_var)
    encode(spec_2ch)  → (mu, log_var)
    decode(z)         → spec_recon
    """

    def __init__(self, config: dict, T_fixed: int):
        super().__init__()
        latent_dim  = config["model"]["latent_dim"]
        t_seq       = config["model"].get("t_seq", 9)
        leaky_slope = config["model"]["leaky_relu_slope"]
        F_bins      = config["stft"]["n_fft"] // 2 + 1  # 513

        self.encoder = Encoder(latent_dim, t_seq, leaky_slope)
        self.decoder = Decoder(latent_dim, t_seq, F_bins, T_fixed, leaky_slope)

    def encode(self, spec_2ch: torch.Tensor):
        return self.encoder(spec_2ch)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, spec_2ch: torch.Tensor):
        mu, log_var = self.encoder(spec_2ch)
        z = self.encoder.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# V3 Global Latent（model_3500_kl005.pt 互換）
# ─────────────────────────────────────────────────────────────────────────────

class EncoderV3(nn.Module):
    """V3 Global Latent Encoder: (B,2,513,T) → mu/log_var (B, latent_dim)"""

    def __init__(self, latent_dim: int, leaky_slope: float = 0.1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(  2,  32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d( 32,  64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d( 64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )
        self.fc_mu      = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_log_var = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.convs(x)                        # (B, 256, H', W')
        h = F.adaptive_avg_pool2d(h, (8, 8))     # (B, 256, 8, 8)
        h = h.view(h.size(0), -1)                # (B, 16384)
        return self.fc_mu(h), self.fc_log_var(h)  # (B, latent_dim) each


class DecoderV3(nn.Module):
    """V3 Global Latent Decoder: (B, latent_dim) → (B, 2, target_H, target_W)"""

    def __init__(self, latent_dim: int, target_H: int, target_W: int,
                 leaky_slope: float = 0.1):
        super().__init__()
        self.target_H = target_H
        self.target_W = target_W
        self.fc = nn.Linear(latent_dim, 256 * 9 * 9)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d( 64,  32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d( 32,  16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d( 16,   8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.ConvTranspose2d(  8,   2, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.fc(z).view(B, 256, 9, 9)  # (B, 256, 9, 9)
        h = self.deconvs(h)                  # (B, 2, 576, 576)
        H, W = h.shape[2], h.shape[3]
        if H > self.target_H:
            h = h[:, :, :self.target_H, :]
        elif H < self.target_H:
            h = F.pad(h, (0, 0, 0, self.target_H - H))
        if W > self.target_W:
            h = h[:, :, :, :self.target_W]
        elif W < self.target_W:
            h = F.pad(h, (0, self.target_W - W))
        return h

    def to_waveform(self, spec_2ch_raw: torch.Tensor, dataset,
                    original_length: int) -> torch.Tensor:
        denorm = dataset.denormalize(spec_2ch_raw.float())
        device = spec_2ch_raw.device
        window = dataset.window.to(device)
        if denorm.dim() == 3:
            complex_spec = torch.complex(denorm[0], denorm[1])
        else:
            complex_spec = torch.complex(denorm[:, 0], denorm[:, 1])
        return torch.istft(
            complex_spec,
            n_fft=dataset.n_fft,
            hop_length=dataset.hop_length,
            win_length=dataset.win_length,
            window=window,
            length=original_length,
        )


class ComplexSpecVAE_V3(nn.Module):
    """ComplexSpecVAE V3 互換（model_3500_kl005.pt 用 Global Latent アーキテクチャ）"""

    def __init__(self, config: dict, T_fixed: int):
        super().__init__()
        latent_dim  = config["model"]["latent_dim"]
        leaky_slope = config["model"]["leaky_relu_slope"]
        F_bins      = config["stft"]["n_fft"] // 2 + 1

        self.encoder = EncoderV3(latent_dim, leaky_slope)
        self.decoder = DecoderV3(latent_dim, F_bins, T_fixed, leaky_slope)

    def encode(self, spec_2ch):
        return self.encoder(spec_2ch)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, spec_2ch):
        mu, log_var = self.encoder(spec_2ch)
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std if self.training else mu
        return self.decoder(z), mu, log_var
