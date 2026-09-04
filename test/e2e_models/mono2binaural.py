#!/usr/bin/env python3
"""
mono2binaural.py — faithful modern-PyTorch port of 2.5D-Visual-Sound (Gao &
Grauman, CVPR 2019) inference, to run the pretrained FAIR-Play split1 checkpoint
(facebookresearch/2.5D-Visual-Sound) on torch 2.5. Architecture and constants
reconstructed from the released checkpoints + repo (networks.py / demo.py):

  VisualNet : ResNet-18 minus last 2 layers -> (512,7,14) feature map
  AudioNet  : 5-down / 5-up U-Net on the mix STFT (real/imag, 2ch), visual
              feature 1x1-> 8ch, flattened to 784 and tiled into the bottleneck
              (1296 = 512 audio + 784 visual); decoder uses encoder skips; final
              layer sigmoid*2-1 = complex ratio MASK in [-1,1].
  Reconstruct: diff_spec = mask (complex) * mix_spec (complex);
               diff = iSTFT; left=(mix+diff)/2, right=(mix-diff)/2; overlap-add.

Audio: 16 kHz, window 0.63 s, hop 0.05 s, RMS-normalized to 0.1; STFT n_fft=512
(drop nyquist -> 256 bins), hop=160, win=400, Hann. Frame: video frame at window
midpoint. Verified by reconstructing GT binaural (STFT-distance sanity check).
"""
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.models import resnet18

SR = 16000
AUDIO_LEN = 0.63
HOP_S = 0.05
NFFT = 512
HOP = 160
WIN = 400
EPS = 1e-4
IMG_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], np.float32)


class VisualNet(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        self.feature_extraction = nn.Sequential(*list(net.children())[:-2])

    def forward(self, x):
        return self.feature_extraction(x)


def unet_conv(inc, outc):
    return nn.Sequential(nn.Conv2d(inc, outc, 4, 2, 1),
                         nn.BatchNorm2d(outc), nn.LeakyReLU(0.2, True))


def unet_upconv(inc, outc, outermost=False):
    if outermost:
        return nn.Sequential(nn.ConvTranspose2d(inc, outc, 4, 2, 1))
    return nn.Sequential(nn.ConvTranspose2d(inc, outc, 4, 2, 1),
                         nn.BatchNorm2d(outc), nn.ReLU(True))


class AudioNet(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.audionet_convlayer1 = unet_conv(2, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.conv1x1 = nn.Sequential(nn.Conv2d(512, 8, 1),
                                     nn.BatchNorm2d(8), nn.ReLU(True))
        self.audionet_upconvlayer1 = unet_upconv(1296, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, 2, outermost=True)

    def forward(self, x, visual_feat):
        c1 = self.audionet_convlayer1(x)
        c2 = self.audionet_convlayer2(c1)
        c3 = self.audionet_convlayer3(c2)
        c4 = self.audionet_convlayer4(c3)
        c5 = self.audionet_convlayer5(c4)
        v = self.conv1x1(visual_feat)
        v = v.view(v.shape[0], -1, 1, 1).repeat(1, 1, c5.shape[-2], c5.shape[-1])
        u1 = self.audionet_upconvlayer1(torch.cat((c5, v), dim=1))
        u2 = self.audionet_upconvlayer2(torch.cat((u1, c4), dim=1))
        u3 = self.audionet_upconvlayer3(torch.cat((u2, c3), dim=1))
        u4 = self.audionet_upconvlayer4(torch.cat((u3, c2), dim=1))
        mask = self.audionet_upconvlayer5(torch.cat((u4, c1), dim=1))
        return torch.sigmoid(mask) * 2 - 1


_WIN_T = torch.hann_window(WIN)        # periodic Hann, matches librosa default


def generate_spectrogram(audio):
    t = torch.from_numpy(np.ascontiguousarray(audio)).float()
    s = torch.stft(t, n_fft=NFFT, hop_length=HOP, win_length=WIN, window=_WIN_T,
                   center=True, pad_mode="reflect", return_complex=True)
    s = s[:-1, :]                      # drop nyquist -> 256 bins
    return np.stack([s.real.numpy(), s.imag.numpy()], axis=0).astype(np.float32)


def istft_diff(diff_stft_256, length):
    """iSTFT of a (256,T) complex difference spec (nyquist row re-appended)."""
    full = np.vstack([diff_stft_256,
                      np.zeros((1, diff_stft_256.shape[1]), np.complex64)])
    dt = torch.from_numpy(full.astype(np.complex64))
    return torch.istft(dt, n_fft=NFFT, hop_length=HOP, win_length=WIN,
                       window=_WIN_T, center=True, length=length).numpy()


def audio_normalize(samples, desired_rms=0.1):
    rms = np.maximum(EPS, np.sqrt(np.mean(samples ** 2)))
    return rms / desired_rms, samples * (desired_rms / rms)


class Mono2Binaural:
    def __init__(self, ckpt_dir, device="cuda:0"):
        self.device = device
        self.visual = VisualNet().to(device).eval()
        self.audio = AudioNet().to(device).eval()
        vsd = torch.load(f"{ckpt_dir}/visual_model.pth", map_location="cpu", weights_only=True)
        asd = torch.load(f"{ckpt_dir}/audio_model.pth", map_location="cpu", weights_only=True)
        self.visual.load_state_dict(vsd)
        self.audio.load_state_dict(asd)

    def _decode_all(self, video_path):
        """Sequentially decode all frames once (fast, no per-window seeking).
        Returns (frames_tensor[F,3,224,448], fps)."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, (448, 224))      # (W,H)=(448,224) -> (H,W)=(224,448)
            fr = (fr.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD
            frames.append(fr.transpose(2, 0, 1))
        cap.release()
        if not frames:
            return None, fps
        return np.stack(frames), fps

    @torch.no_grad()
    def predict(self, video_path, mono_audio, total_sec):
        """mono_audio: 1-D np array @16k. Returns predicted binaural (2, N)."""
        frames, fps = self._decode_all(video_path)
        if frames is None:
            return np.zeros((2, len(mono_audio)), np.float32)
        nF = len(frames)
        N = len(mono_audio)
        out = np.zeros((2, N), np.float32)
        cnt = np.zeros((2, N), np.float32)
        wsamp = int(AUDIO_LEN * SR)
        hsamp = int(HOP_S * SR)
        start = 0
        while start + wsamp < N:
            seg = mono_audio[start:start + wsamp]
            norm, segn = audio_normalize(seg)
            mix_spec = generate_spectrogram(segn)              # (2,256,T)
            t_mid = (start + wsamp / 2.0) / SR
            fidx = min(nF - 1, int(round(t_mid * fps)))
            frame = torch.from_numpy(frames[fidx]).unsqueeze(0).to(self.device)
            vfeat = self.visual(frame)
            mix_t = torch.from_numpy(mix_spec).unsqueeze(0).to(self.device)
            mask = self.audio(mix_t, vfeat)[0].cpu().numpy()    # (2,256,T) in [-1,1]
            mr, mi = mask[0], mask[1]
            xr, xi = mix_spec[0], mix_spec[1]
            # complex mask multiply: diff = mask * mix
            dr = xr * mr - xi * mi
            di = xr * mi + xi * mr
            diff_stft = (dr + 1j * di).astype(np.complex64)
            diff = istft_diff(diff_stft, wsamp)
            mix = segn
            left = (mix + diff) / 2.0
            right = (mix - diff) / 2.0
            recon = np.stack([left, right]) * norm
            out[:, start:start + wsamp] += recon
            cnt[:, start:start + wsamp] += 1
            start += hsamp
        cnt[cnt == 0] = 1
        return out / cnt
