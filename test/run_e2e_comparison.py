#!/usr/bin/env python3
"""
run_e2e_comparison.py — #1 end-to-end spatial-audio comparison.

Runs the pretrained 2.5D-Visual-Sound (mono->binaural) model on FAIR-Play test
clips and measures how faithfully its OUTPUT binaural reproduces the recorded
GT binaural's azimuth trajectory, using the SAME ITD inversion harness on both
(so the ~inversion floor is common-mode). A mono baseline (azimuth == 0, i.e.
no spatialization) bounds the floor: if the model's AzMAE-vs-GT is not below the
mono baseline, it is not adding usable azimuth.

Reported per clip and pooled:
  - azmae_e2e   = mean |az(e2e binaural) - az(GT binaural)|
  - azmae_mono  = mean |0 - az(GT binaural)|        (no-spatialization floor)
  - range compression: e2e azimuth range / GT azimuth range
Contextualized against Vid2Spatial's 1.36° visual-tracking AzMAE (different
dataset/protocol; stated honestly).

Usage: python test/run_e2e_comparison.py [--n 50] [--device cuda:0]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, "test")
sys.path.insert(0, "test/e2e_models")
from run_binaural_az_inversion import gcc_phat, itd_to_az, R_HEAD, C_SND
from mono2binaural import Mono2Binaural, SR

FP = Path("/home/seung/data/fairplay")
TEST_JSONL = Path("data/fairplay/split1_test.jsonl")
WIN_S, HOP_S = 0.40, 0.20


def cue_series(L, R, fs):
    """Return per-window (ITD-azimuth deg, ILD dB). ILD = 10log10(L^2/R^2)."""
    wn = int(WIN_S * fs); hn = int(HOP_S * fs)
    max_tau = (R_HEAD / C_SND) * (math.pi / 2 + 1) * 1.2
    itd_az, ild = [], []
    for s in range(0, len(L) - wn, hn):
        a, b = L[s:s + wn], R[s:s + wn]
        if np.sqrt(np.mean(a ** 2)) < 1e-4 and np.sqrt(np.mean(b ** 2)) < 1e-4:
            itd_az.append(np.nan); ild.append(np.nan); continue
        itd_az.append(itd_to_az(gcc_phat(a, b, fs, max_tau)))
        ild.append(10 * np.log10((np.mean(a ** 2) + 1e-9) / (np.mean(b ** 2) + 1e-9)))
    return np.array(itd_az), np.array(ild)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="test/full_eval/E2E_COMPARISON.json")
    args = ap.parse_args()

    ids = [json.loads(l)["id"] for l in open(TEST_JSONL)]
    if args.n > 0:
        ids = ids[:args.n]
    model = Mono2Binaural("test/e2e_models/checkpoints", device=args.device)
    print(f"[e2e] model loaded; {len(ids)} FAIR-Play test clips")

    # ---- pass 1: run model + collect per-window cues per clip ----
    clips = []
    for i, cid in enumerate(ids):
        wavp = FP / "binaural_audios" / f"{cid}.wav"
        vidp = FP / "videos" / f"{cid}.mp4"
        if not wavp.exists() or not vidp.exists():
            continue
        gt, sr = sf.read(str(wavp))
        if gt.ndim != 2:
            continue
        gL = resample_poly(gt[:, 0], SR, sr); gR = resample_poly(gt[:, 1], SR, sr)
        pred = model.predict(str(vidp), gL + gR, len(gL) / SR)
        n = min(pred.shape[1], len(gL))
        g_itd, g_ild = cue_series(gL[:n], gR[:n], SR)
        e_itd, e_ild = cue_series(pred[0, :n], pred[1, :n], SR)
        m = ~(np.isnan(g_itd) | np.isnan(e_itd) | np.isnan(g_ild) | np.isnan(e_ild))
        if m.sum() < 5:
            continue
        clips.append(dict(clip=cid, g_itd=g_itd[m], g_ild=g_ild[m],
                          e_itd=e_itd[m], e_ild=e_ild[m]))
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(ids)}] processed {cid}")

    # ---- calibrate ILD->azimuth on GT (gives the model its ILD cue, fairly) ----
    GI = np.concatenate([c["g_ild"] for c in clips])
    GA = np.concatenate([c["g_itd"] for c in clips])      # GT ITD-azimuth = truth
    k = float(np.polyfit(GI, GA, 1)[0])                    # deg per dB (slope)
    b = float(np.polyfit(GI, GA, 1)[1])
    ild_az = lambda ild: k * ild + b

    # ---- pass 2: per-clip AzMAE under ITD ruler and ILD-calibrated ruler ----
    rows = []
    for c in clips:
        truth = c["g_itd"]                                 # GT azimuth (reliable ITD)
        e_itd_az = c["e_itd"]
        e_ild_az = ild_az(c["e_ild"])
        rows.append(dict(
            clip=c["clip"], n=int(len(truth)),
            azmae_e2e_itd=float(np.mean(np.abs(e_itd_az - truth))),
            azmae_e2e_ild=float(np.mean(np.abs(e_ild_az - truth))),
            azmae_mono=float(np.mean(np.abs(truth))),      # predict center (0)
            gt_range=float(truth.max() - truth.min()),
            e2e_itd_range=float(e_itd_az.max() - e_itd_az.min()),
            e2e_ild_range=float(e_ild_az.max() - e_ild_az.min()),
            ild_corr=float(np.corrcoef(c["e_ild"], truth)[0, 1]) if np.std(c["e_ild"]) > 1e-6 else 0.0))

    def col(k_): return np.array([r[k_] for r in rows])
    summ = dict(
        n_clips=len(rows), ild_calib_deg_per_dB=round(k, 2), ild_calib_intercept=round(b, 2),
        azmae_e2e_itd_mean=round(float(col("azmae_e2e_itd").mean()), 3),
        azmae_e2e_ild_mean=round(float(col("azmae_e2e_ild").mean()), 3),
        azmae_e2e_ild_median=round(float(np.median(col("azmae_e2e_ild"))), 3),
        azmae_mono_mean=round(float(col("azmae_mono").mean()), 3),
        azmae_mono_median=round(float(np.median(col("azmae_mono"))), 3),
        ild_beats_mono_frac=round(float((col("azmae_e2e_ild") < col("azmae_mono")).mean()), 3),
        gt_range_mean=round(float(col("gt_range").mean()), 2),
        e2e_itd_range_mean=round(float(col("e2e_itd_range").mean()), 2),
        e2e_ild_range_mean=round(float(col("e2e_ild_range").mean()), 2),
        mean_ild_corr=round(float(np.nanmean(col("ild_corr"))), 3),
        vid2spatial_visual_azmae=1.36,
    )
    Path(args.out).write_text(json.dumps(dict(summary=summ, per_clip=rows), indent=2))
    print("\n" + "=" * 74)
    print("#1 END-TO-END (2.5D Visual Sound) — azimuth fidelity on FAIR-Play test")
    print("=" * 74)
    print(f"  clips: {summ['n_clips']}   ILD->az calib: {k:.2f} deg/dB")
    print(f"  e2e AzMAE vs recorded GT azimuth:")
    print(f"    - ITD ruler:           {summ['azmae_e2e_itd_mean']:.2f} deg")
    print(f"    - ILD ruler (fair):    {summ['azmae_e2e_ild_mean']:.2f} deg (median {summ['azmae_e2e_ild_median']:.2f})")
    print(f"  mono no-spatialization floor: {summ['azmae_mono_mean']:.2f} deg (median {summ['azmae_mono_median']:.2f})")
    print(f"  ILD ruler beats mono floor on {summ['ild_beats_mono_frac']*100:.0f}% of clips")
    print(f"  azimuth range: GT={summ['gt_range_mean']:.1f} | e2e-ITD={summ['e2e_itd_range_mean']:.1f} "
          f"| e2e-ILD={summ['e2e_ild_range_mean']:.1f} deg")
    print(f"  mean ILD-vs-truth correlation: {summ['mean_ild_corr']:+.2f}")
    print(f"  [context] Vid2Spatial visual-tracking AzMAE = 1.36 deg (LaSOT)")
    print(f"\n[json] -> {args.out}")


if __name__ == "__main__":
    main()
