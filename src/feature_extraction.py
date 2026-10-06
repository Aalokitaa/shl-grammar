"""
SHL Spoken Grammar Scoring Engine - Feature Extraction Module
Extracts acoustic, prosodic, spectral, and temporal features from audio files.
"""

import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

def extract_audio_features(fpath, fname, split, label=None):
    """
    Extracts RMS energy, silence ratios, pause metrics, spectral centroid/rolloff/zcr,
    sub-band frequencies, and 20 MFCCs + Deltas + Delta-Deltas.
    """
    try:
        y, sr = sf.read(fpath)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        dur = float(len(y) / sr)
        
        # RMS Energy & Silence
        rms = librosa.feature.rms(y=y, hop_length=1024)[0]
        rms_max = float(np.max(rms))
        silence_thresh = 0.01 * rms_max if rms_max > 0 else 0.001
        is_silent = rms < silence_thresh
        frame_dur = 1024.0 / sr
        silence_dur = float(np.sum(is_silent) * frame_dur)
        speech_dur = max(0.001, dur - silence_dur)
        silence_ratio = float(silence_dur / max(0.001, dur))
        
        # Pauses > 0.3s
        min_pause_frames = int(0.3 / frame_dur)
        pause_counts = 0
        pause_lens = []
        curr_silent_run = 0
        for s in is_silent:
            if s:
                curr_silent_run += 1
            else:
                if curr_silent_run >= min_pause_frames:
                    pause_counts += 1
                    pause_lens.append(curr_silent_run * frame_dur)
                curr_silent_run = 0
        if curr_silent_run >= min_pause_frames:
            pause_counts += 1
            pause_lens.append(curr_silent_run * frame_dur)
            
        pause_rate = float((pause_counts / max(0.001, dur)) * 60.0)
        mean_pause_len = float(np.mean(pause_lens)) if pause_lens else 0.0
        max_pause_len = float(np.max(pause_lens)) if pause_lens else 0.0
        
        # Spectral Features
        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=y, hop_length=1024)[0]))
        sc_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=1024)[0]))
        sb_mean = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=1024)[0]))
        sroll_mean = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=1024)[0]))
        
        # MFCCs (20 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=1024)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        feat = {
            "filename": fname,
            "split": split,
            "label": label if label is not None else -1.0,
            "duration": dur,
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_max": rms_max,
            "silence_dur": silence_dur,
            "speech_dur": speech_dur,
            "silence_ratio": silence_ratio,
            "pause_counts": pause_counts,
            "pause_rate": pause_rate,
            "mean_pause_len": mean_pause_len,
            "max_pause_len": max_pause_len,
            "zcr_mean": zcr_mean,
            "sc_mean": sc_mean,
            "sb_mean": sb_mean,
            "sroll_mean": sroll_mean
        }
        
        for i in range(20):
            feat[f"mfcc_mean_{i}"] = float(np.mean(mfcc[i]))
            feat[f"mfcc_std_{i}"] = float(np.std(mfcc[i]))
            feat[f"mfcc_d_mean_{i}"] = float(np.mean(mfcc_delta[i]))
            feat[f"mfcc_d2_mean_{i}"] = float(np.mean(mfcc_delta2[i]))
            
        return feat
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        return None
