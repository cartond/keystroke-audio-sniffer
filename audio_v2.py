import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------------
# Parameters
# -------------------------
AUDIO_FILE = "./typing_2_abc.wav"   # path to your recording
SR_TARGET = 44100          # target sample rate
FRAME_LENGTH = 1024        # for spectral flux
HOP_LENGTH = 512
AMP_PEAK_HEIGHT = 0.02     # adjust for your audio
FLUX_PEAK_HEIGHT = 0.01

# -------------------------
# Step 1: Load audio
# -------------------------
waveform, sr = librosa.load(AUDIO_FILE, sr=SR_TARGET, mono=True)
times = np.arange(len(waveform)) / sr

# -------------------------
# Step 2a: Amplitude envelope
# -------------------------
amplitude_envelope = np.abs(librosa.stft(waveform, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
amp_env = amplitude_envelope.mean(axis=0)  # mean across frequencies
amp_env_times = np.arange(len(amp_env)) * HOP_LENGTH / sr

# -------------------------
# Step 2b: Spectral flux
# -------------------------
# Compute magnitude spectrogram
S = np.abs(librosa.stft(waveform, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
# Compute flux as difference along time axis
flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)) / S.shape[0]

# -------------------------
# Step 3: Detect peaks
# -------------------------
amp_peaks, _ = find_peaks(amp_env, height=AMP_PEAK_HEIGHT, distance=10)
flux_peaks, _ = find_peaks(flux, height=FLUX_PEAK_HEIGHT, distance=10)

# Combine peaks (optional: intersection or union)
candidate_peaks = np.unique(np.concatenate([amp_peaks, flux_peaks]))
candidate_times = candidate_peaks * HOP_LENGTH / sr

# -------------------------
# Step 4: Plot waveform + markers
# -------------------------
plt.figure(figsize=(14, 6), facecolor='lightgray')

# Waveform
plt.plot(times, waveform, alpha=0.7, label="Waveform")
plt.vlines(candidate_times, ymin=-1, ymax=1, color='r', alpha=0.5, label="Detected Peaks")

# Optional: overlay envelope
plt.plot(amp_env_times, amp_env/amp_env.max(), color='orange', label="Amplitude Envelope (norm)")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Keystroke Detection - Candidate Peaks")
plt.legend()
plt.tight_layout()
plt.show()
