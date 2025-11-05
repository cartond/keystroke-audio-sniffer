import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# would be cl commands or func. args, but easier to test this way
AUDIO_FILE = "./training_data/typing_2_abc.wav"   # path to your recording
TYPING_FILE = "./training_data/typing_2_abc.txt" # what we typed in that recording

# some knobs for tuning
SR_TARGET = 44100          # target sample rate
FRAME_LENGTH = 1024        # for spectral flux
HOP_LENGTH = 512
AMP_PEAK_HEIGHT = 0.15     # adjust for the audio
FLUX_PEAK_HEIGHT = 0.01
MERGE_MS = 80              # peaks within MERGE_MS will be merged to aovid double samples (e.g., press+release)

# step 1a: Load audio
waveform, sr = librosa.load(AUDIO_FILE, sr=SR_TARGET, mono=True)
times = np.arange(len(waveform)) / sr

# step 1b: Load typed file text
typed_content = ""
with open(TYPING_FILE, "r") as f:
    typed_content = f.read().strip()
print(f"Typed text: {typed_content}")

# step 2a: Amplitude envelope
amplitude_envelope = np.abs(librosa.stft(waveform, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
amp_env = amplitude_envelope.mean(axis=0)  # mean across frequencies
amp_env_times = np.arange(len(amp_env)) * HOP_LENGTH / sr

# step 2b: Spectral flux
S = np.abs(librosa.stft(waveform, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)) / S.shape[0]

# step 3: Detect peaks
amp_peaks, _ = find_peaks(amp_env, height=AMP_PEAK_HEIGHT, distance=10)
flux_peaks, _ = find_peaks(flux, height=FLUX_PEAK_HEIGHT, distance=10)

# Combine peaks (union)
candidate_peaks = np.unique(np.concatenate([amp_peaks, flux_peaks]))
candidate_peaks = np.sort(candidate_peaks)  # indices into amp_env / flux arrays
candidate_times = candidate_peaks * HOP_LENGTH / sr
candidate_amps = amp_env[candidate_peaks]   # amplitude measure for selection in merging

# step 3.5: Merge nearby peaks into one event to envelope "frame" threshold:
# See img/progress.png and img/progress2.png to see issue with not merging
# See img/progress3-merged-peaks.png to see the after applying this merging
merge_frames = int(np.ceil((MERGE_MS / 1000.0) * sr / HOP_LENGTH))
if merge_frames < 1:
    merge_frames = 1

merged_peak_indices = []
i = 0
N = len(candidate_peaks)
while i < N:
    cluster = [candidate_peaks[i]]
    j = i + 1
    while j < N and (candidate_peaks[j] - candidate_peaks[j-1]) <= merge_frames:
        cluster.append(candidate_peaks[j])
        j += 1

    if len(cluster) == 1:
        # nothing to merge, take initial peak
        chosen = cluster[0]
    else:
        # choose the cluster peak with the largest amplitude envelope
        cluster_amps = amp_env[np.array(cluster)]
        chosen = cluster[np.argmax(cluster_amps)]

    merged_peak_indices.append(chosen)
    i = j

merged_peak_indices = np.array(merged_peak_indices, dtype=int)
merged_times = merged_peak_indices * HOP_LENGTH / sr
merged_amps = amp_env[merged_peak_indices]

print(f"Found {len(candidate_peaks)} initial candidate peaks; merged to {len(merged_times)} events")

# step 4: Plot waveform + markers
plt.figure(figsize=(14, 6), facecolor='lightgray')

# Waveform
plt.plot(times, waveform, alpha=0.7, label="Waveform")

# Plot merged peaks as vertical lines
plt.vlines(merged_times, ymin=-1, ymax=1, color='r',
    linestyle='--',   # dashed
    alpha=0.6,
    linewidth=1.2,
    label="Detected Peaks (merged)"
)

# add labels from typed text files
for t, lab in zip(candidate_times, typed_content):
    plt.text(
        t,                # x position
        1.1,              # y position (slightly above ymax=1)
        lab,
        ha='center',
        va='bottom',
        fontsize=8,
        color='r',
        alpha=0.7
    )

# Optional: overlay envelope (normalized)
plt.plot(amp_env_times, amp_env/amp_env.max(), color='orange', label="Amplitude Envelope (norm)")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Keystroke Detection - Candidate Peaks (merged)")
plt.legend()
plt.tight_layout()
plt.show()
