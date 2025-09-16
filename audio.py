# This will be loosely formatted as I form thoughts. No linting or prettier for this stage

# 1. load audio file (apple voice recording format is .m4a)
# I am converting manually via https://cloudconvert.com/m4a-to-wav
# I wonder if the bitrate drastically drops from conversion

import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
# This would be a parameter or audio source in the future
# audio_file = './typing_1.wav'
audio_file = './typing_2_abc.wav'

sample_rate, data = wavfile.read(audio_file)
# convert to mono if stereo
if data.ndim > 1:
    data = np.mean(data, axis=1)



# 2. compute the amplitude envelope (how loud) + spectral flux (did it change suddenly).
# alt method ignored for now: take the absolute value of the signal and then applying a smoothing filter (e.g., a rolling mean).
from scipy.signal import hilbert

analytic_signal = hilbert(data)
amplitude_envelope = np.abs(analytic_signal)
# x-axis
time = np.arange(len(data)) / sample_rate


# 3. detect peaks (these would be keystrokes) (helps reduce background noise by combining envelope and flux)


# 4. plot waveform with detected keystrokes with some kkind of markers
plt.figure(figsize=(12, 6))
plt.plot(time, data, label='Original Waveform', alpha=0.7)
plt.plot(time, amplitude_envelope, label='Amplitude Envelope', color='red', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('WAV Amplitude Envelope')
plt.legend()
plt.grid(True)
plt.show()