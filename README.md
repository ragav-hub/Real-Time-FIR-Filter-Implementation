# Real-Time-FIR-Filter-Implementation
#Implement a low-pass FIR filter on a DSP processor/Arduino with audio input.

#Aim
To design and implement a low-pass Finite Impulse Response (FIR) filter and demonstrate real-time (chunked) filtering of an audio signal using Python in Google Colab. The objective is to remove high-frequency noise from audio and visualize/compare input vs filtered waveforms and spectra.

#Apparatus (Software / Virtual Lab)

Google Colab (web)
Python packages: numpy, scipy, matplotlib, soundfile, IPython.display
(Colab install commands included in the program)
An audio file to upload (mono or stereo .wav, .flac, or .mp3 — stereo will be converted to mono)
(Optional) Google Drive if you prefer to load/store large files

#Program
```
# === Real-Time FIR Low-Pass Filter (Colab) ===
# Paste into Google Colab and run.

# Install required packages (run once)
!pip install -q scipy matplotlib numpy soundfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz
import soundfile as sf
from IPython.display import Audio, display
from google.colab import files
import io

# ---------------------------
# 1) Upload audio file
# ---------------------------
print("Upload a mono/stereo audio file (wav/mp3/flac).")
uploaded = files.upload()  # Colab file picker
filename = list(uploaded.keys())[0]
print("Loaded:", filename)

# Read audio (soundfile auto-detects format)
audio_in, fs = sf.read(io.BytesIO(uploaded[filename]))
if audio_in.ndim > 1:
    audio_in = audio_in[:, 0]   # convert to mono (use left channel)
audio_in = audio_in.astype(np.float32)
duration_s = len(audio_in) / fs
print(f"Sampling rate: {fs} Hz, Duration: {duration_s:.2f} s, Samples: {len(audio_in)}")

# Normalize if not in -1..1
max_abs = np.max(np.abs(audio_in))
if max_abs > 1.0:
    audio_in = audio_in / max_abs
    print("Normalized audio to [-1,1].")

# ---------------------------
# 2) FIR design parameters
# ---------------------------
cutoff_hz = 1000      # cutoff frequency in Hz (change as needed)
numtaps = 51          # number of taps (odd recommended)
window = 'hamming'    # window type

# Design FIR low-pass
fir_coeff = firwin(numtaps, cutoff_hz, fs=fs, window=window, pass_zero=True)
print(f"Designed {numtaps}-tap FIR low-pass (cutoff {cutoff_hz} Hz).")

# Show frequency response
w, h = freqz(fir_coeff, worN=8000, fs=fs)
plt.figure(figsize=(8,3))
plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-12)))
plt.title("FIR Frequency Response (dB)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.savefig("freq_response.png", dpi=150)
plt.show()

# ---------------------------
# 3) Simulated real-time filtering (chunked)
# ---------------------------
chunk_size = 256  # emulate processing block size
output = np.zeros_like(audio_in)
zi = np.zeros(numtaps-1)  # filter state (for continuity between chunks)

for start in range(0, len(audio_in), chunk_size):
    chunk = audio_in[start : start + chunk_size]
    # lfilter with state to simulate streaming
    y, zi = lfilter(fir_coeff, 1.0, chunk, zi=zi)
    output[start : start + len(y)] = y

# Save filtered audio
sf.write("filtered_audio.wav", output, fs)
print("Filtered audio saved as 'filtered_audio.wav'")

# ---------------------------
# 4) Visualization and playback
# ---------------------------
t = np.arange(len(audio_in)) / fs

# Time-domain plot (input vs filtered)
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, audio_in, linewidth=0.6)
plt.title("Original Audio (time domain)")
plt.ylabel("Amplitude")
plt.xlim(0, min(2.0, duration_s))  # show first 2 seconds for clarity

plt.subplot(2,1,2)
plt.plot(t, output, linewidth=0.6)
plt.title("Filtered Audio (time domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(0, min(2.0, duration_s))
plt.tight_layout()
plt.savefig("time_domain_compare.png", dpi=150)
plt.show()

# Frequency domain (FFT) comparison
def plot_spectrum(sig, fs, title):
    N = min(16384, len(sig))
    X = np.fft.rfft(sig[:N] * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1/fs)
    mag = 20*np.log10(np.abs(X) + 1e-12)
    plt.semilogx(freqs, mag, linewidth=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True)

plt.figure(figsize=(10,5))
plot_spectrum(audio_in, fs, "Spectrum - Original")
plot_spectrum(output, fs, "Spectrum - Filtered")
plt.legend(["Original","Filtered"])
plt.savefig("spectrum_compare.png", dpi=150)
plt.show()

# Playback widgets
print("Original audio:")
display(Audio(audio_in, rate=fs))
print("Filtered audio:")
display(Audio(output, rate=fs))

# Provide downloads
files.download("filtered_audio.wav")
files.download("time_domain_compare.png")
files.download("spectrum_compare.png")
files.download("freq_response.png")
```

#Original Audio
[Lokiverse_Theme_Video_-_Vikram_Ka_(getmp3.pro).mp3](https://github.com/user-attachments/files/22945034/Lokiverse_Theme_Video_-_Vikram_Ka_.getmp3.pro.mp3)

#Filtered Audio

[filtered_audio.wav](https://github.com/user-attachments/files/22945062/filtered_audio.wav)

#Output Waveform
<img width="1200" height="450" alt="freq_response" src="https://github.com/user-attachments/assets/a0084cce-60bc-446e-b663-e32d8b49e763" />

<img width="1500" height="900" alt="time_domain_compare" src="https://github.com/user-attachments/assets/2e450bea-8c44-4c1a-a263-89903757ad4d" />

<img width="1500" height="750" alt="spectrum_compare" src="https://github.com/user-attachments/assets/46b661a9-e577-40e4-9aae-15f78f41cef8" />

#Conclusion
Outcome: Using the FIR low-pass filter (Hamming window, numtaps=51, cutoff=1000 Hz) we successfully simulated real-time chunked filtering of an uploaded audio file in Google Colab. The filtered waveform and spectrum verify attenuation of high-frequency components and audible noise reduction.



