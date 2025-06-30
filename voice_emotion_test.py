import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import scipy.io.wavfile as wav
import os

# Record audio
fs = 22050  # Sampling rate
seconds = 5
print("Recording started... Speak now.")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
print("Recording finished.")

# Save to file
wav.write("voice_input.wav", fs, audio)

# Load and extract features using librosa
y, sr = librosa.load("voice_input.wav", sr=fs)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs_mean = np.mean(mfccs.T, axis=0)

print("MFCC features extracted:")
print(mfccs_mean)
