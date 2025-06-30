import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import joblib

# Record 5 seconds of audio
fs = 22050
seconds = 5
print("🎙️ Speak now. Recording for 5 seconds...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
print("✅ Recording finished!")

# Save and load audio
wav.write("voice_input.wav", fs, audio)
y, sr = librosa.load("voice_input.wav", sr=fs)

# Extract features
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

# Load trained model
model = joblib.load("voice_emotion_model.pkl")

# Predict
predicted_emotion = model.predict(mfcc_mean)[0]
print(f"🧠 Predicted Emotion: {predicted_emotion}")
