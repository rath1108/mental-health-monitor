import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime
import os
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import joblib
import numpy as np

# Setup for Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google_sheets_creds.json", scope)
client = gspread.authorize(creds)
sheet = client.open("MentalHealthLogs").sheet1  # Name of your sheet

# Load the emotion detection model
classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

# UI title
st.title("🧠 Mental Health AI Check")
st.write("Type how you're feeling. The AI will detect your emotion.")

# User login name
user_name = st.text_input("👤 Enter your name or email to begin:")

if not user_name:
    st.warning("Please enter your name to start.")
    st.stop()

# User input
user_input = st.text_input("How are you feeling today?")

# If user enters something
if user_input:
    result = classifier(user_input)
    emotion = result[0]['label']
    score = round(result[0]['score'], 2)

    st.write(f"**Emotion Detected:** {emotion}")
    st.write(f"**Confidence Score:** {score}")

    if emotion in ['sadness', 'fear', 'anger']:
        st.warning("⚠️ Consider talking to someone or seeking help. You're not alone ❤️")
    else:
        st.success("😊 You're doing great!")

    # Prepare log entry
    log_entry = {
        "User": user_name,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Input": user_input,
        "Emotion": emotion,
        "Score": score
    }

    # Save to Google Sheets
    sheet.append_row([
        log_entry["User"],
        log_entry["Timestamp"],
        log_entry["Input"],
        log_entry["Emotion"],
        log_entry["Score"]
    ])

# --- Mood Trend Visualization Section ---
st.markdown("---")
st.subheader("📊 Mood Tracker - Emotion History")

try:
    # Load logs from Google Sheets
    records = sheet.get_all_records()
    df_logs = pd.DataFrame(records)
    df_logs = df_logs[df_logs["User"] == user_name]

    if not df_logs.empty:
        df_logs['Timestamp'] = pd.to_datetime(df_logs['Timestamp'], errors='coerce')
        df_logs = df_logs.dropna()

        # Line Chart - Emotions Over Time (numerical score only)
        if "Score" in df_logs.columns:
            df_logs["Score"] = pd.to_numeric(df_logs["Score"], errors="coerce")
            df_logs = df_logs.dropna(subset=["Score"])
            st.line_chart(df_logs.set_index('Timestamp')["Score"])

        # Bar Chart - Emotion Counts
        emotion_counts = df_logs['Emotion'].value_counts()
        st.bar_chart(emotion_counts)
    else:
        st.info("No mood data yet. Start by entering how you feel above.")

except Exception as e:
    st.error(f"❌ Error loading history: {e}")

# --- Voice Section ---
st.markdown("---")
st.subheader("🎙️ Voice Emotion Detection")

if st.button("Record My Voice"):
    try:
        fs = 22050
        seconds = 5
        st.info("🎙️ Recording... Speak now.")
        audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        wav.write("voice_input.wav", fs, audio)

        # Load audio and extract features
        y, sr = librosa.load("voice_input.wav", sr=fs)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Load trained model
        model = joblib.load("voice_emotion_model.pkl")
        voice_emotion = model.predict(mfcc_mean)[0]

        st.success(f"🧠 Voice Emotion Detected: **{voice_emotion.upper()}**")

        # Create voice log entry
        voice_log = {
            "User": user_name,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Input": "[Voice]",
            "Emotion": voice_emotion,
            "Score": "N/A"
        }

        # Save to Google Sheets
        sheet.append_row([
            voice_log["User"],
            voice_log["Timestamp"],
            voice_log["Input"],
            voice_log["Emotion"],
            voice_log["Score"]
        ])

    except Exception as e:
        st.error(f"❌ Error during voice processing: {e}")

# --- Footer ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 AI Mental Health Monitor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Check your mood, track your mind, take care of yourself 💚</h5>", unsafe_allow_html=True)
