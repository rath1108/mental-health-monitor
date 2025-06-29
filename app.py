import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime
import os

# Load the emotion detection model
classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

# UI title
st.title("🧠 Mental Health AI Check")
st.write("Type how you're feeling. The AI will detect your emotion.")

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
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Input": user_input,
        "Emotion": emotion,
        "Score": score
    }

    # Create or append to CSV
    log_file = "user_logs.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = df.append(log_entry, ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])

    df.to_csv(log_file, index=False)

# --- Mood Trend Visualization Section ---
st.markdown("---")
st.subheader("📊 Mood Tracker - Emotion History")

# Check if log file exists
if os.path.exists("user_logs.csv"):
    df_logs = pd.read_csv("user_logs.csv")
    df_logs['Timestamp'] = pd.to_datetime(df_logs['Timestamp'])

    # Line Chart - Emotions Over Time
    st.line_chart(df_logs.set_index('Timestamp')['Score'])

    # Bar Chart - Emotion Counts
    emotion_counts = df_logs['Emotion'].value_counts()
    st.bar_chart(emotion_counts)
else:
    st.info("No mood data yet. Start by entering how you feel above.")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 AI Mental Health Monitor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Check your mood, track your mind, take care of yourself 💚</h5>", unsafe_allow_html=True)
