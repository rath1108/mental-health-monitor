from transformers import pipeline

# Load a pre-trained emotion classification model from Hugging Face
classifier = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

# Sample inputs
texts = [
    "I feel very sad and lonely.",
    "I'm so happy and excited about my future!",
    "I'm anxious about the exam tomorrow."
]

# Analyze each text
for text in texts:
    result = classifier(text)
    print(f"Input: {text}")
    print(f"Prediction: {result[0]['label']} (Score: {round(result[0]['score'], 2)})\n")
