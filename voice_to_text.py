import speech_recognition as sr

# Create recognizer
r = sr.Recognizer()

# Use microphone as source
with sr.Microphone() as source:
    print("🎤 Speak something...")

    # Listen to user
    audio = r.listen(source)

    try:
        # Convert speech to text
        text = r.recognize_google(audio)
        print("You said:", text)

    except sr.UnknownValueError:
        print("😢 Sorry, I couldn't understand.")
    except sr.RequestError:
        print("⚠️ Network error.")
    from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert to embedding
embedding = model.encode(text)

print("🧠 Meaning understood! (Text embedding vector)")
print(embedding[:10])  # Print first 10 values

