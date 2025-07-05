import os
import streamlit as st
from transformers import pipeline
import torch
from gtts import gTTS
import tempfile

# Optional: for translation
from transformers import pipeline as hf_pipeline

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="📖 AI Storyteller", layout="centered")

# --- Sidebar Info ---
with st.sidebar:
    st.title("📌 Instructions")
    st.markdown("""
    1. Enter a **story prompt**
    2. Choose **genre**, **tone**, **length**, and **language**
    3. Click **Generate Story**
    4. Enjoy reading or listening to your story! 🎧
    """)
    st.markdown("---")
    st.caption("🌍 Supports multiple languages for narration")

# --- Title ---
st.title("🧠 AI Storyteller with Multilingual Narration")
st.markdown("Let AI craft a story and narrate it in your language!")

# --- Load Story Generator ---
@st.cache_resource
def load_story_model():
    return pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        device=0 if torch.cuda.is_available() else -1
    )

story_gen = load_story_model()

# --- Load Translator Pipeline ---
@st.cache_resource
def load_translators():
    return {
        "hi": hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi"),
        "fr": hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
        "es": hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    }

translators = load_translators()

# --- Genre & Tone Options ---
genre = st.selectbox("📚 Choose a Genre", ["Fantasy", "Sci-Fi", "Mystery", "Comedy", "Adventure", "Horror"])
tone = st.selectbox("🎭 Select the Tone", ["Light-hearted", "Dramatic", "Suspenseful", "Funny", "Inspiring"])
length = st.slider("🧾 Story Length (in words)", min_value=100, max_value=1000, step=100, value=300)

# --- Language Selection ---
language_options = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es"
}
lang_label = st.selectbox("🌐 Choose Narration Language", list(language_options.keys()))
lang_code = language_options[lang_label]

# --- User Prompt ---
prompt = st.text_input("📝 Enter your story prompt:", "A dragon who learns to cook pizza")

# --- Generate Story Function ---
def generate_story(prompt, genre, tone, length):
    instruction = (
        f"story with a {tone.lower()} tone about: {prompt}. "
    )
    result = story_gen(
        instruction,
        max_length=int(length * 1.5),
        do_sample=True,
        top_k=50
    )
    return result[0]["generated_text"]

# --- Translate Story if Needed ---
def translate_story(story_text, target_lang):
    if target_lang == "en":
        return story_text
    elif target_lang in translators:
        translated = translators[target_lang](story_text[:1000])  # limit to 1000 tokens
        return translated[0]['translation_text']
    else:
        return story_text

# --- Text to Speech ---
def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts.save(tmpfile.name)
        return tmpfile.name

# --- Generate Button ---
if st.button("✨ Generate Story"):
    if not prompt.strip():
        st.warning("Please enter a story prompt.")
    else:
        with st.spinner("Generating your story..."):
            story = generate_story(prompt, genre, tone, length)

        st.subheader("📜 Your AI Story (English)")
        st.write(story)

        with st.spinner(f"Translating story to {lang_label}..."):
            translated_story = translate_story(story, lang_code)

        st.subheader(f"🌍 Translated Story ({lang_label})")
        st.write(translated_story)

        with st.spinner("Generating narration..."):
            audio_path = text_to_speech(translated_story, lang_code)

        st.subheader("🔊 Listen to the Story")
        st.audio(audio_path, format="audio/mp3")

# --- Footer ---
st.markdown("---")
st.caption("© 2025 | Built with ❤️ using Streamlit, Hugging Face, and gTTS")
