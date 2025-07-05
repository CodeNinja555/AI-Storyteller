# AI-Storyteller

# 🧠 AI Storyteller with Multilingual Narration

This project is an AI-powered storytelling app that uses Hugging Face Transformers, Streamlit, and gTTS to generate engaging stories from user prompts. Users can customize the story’s **genre**, **tone**, and **length**, translate it into multiple languages, and listen to an AI-generated **narration** in their selected language.

---

## 🚀 Features

📜 Enter your **story prompt** and select genre/tone  
📚 Choose from **Fantasy**, **Sci-Fi**, **Mystery**, and more  
🎭 Set the **tone**: Light-hearted, Dramatic, Funny, etc.  
🧾 Choose **story length** (100–1000 words)  
🌐 **Multilingual translation**: English, Hindi, French, Spanish  
🔊 **Narration** using gTTS (Google Text-to-Speech)  
📥 Downloadable **MP3 narration**  
🎨 Streamlit-based **UI** for seamless interaction  

---

## 🛠️ Tech Stack

| Tool             | Purpose                          |
|------------------|----------------------------------|
| 🧠 Hugging Face Transformers | Story generation & translation |
| 🌐 Falcon-7B-Instruct         | Large Language Model (LLM)     |
| 🌍 Helsinki-NLP Models        | Language translation            |
| 🔊 gTTS                       | Text-to-speech narration       |
| 🎨 Streamlit                  | Web UI Framework               |
| 🐍 Python                     | Core Logic                     |
| 🗂️ tempfile/os               | Temporary MP3 handling         |

---

## 📦 Installation

### 🔧 Prerequisites
- Python 3.8+
- Good internet connection (for downloading models)
- Optional: GPU for faster inference

### 📥 Clone and Install

```bash
git clone https://github.com/your-username/ai-storyteller.git
cd ai-storyteller
pip install -r requirements.txt
