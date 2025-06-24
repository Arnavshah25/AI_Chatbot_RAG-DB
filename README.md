# 🤖 AI Chatbot with Memory & Document RAG

A full-stack, local-first chatbot project featuring:

- 💬 Normal conversation with short-term memory (up to 5 turns)
- 📄 Document-based Q&A via RAG (Retrieval-Augmented Generation)
- 🖥️ ChatGPT-style UI using only HTML, CSS & JS
- ⚡ FastAPI backend with local LLaMA model

---

## ✨ Features

- ✅ **Normal Chat Mode** — remembers conversation using session ID
- 📂 **RAG Mode** — upload `.txt`, `.pdf`, or `.docx` documents and ask questions
- 🔀 **Mode Toggle** — switch between Chat and RAG modes via UI
- 📥 **File Upload** — see uploaded document name and ask questions from it
- 🧠 **Local Model Inference** — no external API calls; runs offline
- 📱 **Responsive UI** — desktop-first, mobile-adaptive design

---

## 🧰 Tech Stack

| Layer         | Technology                                   |
|---------------|----------------------------------------------|
| Backend       | [FastAPI](https://fastapi.tiangolo.com/)     |
| Language Model| LLaMA 3.2B (via Hugging Face Transformers)   |
| Embeddings    | `sentence-transformers` (MiniLM)             |
| Frontend      | Vanilla HTML, CSS, JavaScript                |
| DB (Optional) | SQLite (in `chatbot_with_DB.py`)             |

---

## 🗂️ Project Structure
📦 your-chatbot-project/
├── app.py # FastAPI backend entry point
├── chatbot_with_DB.py # SQLite-based chat memory (used in normal mode)
├── chatbot_with_rag.py # RAG functions: chunking, embeddings, retrieval
├── index.html # Beautiful responsive frontend
├── README.md # You're reading this

---

## 🧰 Tech Stack

| Layer         | Technology                               |
|---------------|------------------------------------------|
| Backend       | [FastAPI](https://fastapi.tiangolo.com/) |
| LLM           | Meta LLaMA 3.2B                          |
| Embeddings    | `all-MiniLM-L6-v2` via SentenceTransformers |
| Frontend      | HTML, CSS, JS (Vanilla)                  |
| Chat Memory   | SQLite (`chatbot_with_DB.py`)            |


