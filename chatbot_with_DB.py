import os
import sqlite3
import json
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

# === CONFIG ===
model_path = r"C:\Users\arnav.shah\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct\snapshots\0cb88a4f764b7a12671c53f0838cd831a0843b95"
device = "cpu"  # or "cuda" if GPU is available
DB_PATH = "chat_history.db"
MAX_HISTORY_TURNS = 5

# === DB FUNCTIONS ===
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_input TEXT,
            assistant_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def save_conversation(session_id, user_input, assistant_response):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (session_id, user_input, assistant_response) VALUES (?, ?, ?)",
        (session_id, user_input, assistant_response)
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id, limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_input, assistant_response FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return list(reversed(rows))  # Oldest first

def delete_last_message(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM chat_history WHERE id = (SELECT MAX(id) FROM chat_history WHERE session_id = ?)",
        (session_id,)
    )
    conn.commit()
    conn.close()

def clear_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def export_history(session_id, filename):
    history = get_chat_history(session_id, limit=1000)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def import_history(session_id, filename):
    with open(filename, "r", encoding="utf-8") as f:
        history = json.load(f)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for user_input, assistant_response in history:
        cursor.execute(
            "INSERT INTO chat_history (session_id, user_input, assistant_response) VALUES (?, ?, ?)",
            (session_id, user_input, assistant_response)
        )
    conn.commit()
    conn.close()

def search_history(session_id, keyword, limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    pattern = f"%{keyword}%"
    cursor.execute(
        """SELECT user_input, assistant_response FROM chat_history 
           WHERE session_id = ? AND (user_input LIKE ? OR assistant_response LIKE ?)
           ORDER BY id DESC LIMIT ?""",
        (session_id, pattern, pattern, limit)
    )
    results = cursor.fetchall()
    conn.close()
    return list(reversed(results))

# === MODEL LOAD ===
print("🔄 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
dtype = torch.float16 if device != "cpu" else torch.float32
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
model.to(device)
model.eval()
print("✅ Model loaded.")

# === PROMPT & GENERATION ===
def build_prompt_with_history(history, user_input):
    prompt = ""
    for q, a in history:
        prompt += f"User: {q}\nAssistant: {a}\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

def generate_response(prompt, max_new_tokens=256):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean and isolate final assistant reply
        last_part = decoded.split("User:")[-1]
        if "Assistant:" in last_part:
            return last_part.split("Assistant:")[-1].strip()
        return decoded.strip()
    except Exception as e:
        return f"[Error] Failed to generate response: {e}"

# === CHAT LOOP ===
def chat_loop(session_id):
    print(f"\n🧠 LLaMA 3.2B Chatbot with DB Memory — session: '{session_id}' — type 'exit' to quit.")
    print("Commands:")
    print("  /forget_last      - Delete last message")
    print("  /clear_history    - Clear all chat history")
    print("  /export filename  - Export chat history to JSON")
    print("  /import filename  - Import chat history from JSON")
    print("  /search keyword   - Search chat history\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Bye!")
            break

        # === COMMAND HANDLING ===
        if user_input.startswith("/forget_last"):
            delete_last_message(session_id)
            print("[System] Last message deleted.")
            continue

        if user_input.startswith("/clear_history"):
            clear_history(session_id)
            print("[System] Chat history cleared.")
            continue

        if user_input.startswith("/export "):
            filename = user_input.split(maxsplit=1)[1]
            export_history(session_id, filename)
            print(f"[System] History exported to '{filename}'.")
            continue

        if user_input.startswith("/import "):
            filename = user_input.split(maxsplit=1)[1]
            if not os.path.exists(filename):
                print(f"[System] File '{filename}' does not exist.")
                continue
            import_history(session_id, filename)
            print(f"[System] History imported from '{filename}'.")
            continue

        if user_input.startswith("/search "):
            keyword = user_input.split(maxsplit=1)[1]
            results = search_history(session_id, keyword)
            if not results:
                print("[System] No results found.")
            else:
                print("[System] Search results:")
                for u, a in results:
                    print(f"Q: {u}\nA: {a}\n---")
            continue

        # === NORMAL CHAT FLOW ===
        history = get_chat_history(session_id, limit=MAX_HISTORY_TURNS)
        prompt = build_prompt_with_history(history, user_input)
        assistant_response = generate_response(prompt)
        print(f"Assistant: {assistant_response}")
        save_conversation(session_id, user_input, assistant_response)

# === ENTRY POINT ===
if __name__ == "__main__":
    init_db()
    session_id = input("Enter session id (username or chat id): ").strip()
    chat_loop(session_id)
