import os
import torch
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM

# === CONFIG ===
model_path = r"C:\Users\arnav.shah\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct\snapshots\0cb88a4f764b7a12671c53f0838cd831a0843b95"
device = "cpu"  # or "cuda"

# === Load LLaMA model ===
print("🔄 Loading LLaMA model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
dtype = torch.float16 if device != "cpu" else torch.float32
llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
print("✅ Model loaded.")

# === Document Loader ===
def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_pdf_file(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def load_docx_file(path):
    doc = docx.Document(path)
    return "\n".join(para.text for para in doc.paragraphs)

# === RAG Utilities ===
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def retrieve_relevant_chunks(embedding_model, chunks, query, top_k=3):
    chunk_embeddings = embedding_model.encode(chunks)
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]

def build_prompt(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    # Add explicit summarization instruction if the query implies it
    if "summary" in query.lower() or "summarize" in query.lower():
        return f"You are a helpful assistant. Based on the following document excerpts, provide a concise and comprehensive summary of the document.\n\nContext:\n{context}\n\nSummary:"
    else:
        return f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""

def generate_response(prompt, max_new_tokens=256):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in decoded:
            return decoded.split("Answer:")[-1].strip()
        return decoded.strip()
    except Exception as e:
        return f"[Error] {e}"

# === Main Chat Loop ===
def rag_chat_loop(chunks, embedding_model):
    print("\n🔍 Ask questions based on the uploaded document (type 'exit' to quit):\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("👋 Goodbye.")
            break
        relevant = retrieve_relevant_chunks(embedding_model, chunks, question)
        prompt = build_prompt(question, relevant)
        response = generate_response(prompt)
        print(f"Assistant: {response}")

# === Entry Point ===
if __name__ == "__main__":
    filepath = input("📂 Enter document path (.txt, .pdf, .docx): ").strip()
    filepath = filepath.strip('"').replace("\\", "/")
    if not os.path.exists(filepath):
        print("❌ File not found.")
        exit()

    if filepath.endswith(".txt"):
        content = load_text_file(filepath)
    elif filepath.endswith(".pdf"):
        content = load_pdf_file(filepath)
    elif filepath.endswith(".docx"):
        content = load_docx_file(filepath)
    else:
        print("❌ Unsupported file type.")
        exit()

    chunks = chunk_text(content)
    print(f"📄 Document loaded and split into {len(chunks)} chunks.")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    rag_chat_loop(chunks, embedding_model)
