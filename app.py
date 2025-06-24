
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx
import os
import torch
from fastapi import FastAPI, UploadFile, File, Form # Added UploadFile, File, Form
import io # Added for BytesIO
# === CONFIG ===
model_path = r"C:\Users\arnav.shah\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct\snapshots\0cb88a4f764b7a12671c53f0838cd831a0843b95"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === FASTAPI SETUP ===
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === IN-MEMORY SESSION HISTORY ===
history_store = {}

# === UTILITIES ===
# This is the updated generate_response function.
# You should replace your old generate_response function with this one.
def generate_response(prompt, max_new_tokens=1024):
    print(f"[GENERATION] Generating response for prompt of length {len(prompt)}")
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
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # ✅ Remove repeated prompt (if echoed)
    if decoded_output.startswith(prompt):
        decoded_output = decoded_output[len(prompt):].strip()

    # ✅ If "Answer:" appears, split and return only what's after
    if "Answer:" in decoded_output:
        return decoded_output.split("Answer:", 1)[-1].strip()

    # ✅ Fallback: Try final answer marker
    if "The final answer is:" in decoded_output:
        return decoded_output.split("The final answer is:", 1)[-1].strip()

    # ✅ Fallback: Remove known preambles
    unwanted_prefixes = [
        "Assistant:", "Summary:", "**Summary**", 
        "You are a helpful assistant.", "Based on the context,",
        "Here's what I found:", "In summary:", "To summarize:"
    ]
    for prefix in unwanted_prefixes:
        if decoded_output.lower().startswith(prefix.lower()):
            decoded_output = decoded_output[len(prefix):].strip()

    return decoded_output

def build_prompt_with_history(history, user_input):
    prompt = ""
    for q, a in history:
        prompt += f"User: {q}\nAssistant: {a}\n"
    prompt += f"User: {user_input}\nAssistant:"
    return prompt

def build_rag_prompt(question: str, relevant_chunks: list[str]) -> str:
    context = "\n".join(relevant_chunks)
    return f"""You are a helpful AI assistant.
    You have been provided with the following context from a document:
    ---
    {context}
    ---
    Based on the context, please answer the following question: {question}"""

def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def retrieve_relevant_chunks(chunks, query, top_k=3):
    chunk_embeddings = embedding_model.encode(chunks)
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]
# === NEW UTILITY FOR FILE READING ===
def read_uploaded_file_content(file: UploadFile):
    file_extension = os.path.splitext(file.filename)[1].lower()
    # Read the file content into a BytesIO object for libraries that expect file-like objects
    file_content = io.BytesIO(file.file.read())

    if file_extension == ".txt":
        return file_content.getvalue().decode('utf-8')
    elif file_extension == ".pdf":
        reader = PdfReader(file_content)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file_extension == ".docx":
        # docx.Document expects a file-like object
        doc = docx.Document(file_content)
        return "\n".join(para.text for para in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
# === REQUEST MODELS ===
class ChatRequest(BaseModel):
    session_id: str
    message: str

#class RAGRequest(BaseModel):
 #   message: str
 #   document: str

# === ENDPOINTS ===
@app.post("/chat")
def chat(req: ChatRequest):
    print(f"[CHAT] Session: {req.session_id}, Message: {req.message}")
    session_id = req.session_id
    user_input = req.message
    history = history_store.get(session_id, [])
    prompt = build_prompt_with_history(history, user_input)
    response = generate_response(prompt)
    history.append((user_input, response))
    history_store[session_id] = history[-5:]  # keep last 5 turns
    print(f"[CHAT] Response: {response}")
    return {"response": response}

#@app.post("/rag")
#def rag_chat(req: RAGRequest):
 #   print(f"[RAG] Question: {req.message[:50]}... Document length: {len(req.document)}")
  #  chunks = chunk_text(req.document)
   # top_chunks = retrieve_relevant_chunks(chunks, req.message)
    #prompt = build_rag_prompt(req.message, top_chunks)
    #response = generate_response(prompt)
    #print(f"[RAG] Response: {response}")
    #return {"response": response}
    #print(f"[RAG] Received question: {question}")
    #print(f"[RAG] Document preview: {doc_text[:200]}...")
    # === NEW ENDPOINT for file uploads ===
@app.post("/rag_upload") # New endpoint name
async def rag_chat_upload(
    message: str = Form(...), # User message from form
    file: UploadFile = File(...) # The uploaded file
):
    print(f"[RAG_UPLOAD] Question: {message[:50]}... File Name: {file.filename}")
    try:
        document_content = read_uploaded_file_content(file) # Use the helper function
        print(f"[RAG_UPLOAD] Document content read (first 200 chars): {document_content[:200]}...")

        chunks = chunk_text(document_content)
        top_chunks = retrieve_relevant_chunks(chunks, message)
        prompt = build_rag_prompt(message, top_chunks)
        response = generate_response(prompt)
        print(f"[RAG_UPLOAD] Response: {response}")
        return {"response": response}
    except ValueError as e:
        print(f"[RAG_UPLOAD] Error: {e}")
        return {"response": f"Error: {e}"}
    except Exception as e:
        print(f"[RAG_UPLOAD] An unexpected error occurred: {e}")
        return {"response": "An internal server error occurred."}
