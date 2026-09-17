# import os
# import shutil
# import pathlib
# import fitz
# import faiss
# import numpy as np
# import requests
# from fastapi import FastAPI, Request, UploadFile, File, Form
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# llm = genai.GenerativeModel("gemini-3.6-flash")


# HF_TOKEN = os.getenv("HF_TOKEN", "")
# HF_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

# def get_embeddings_api(texts):
#     if HF_TOKEN:
#         headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#         r = requests.post(HF_URL, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}}, timeout=60)
#         if r.status_code == 200:
#             return np.array(r.json()).astype('float32')
#     # If no HF token, fallback to local TF-IDF 
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     global vectorizer
#     vectorizer = TfidfVectorizer(max_features=384)
#     return vectorizer.fit_transform(texts).toarray().astype('float32')

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# chunks_db = []
# faiss_index = None
# vectorizer = None

# def chunk_text(text, chunk_size=600, overlap=100):
#     chunks = []
#     for i in range(0, len(text), chunk_size - overlap):
#         chunks.append(text[i:i+chunk_size])
#     return chunks

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse(request, "index.html", {
#         "answer": None, "sources": [],
#         "docs_uploaded": len(set([c['doc_name'] for c in chunks_db])) if chunks_db else 0
#     })

# @app.post("/upload", response_class=HTMLResponse)
# async def upload_papers(request: Request, files: list[UploadFile] = File(...)):
#     global chunks_db, faiss_index, vectorizer
#     if not files or files[0].filename == "":
#         return templates.TemplateResponse(request, "index.html", {
#             "error": "No document uploaded.", "answer": None, "sources": [],
#             "docs_uploaded": len(set([c['doc_name'] for c in chunks_db])) if chunks_db else 0
#         })
#     uploads_dir = pathlib.Path("uploads")
#     if uploads_dir.is_file(): uploads_dir.unlink()
#     uploads_dir.mkdir(parents=True, exist_ok=True)

#     new_chunks = []
#     for file in files:
#         if not file.filename.lower().endswith(".pdf"):
#             return templates.TemplateResponse(request, "index.html", {
#                 "error": f"Only PDF: {file.filename}", "answer": None, "sources": [],
#                 "docs_uploaded": len(set([c['doc_name'] for c in chunks_db])) if chunks_db else 0
#             })
#         file_path = uploads_dir / file.filename
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)
#         doc = fitz.open(str(file_path))
#         for page_num in range(len(doc)):
#             page_text = doc[page_num].get_text()
#             if not page_text.strip(): continue
#             for chunk in chunk_text(page_text):
#                 new_chunks.append({"text": chunk, "doc_name": file.filename, "page_no": page_num + 1})

#     if new_chunks:
#         all_texts = [c["text"] for c in chunks_db] + [c["text"] for c in new_chunks]
        
#         if HF_TOKEN:
#             headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#             r = requests.post(HF_URL, headers=headers, json={"inputs": all_texts, "options": {"wait_for_model": True}}, timeout=120)
#             embeddings = np.array(r.json()).astype('float32')
#         else:
#             from sklearn.feature_extraction.text import TfidfVectorizer
#             vectorizer = TfidfVectorizer(max_features=384)
#             embeddings = vectorizer.fit_transform(all_texts).toarray().astype('float32')

#         faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
#         faiss_index.add(embeddings)

#         chunks_db = [{"text": all_texts[i], "doc_name": (chunks_db + new_chunks)[i]["doc_name"], "page_no": (chunks_db + new_chunks)[i]["page_no"]} for i in range(len(all_texts))]

#     return templates.TemplateResponse(request, "index.html", {
#         "message": f"Uploaded {len(files)} paper(s). Total chunks: {len(chunks_db)} - Using all-MiniLM-L6-v2 embeddings!",
#         "answer": None, "sources": [],
#         "docs_uploaded": len(set([c['doc_name'] for c in chunks_db]))
#     })

# @app.post("/ask", response_class=HTMLResponse)
# async def ask_question(request: Request, question: str = Form(None), query: str = Form(None), prompt: str = Form(None), q: str = Form(None)):
#     global faiss_index, chunks_db, vectorizer
#     form_data = await request.form()
#     actual_question = (question or query or prompt or q or form_data.get("question") or form_data.get("query") or form_data.get("prompt") or form_data.get("q") or form_data.get("message"))
#     if not actual_question or not str(actual_question).strip():
#         return templates.TemplateResponse(request, "index.html", {"error": "Question empty.", "answer": None, "sources": [], "docs_uploaded": len(set([c['doc_name'] for c in chunks_db])) if chunks_db else 0})
#     actual_question = str(actual_question).strip()
#     if not chunks_db or faiss_index is None:
#         return templates.TemplateResponse(request, "index.html", {"error": "Upload PDFs first.", "answer": None, "sources": [], "docs_uploaded": 0})

#     if HF_TOKEN:
#         headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#         r = requests.post(HF_URL, headers=headers, json={"inputs": [actual_question], "options": {"wait_for_model": True}}, timeout=30)
#         q_emb = np.array(r.json()).astype('float32')
#     else:
#         if vectorizer is None:
#             q_emb = np.zeros((1, 384), dtype='float32')
#         else:
#             q_emb = vectorizer.transform([actual_question]).toarray().astype('float32')

#     D, I = faiss_index.search(q_emb, k=5)
#     retrieved = [chunks_db[idx] for idx in I[0] if idx < len(chunks_db)]
#     context_str = "".join([f"\n[Source: {c['doc_name']} | Page {c['page_no']}]\n{c['text']}\n" for c in retrieved])
#     prompt_final = f"You are Research Paper Assistant. Answer ONLY from context. If not in context say not available.\nContext:{context_str}\nQuestion: {actual_question}\nMention doc and page."
#     response = llm.generate_content(prompt_final)
#     return templates.TemplateResponse(request, "index.html", {"answer": response.text, "sources": retrieved, "question": actual_question, "docs_uploaded": len(set([c['doc_name'] for c in chunks_db]))})


import os, re
from pathlib import Path
from typing import List
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
BASE_DIR = Path(__file__).parent
#app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
#UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

chunks_db = []

def chunk_text(text, size=800, overlap=150):
    chunks=[]; start=0
    while start < len(text):
        end=start+size
        chunks.append(text[start:end])
        start=end-overlap
    return [c.strip() for c in chunks if len(c.strip())>60]

def simple_search(question, chunks, top_k=6):
    q_words = set(re.findall(r'\w+', question.lower()))
    scored=[]
    for c in chunks:
        c_words = set(re.findall(r'\w+', c['text'].lower()))
        overlap = len(q_words & c_words)
        score = overlap / (len(q_words)+1)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for s,c in scored[:top_k] if s>0] or [c for s,c in scored[:3]]

def get_model():
    try:
        return genai.GenerativeModel("gemini-3.6-flash")
    except:
        return genai.GenerativeModel("gemini-2.0-flash")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    docs = list(set(c['doc_name'] for c in chunks_db))
    return templates.TemplateResponse(request, "index.html", {"docs_uploaded": len(docs), "doc_names": docs, "answer": None, "total_chunks": len(chunks_db)})

@app.post("/upload", response_class=HTMLResponse)
async def upload_pdfs(request: Request, files: List[UploadFile] = File(...)):
    global chunks_db
    added=0
    for f in files:
        if not f.filename.lower().endswith('.pdf'): continue
        path = UPLOAD_DIR / f.filename
        path.write_bytes(await f.read())
        try:
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                for ch in chunk_text(txt):
                    chunks_db.append({"text": ch, "doc_name": f.filename, "page_no": i+1})
                    added+=1
        except Exception as e:
            print(e)
    docs = list(set(c['doc_name'] for c in chunks_db))
    return templates.TemplateResponse(request, "index.html", {"message": f"✅ {len(files)} files uploaded! {added} chunks.", "docs_uploaded": len(docs), "doc_names": docs, "total_chunks": len(chunks_db), "answer": None})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...), summarize: str = Form(None)):
    docs = list(set(c['doc_name'] for c in chunks_db))
    if not chunks_db:
        return templates.TemplateResponse(request, "index.html", {"error": "Upload PDFs first!", "docs_uploaded": 0, "answer": None})

    top_chunks = simple_search(question, chunks_db, top_k=6)
    context_str = "\n\n".join([f"[Source: {c['doc_name']} | Page {c['page_no']}]\n{c['text'][:1200]}" for c in top_chunks])

    if summarize:
        prompt = f"Summarize based ONLY on context with citations [Source: filename | Page X]\n\nContext:\n{context_str}\n\nSummary:"
    else:
        prompt = f"Answer based ONLY on context. MUST cite as [Source: filename | Page X]\n\nContext:\n{context_str}\n\nQuestion: {question}\nAnswer:"

    try:
        model = get_model()
        answer_text = model.generate_content(prompt).text
    except Exception as e:
        answer_text = f"AI Error: {e}\n\nContext:\n{context_str[:2500]}"

    return templates.TemplateResponse(request, "index.html", {"answer": answer_text, "question": question, "sources": top_chunks, "docs_uploaded": len(docs), "doc_names": docs, "total_chunks": len(chunks_db)})

@app.post("/clear")
async def clear_db(request: Request):
    global chunks_db
    chunks_db=[]
    for f in UPLOAD_DIR.glob("*.pdf"):
        try: f.unlink()
        except: pass
    return templates.TemplateResponse(request, "index.html", {"message": "Cleared!", "docs_uploaded": 0, "answer": None})