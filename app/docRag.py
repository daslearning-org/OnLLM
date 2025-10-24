import sqlite3, json, re
import numpy as np
from onnxruntime import InferenceSession
from docx import Document
from pypdf import PdfReader
from tokenizers import Tokenizer
import os
from kivy.clock import Clock

# ================== 1️⃣ TEXT EXTRACTION ==================
def extract_docx_text(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_pdf_text(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def clean_text(text):
    # Looks for a word ending in a hyphen, a newline, and another word.
    # Replaces "word-\nword" with "wordword"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # 2. Now, normalize all other whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', " ")
    return text.strip()

# ================== 3️⃣ TOKENIZER & EMBEDDINGS ==================
class HuggingFaceTokenizer:
    def __init__(self, tokenizer_json, max_len=256):
        # Load the tokenizer from the file
        self.tokenizer = Tokenizer.from_file(tokenizer_json)
        
        # Configure truncation and padding
        self.tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer.enable_padding(direction='right', length=max_len, pad_id=0, pad_token="[PAD]")

    def encode(self, text, max_len=None): # max_len is now handled by the instance
        # The library handles everything: CLS/SEP tokens, tokenization, padding
        encoded = self.tokenizer.encode(text)

        # Return numpy arrays, as your embedder expects
        ids = np.array(encoded.ids, dtype=np.int64)
        attn = np.array(encoded.attention_mask, dtype=np.int64)
        return ids, attn

class SentenceEmbedder:
    def __init__(self, model_path, tokenizer_json):
        self.session = InferenceSession(model_path, providers=["CPUExecutionProvider"])

        # --- Use the new tokenizer ---
        self.tokenizer = HuggingFaceTokenizer(tokenizer_json) 

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        # The encode method now returns batched numpy arrays directly
        # We can encode all sentences at once!
        all_ids = []
        all_masks = []
        for s in sentences:
            ids, mask = self.tokenizer.encode(s)
            all_ids.append(ids)
            all_masks.append(mask)

        input_ids = np.stack(all_ids)
        attention_mask = np.stack(all_masks)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}

        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        # This pooling logic is correct and remains unchanged
        outputs = self.session.run([self.output_name], feed)[0]
        mask_exp = np.expand_dims(attention_mask, -1)
        sum_emb = np.sum(outputs * mask_exp, axis=1)
        emb = sum_emb / np.clip(mask_exp.sum(axis=1), 1e-9, None)
        return emb.astype(np.float32)

def create_rag_prompt(question, context):
    prompt = f"""Based on the following context, please answer the question:

---
Context:
{context}
---

Question:
{question}
"""
    return prompt

# RAG prompt class
class LocalRag:
    """
    The local RAG calls which will take a PDF or Word (doc/docx) & put it in on device embedding database.
    """

    def __init__(self, model_dir, config_dir) -> None:
        self.model_dir = model_dir
        self.config_dir = config_dir
        self.conn = None
        self.cursor = None

    # ================== SQLITE VECTOR STORE ==================
    def init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS docs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk TEXT,
                embedding TEXT
            )
        """)
        self.conn.commit()

    def insert_chunk(self, chunk, embedding):
        self.cursor.execute("INSERT INTO docs (chunk, embedding) VALUES (?, ?)",
                     (chunk, json.dumps(embedding.tolist())))
        self.conn.commit()

    def start_rag_onnx_sess(self, doc_path, callback=None):
        onnx_path = os.path.join(self.model_dir, "all-MiniLM-L6-V2", "model.onnx")
        tokenizer_path = os.path.join(self.model_dir, "all-MiniLM-L6-V2", "tokenizer.json")
        db_path = os.path.join(self.config_dir, "vector.db")
        if self.conn:
            self.conn.close() # close previous db connection
        if os.path.exists(db_path):
            os.remove(db_path) # remove older rag db
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.init_db()
        self.embedder = SentenceEmbedder(onnx_path, tokenizer_path)
        indx_stat = self.build_index(doc_path)
        self.conn_closer()
        if indx_stat:
            final_stat = True
        else:
            final_stat = False
        if callback:
            Clock.schedule_once(lambda dt: callback(final_stat))
        else:
            return final_stat

    # ================== PIPELINE FUNCTIONS ==================
    def build_index(self, file_path):
        if file_path.endswith(".docx") or file_path.endswith(".doc"):
            text = extract_docx_text(file_path)
        elif file_path.endswith(".pdf"):
            text = extract_pdf_text(file_path)
        text = clean_text(text)
        # chunk properties
        chunk_size = 500
        overlap = 100 # 20%
        step = chunk_size - overlap
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), step)]
        for ch in chunks:
            emb = self.embedder.embed(ch)[0]
            # Normalize the vector to unit length
            norm_emb = emb / np.linalg.norm(emb)
            self.insert_chunk(ch, norm_emb) # Save the normalized embedding
        print(f"Indexed {len(chunks)} chunks from {file_path}")
        if len(chunks) >= 1:
            return True
        else:
            return False

    def search_similar(self, query_emb, top_k=3): # query_emb is now normalized
        rows = self.cursor.execute("SELECT chunk, embedding FROM docs").fetchall()
        results = []
        for chunk, emb_json in rows:
            emb = np.array(json.loads(emb_json)) # This is already normalized
            # --- SIMPLIFIED FORMULA ---
            sim = np.dot(query_emb, emb) 
            results.append((chunk, float(sim)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def query_pipeline(self, question, top_k=3):
        q_emb = self.embedder.embed(question)[0]
        norm_q_emb = q_emb / np.linalg.norm(q_emb)
        top_chunks = self.search_similar(norm_q_emb, top_k)
        context = " ".join(ch for ch, _ in top_chunks)
        context = context.replace("\n", " ")
        return context

    def get_rag_prompt(self, question, callback=None):
        db_path = os.path.join(self.config_dir, "vector.db")
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        context = self.query_pipeline(question)
        final_prompt = create_rag_prompt(question, context)
        self.conn_closer()
        if callback:
            Clock.schedule_once(lambda dt: callback(final_prompt))
        else:
            return final_prompt

    def conn_closer(self):
        if self.conn:
            self.conn.close()
            self.conn = None

