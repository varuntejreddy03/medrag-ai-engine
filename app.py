# app.py
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

import faiss
import numpy as np
import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from huggingface_hub import hf_hub_download
import requests
import shutil

load_dotenv()
LOG = logging.getLogger("medrag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Config (set on Render as env vars)
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None
HF_REPO = os.getenv("HF_REPO", "varuntejreddy/medrag")
FAISS_FILENAME = os.getenv("FAISS_FILENAME", "ddxplus_faiss.index")
KG_FILENAME = os.getenv("KG_FILENAME", "advanced_medrag_kg.json")
DRIVE_INDEX_URL = os.getenv("DRIVE_INDEX_URL", "").strip() or None
DRIVE_KG_URL = os.getenv("DRIVE_KG_URL", "").strip() or None
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Globals
embedder = None
faiss_index = None
kg = None
llm = None

app = Flask(__name__)

def download_hf_or_http(repo_id, filename, token=None):
    """Try hf_hub_download first (cached), then HTTP fallback to public URL."""
    # 1) hf_hub_download (cached)
    try:
        LOG.info("Trying hf_hub_download for %s/%s", repo_id, filename)
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
        return Path(path)
    except Exception as e:
        LOG.warning("hf_hub_download failed: %s", e)

    # 2) public HTTP URL
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    LOG.info("Attempting public HTTP download: %s", url)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code == 200:
            out = CACHE_DIR / filename
            with open(out, "wb") as fh:
                shutil.copyfileobj(resp.raw, fh)
            return out
        LOG.warning("Public HTTP returned status %s", resp.status_code)
    except Exception as e:
        LOG.warning("Public HTTP download failed: %s", e)
    return None

def download_from_drive(url, filename):
    """Download from public Google Drive share (uc?id=...)."""
    if not url:
        return None
    LOG.info("Downloading from Drive URL: %s", url)
    out = CACHE_DIR / filename
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(out, "wb") as fh:
            shutil.copyfileobj(resp.raw, fh)
        return out
    except Exception as e:
        LOG.warning("Drive download failed: %s", e)
        return None

def ensure_files():
    """Ensure index and KG are present locally; return paths."""
    faiss_path = None
    kg_path = None

    # Try Drive URLs first (if provided)
    if DRIVE_INDEX_URL:
        faiss_path = download_from_drive(DRIVE_INDEX_URL, FAISS_FILENAME)
    if DRIVE_KG_URL:
        kg_path = download_from_drive(DRIVE_KG_URL, KG_FILENAME)

    # Else try Hugging Face
    if not faiss_path:
        faiss_path = download_hf_or_http(HF_REPO, FAISS_FILENAME, token=HF_TOKEN)
    if not kg_path:
        kg_path = download_hf_or_http(HF_REPO, KG_FILENAME, token=HF_TOKEN)

    if not faiss_path or not faiss_path.exists():
        raise FileNotFoundError("FAISS index not found. Provide HF_TOKEN/HF_REPO or DRIVE_INDEX_URL.")
    if not kg_path or not kg_path.exists():
        raise FileNotFoundError("KG JSON not found. Provide HF_TOKEN/HF_REPO or DRIVE_KG_URL.")
    return faiss_path, kg_path

def load_components():
    global embedder, faiss_index, kg, llm
    LOG.info("Ensuring data files...")
    faiss_path, kg_path = ensure_files()

    # Load embedding model
    LOG.info("Loading embedder (BAAI/bge-m3 preferred)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda":
            embedder = SentenceTransformer("BAAI/bge-m3", device=device, model_kwargs={"torch_dtype": torch.bfloat16})
        else:
            embedder = SentenceTransformer("BAAI/bge-m3", device=device)
    except Exception as e:
        LOG.warning("Failed to load BGE-M3; falling back to all-MiniLM-L6-v2: %s", e)
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # Load FAISS with memory-mapping
    LOG.info("Loading FAISS index with memory-map...")
    faiss.omp_set_num_threads(2)
    try:
        faiss_index = faiss.read_index(str(faiss_path), faiss.IO_FLAG_MMAP)
    except Exception as e:
        LOG.warning("mmap load failed (%s), trying normal read...", e)
        faiss_index = faiss.read_index(str(faiss_path))
    # if index has nprobe parameter (IVF), set low nprobe to reduce memory during search
    if hasattr(faiss_index, "nprobe"):
        try:
            faiss_index.nprobe = int(os.getenv("FAISS_NPROBE", "4"))
        except:
            pass
    LOG.info("FAISS loaded: dim=%d, ntotal=%d", faiss_index.d, faiss_index.ntotal)

    # Load KG
    LOG.info("Loading knowledge graph...")
    with open(kg_path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    # Load LLM (Mistral or fallback)
    LOG.info("Loading LLM pipeline (may be slow)...")
    try:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (torch.float16 if torch.cuda.is_available() else torch.float32)
        llm = pipeline("text-generation", model=os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"), device_map="auto", torch_dtype=dtype)
    except Exception as e:
        LOG.warning("Failed to load Mistral (maybe no GPU). Falling back to smaller instruct model: %s", e)
        llm = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device_map="auto")

    LOG.info("All components loaded successfully.")

def medrag_reasoning(query: str, k: int = 5):
    # 1. embed
    qv = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)
    # 2. search
    D, I = faiss_index.search(qv, k)
    retrieved = []
    for score, idx in zip(D[0], I[0]):
        retrieved.append({"id": int(idx), "score": float(score)})
    # 3. gather a few KG triplets (simple heuristic using top diseases if present in KG keys)
    diseases = list(kg.keys())[:5]
    kg_triplets = []
    for d in diseases[:3]:
        rels = kg.get(d, {})
        if isinstance(rels, dict):
            for r, ts in rels.items():
                if isinstance(ts, list):
                    for t in ts[:5]:
                        kg_triplets.append(f"{d} -[{r}]-> {t}")
                else:
                    kg_triplets.append(f"{d} -[{r}]-> {ts}")
    # 4. prompt LLM
    cases_text = "\n".join([f"- Case_{r['id']} (score={r['score']:.3f})" for r in retrieved])
    kg_text = "\n".join(kg_triplets[:20])
    prompt = f"""
You are **MedRAG**, an advanced clinical reasoning system and expert in **pain management and internal medicine**.

Your objective:
Given the *new patient case*, the *retrieved similar cases* from the FAISS index, and the *medical knowledge graph context*, you must:
1. Analyze the most relevant symptoms, comorbidities, and pain patterns.
2. Refer to prior cases and the knowledge graph to reason through the diagnosis.
3. Generate evidence-based, structured clinical output.

---

### Clinical Context

**🧑‍⚕️ Patient Query:**
{query}

**📂 Retrieved Similar Cases (Top-{k}):**
{cases_text}

**🧬 Knowledge Graph Context:**
{kg_text}

---

### Diagnostic Rules

- Only consider diagnoses from this controlled set:

{{
acute copd exacerbation infection, bronchiectasis, bronchiolitis, bronchitis, bronchospasm acute asthma exacerbation,
pulmonary embolism, pulmonary neoplasm, spontaneous pneumothorax, urti, viral pharyngitis, whooping cough,
acute laryngitis, acute pulmonary edema, croup, larygospasm, epiglottitis, pneumonia, atrial fibrillation,
myocarditis, pericarditis, psvt, possible nstemi stemi, stable angina, unstable angina, gerd, boerhaave syndrome,
pancreatic neoplasm, scombroid food poisoning, inguinal hernia, tuberculosis, hiv initial infection, ebola, influenza,
chagas, acute otitis media, acute rhinosinusitis, allergic sinusitis, chronic rhinosinusitis, myasthenia gravis,
guillain barre syndrome, cluster headache, acute dystonic reactions, sle, sarcoidosis, anaphylaxis, panic attack,
spontaneous rib fracture, anemia
}}

- You may reference knowledge graph edges such as *has_symptom*, *caused_by*, or *affects* for reasoning.
- Always differentiate conditions with overlapping symptoms or pain locations using logical clinical distinctions.

---

### Output Formatting (strict)

Return your answer in this structured markdown format:

### Diagnoses
1. **Diagnosis**: (select one or few from the given set)
2. **Explanations of diagnose**: Explain how symptoms, evidence, and knowledge graph context support this diagnosis.

### Instructive question
1. **Questions**: Suggest questions the doctor should ask to further distinguish between possible diagnoses.
   Only include aspects like **Pain restriction, Location, or Symptoms**, separated by commas.

### Pain/General Physiotherapist Treatments
1. **Session No.: General Overview**
   - **Specific interventions/treatments**: (manual therapy, mobilization, stretching, etc.)
   - **Goals**: (restore function, reduce pain, etc.)
   - **Exercises**: (specific movement or strengthening routines)
   - **Manual Therapy**: (joint mobilization, trigger-point work)
   - **Techniques**: (soft-tissue release, respiratory training, etc.)

2. **Exercise Recommendations from the Exercise List**:
   - (Provide specific exercises or leave blank if not applicable)

### Pain Psychologist Treatments (if applicable)
1. **Treatment 1**: (cognitive or behavioral strategy)
   - If not applicable, write "Not applicable"

### Pain Medicine Treatments
1. **Medication-based recommendations**: (short evidence-based pharmacological guidance)

### Recommendations for Further Evaluations
1. **Evaluation 1**: (specific diagnostic or imaging test that can confirm/clarify)

---

### Clinical Notes

- Be concise, evidence-based, and medically sound.
- Avoid overdiagnosis.
- Integrate both **retrieved cases** and **knowledge graph** reasoning.
- Prefer high-probability, pain-related, and clinically relevant outputs.
- Format **exactly as markdown**, with bold section headers.

---
Respond **only with the formatted structured markdown**, no explanations outside of the format.
"""

    out = llm(prompt, max_new_tokens=300, do_sample=False, temperature=0.2)
    raw = out[0].get("generated_text", "") if isinstance(out, list) else str(out)
    # attempt to parse JSON substring returned by model
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end != -1:
        try:
            parsed = json.loads(raw[start:end])
            return {**parsed, "similar_cases": retrieved, "kg_triplets": kg_triplets[:20]}
        except Exception:
            pass
    # fallback safe structure
    return {
        "diagnosis": "unknown",
        "differentials": [],
        "tests": [],
        "reasoning": raw.strip(),
        "similar_cases": retrieved,
        "kg_triplets": kg_triplets[:20]
    }

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":"ok" if embedder else "loading",
        "faiss_vectors": faiss_index.ntotal if faiss_index else None,
        "components_loaded": embedder is not None
    })

@app.route("/diagnose", methods=["POST"])
def diagnose():
    if embedder is None:
        return jsonify({"error":"system not ready, components still loading"}), 503
    payload = request.get_json(silent=True) or {}
    query = payload.get("query","").strip()
    if not query:
        return jsonify({"error":"missing query"}), 400
    result = medrag_reasoning(query, k=int(payload.get("k",5)))
    return jsonify(result)

# Load components on startup
LOG.info("Starting MedRAG engine startup...")
try:
    load_components()
except Exception as e:
    LOG.error("Failed to load components: %s", e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
