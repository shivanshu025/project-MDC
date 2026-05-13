"""
TalentRank AI — Python Backend v2.0
────────────────────────────────────────────
AI Engine:  TF-IDF Vectorization + Cosine Similarity  (scikit-learn)
            Hybrid scored with: skill overlap ratio, experience, education
No paid API. No GPU. Runs on any machine, fully offline.

Run:   python app.py
Open:  http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import io

try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

app = Flask(__name__, static_folder="static")

# ─────────────────────────────────────────────
#  TECH SKILL DICTIONARY  (90+ skills)
# ─────────────────────────────────────────────
SKILL_DICT = [
    # Languages
    "python","javascript","typescript","java","golang","go","rust","c++","c#","kotlin",
    "swift","php","ruby","scala","r","matlab","bash","shell","sql",
    # Frontend
    "react","vue","angular","svelte","next.js","nuxt","html","css","tailwind","sass",
    "webpack","vite","redux","mobx",
    # Backend
    "node.js","django","flask","fastapi","spring","rails","express","nestjs",
    "graphql","rest api","grpc","websocket","microservices","serverless",
    # Data / AI / ML
    "machine learning","deep learning","nlp","computer vision",
    "pytorch","tensorflow","keras","scikit-learn",
    "openai","anthropic","gemini","llm","rag","langchain","hugging face",
    "transformers","gpt","bert","embeddings","vector database",
    "pandas","numpy","scipy","matplotlib","seaborn",
    "spark","airflow","kafka","dbt","etl","data pipeline",
    "pgvector","pinecone","weaviate","chroma",
    # Databases
    "postgresql","mysql","mongodb","redis","sqlite","dynamodb",
    "firebase","supabase","elasticsearch",
    # Cloud / Infra
    "aws","azure","gcp","google cloud","docker","kubernetes","terraform",
    "ci/cd","devops","linux","lambda","s3","ec2","ansible","cloudformation",
    # Tools / Process
    "git","github","agile","scrum","product management","startup","mobile",
    "react native","flutter","ios","android",
]

# ─────────────────────────────────────────────
#  DISPLAY FORMATTING
# ─────────────────────────────────────────────
_CAPS = {"llm","nlp","aws","gcp","api","sql","css","html","gpu","rest",
         "grpc","etl","rag","ios","mit","cs","ux","ui","s3","ec2"}
_SPECIAL = {
    "node.js":"Node.js","next.js":"Next.js","postgresql":"PostgreSQL",
    "ci/cd":"CI/CD","scikit-learn":"scikit-learn",
    "google cloud":"Google Cloud","react native":"React Native",
    "machine learning":"Machine Learning","deep learning":"Deep Learning",
    "computer vision":"Computer Vision","vector database":"Vector Database",
    "data pipeline":"Data Pipeline","hugging face":"Hugging Face",
    "openai":"OpenAI","anthropic":"Anthropic","pgvector":"pgvector",
    "typescript":"TypeScript","javascript":"JavaScript",
    "react":"React","vue":"Vue","python":"Python","golang":"Go","rust":"Rust",
    "c++":"C++","c#":"C#","kotlin":"Kotlin","swift":"Swift","php":"PHP",
    "ruby":"Ruby","scala":"Scala","angular":"Angular","svelte":"Svelte",
    "django":"Django","flask":"Flask","fastapi":"FastAPI","redis":"Redis",
    "mongodb":"MongoDB","firebase":"Firebase","supabase":"Supabase",
    "docker":"Docker","kubernetes":"Kubernetes","terraform":"Terraform",
    "github":"GitHub","linux":"Linux","pandas":"Pandas","numpy":"NumPy",
    "pytorch":"PyTorch","tensorflow":"TensorFlow","keras":"Keras",
    "spark":"Apache Spark","kafka":"Kafka","airflow":"Apache Airflow",
    "pinecone":"Pinecone","weaviate":"Weaviate","langchain":"LangChain",
    "gemini":"Gemini","gpt":"GPT","bert":"BERT","startup":"Startup",
    "aws":"AWS","azure":"Azure","gcp":"GCP","grpc":"gRPC","graphql":"GraphQL",
    "rest api":"REST API","websocket":"WebSocket","microservices":"Microservices",
    "serverless":"Serverless","mobile":"Mobile","flutter":"Flutter",
    "embeddings":"Embeddings","chroma":"Chroma","dbt":"dbt",
}

def fmt_skill(s: str) -> str:
    if s in _SPECIAL: return _SPECIAL[s]
    if s in _CAPS:    return s.upper()
    return s.title()

# ─────────────────────────────────────────────
#  TEXT UTILITIES
# ─────────────────────────────────────────────
def extract_skills(text: str) -> list:
    tl = text.lower()
    seen, found = set(), []
    for skill in SKILL_DICT:
        if skill in seen: continue
        if re.search(r'\b' + re.escape(skill) + r'\b', tl):
            seen.add(skill)
            found.append(skill)
    return found

def extract_years(text: str) -> int:
    m = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)', text, re.I)
    return max([int(x) for x in m], default=0)

def detect_seniority(text: str):
    t = text.lower()
    if re.search(r'\b(staff|principal|distinguished|vp\b|cto\b|1[0-9]\+?\s*years?)', t): return "Staff", 4
    if re.search(r'\b(senior|sr\.?\b|lead|[7-9]\+?\s*years?)', t):   return "Senior", 3
    if re.search(r'\b(mid.?level|intermediate|[4-6]\+?\s*years?)', t): return "Mid", 2
    if re.search(r'\b(junior|jr\.?\b|entry.?level|associate|[0-3]\+?\s*years?)', t): return "Junior", 1
    return "Mid", 2

# ─────────────────────────────────────────────
#  AI CORE — TF-IDF + COSINE SIMILARITY
# ─────────────────────────────────────────────
def tfidf_similarity(doc_a: str, doc_b: str) -> float:
    """
    Core AI model.
    TF-IDF converts each document into a high-dimensional weighted
    vector, then cosine similarity measures the angle between them.
    Score: 0.0 (no overlap) to 1.0 (identical).
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        min_df=1,
        max_features=8000,
        sublinear_tf=True,
    )
    try:
        mat = vectorizer.fit_transform([doc_a, doc_b])
        return float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0

# ─────────────────────────────────────────────
#  SUB-SCORES
# ─────────────────────────────────────────────
def compute_skills_score(sim: float, jd_skills: list, resume_skills: list) -> int:
    """
    Hybrid: 60% TF-IDF cosine sim + 40% explicit skill-overlap ratio.
    TF-IDF catches paraphrasing; skill-overlap catches exact keywords.
    """
    jd_set  = set(jd_skills)
    res_set = set(resume_skills)
    overlap = len(jd_set & res_set) / max(len(jd_set), 1)
    hybrid  = sim * 0.60 + overlap * 0.40
    return min(100, int(hybrid * 140))

def compute_experience_score(jd: str, resume: str) -> int:
    jd_yrs  = extract_years(jd) or 3
    res_yrs = extract_years(resume)
    _, jd_lvl  = detect_seniority(jd)
    _, res_lvl = detect_seniority(resume)

    yrs_sc = min(100, int((res_yrs / max(jd_yrs, 1)) * 100))
    lvl_sc = 100 if res_lvl >= jd_lvl else int((res_lvl / max(jd_lvl, 1)) * 100)

    bonus = 0
    jl, rl = jd.lower(), resume.lower()
    if 'startup' in jl and 'startup' in rl:                                   bonus += 8
    if ('0 to 1' in jl or '0→1' in jl) and \
       ('founder' in rl or '0→1' in rl or '0 to 1' in rl):                   bonus += 8
    if 'product' in jl and 'product' in rl:                                   bonus += 4

    return min(100, int(yrs_sc * 0.5 + lvl_sc * 0.5) + bonus)

def compute_education_score(resume: str) -> int:
    r = resume.lower()
    if re.search(r'\b(ph\.?d|doctorate)\b', r):                    return 100
    if re.search(r'\b(m\.?s|m\.?eng|m\.?tech|masters?|mba)\b', r): return 90
    if re.search(r'\b(b\.?s|b\.?eng|b\.?tech|bachelors?)\b', r):   return 80
    if re.search(r'\b(associate|diploma|bootcamp)\b', r):           return 65
    if re.search(r'\b(self.?taught|autodidact)\b', r):              return 62
    return 68

# ─────────────────────────────────────────────
#  NARRATIVE GENERATORS
# ─────────────────────────────────────────────
def gen_strengths(resume: str, matched: list, yrs: int, jd: str) -> list:
    r, j = resume.lower(), jd.lower()
    out  = []
    if yrs >= 5:   out.append(f"{yrs}+ years of hands-on engineering experience")
    elif yrs > 0:  out.append(f"{yrs} years of relevant development experience")
    top = [fmt_skill(s) for s in matched[:3]]
    if top: out.append(f"Proven in: {', '.join(top)}")
    if re.search(r'\b(founder|cto|startup|0→1)\b', r):
        out.append("Startup & 0-to-1 product-building experience")
    if re.search(r'\b(lead|architect|principal|staff)\b', r):
        out.append("Technical leadership and system design experience")
    if re.search(r'\b(shipped|launched|deployed|production|scaled)\b', r):
        out.append("Strong track record of shipping to production")
    if ('llm' in j or 'ai' in j) and re.search(r'\b(openai|anthropic|gemini|llm|gpt|claude)\b', r):
        out.append("Direct LLM / AI API integration experience")
    if re.search(r'\b(open.?source|github\.com)\b', r):
        out.append("Active open-source contributor")
    if len(out) < 2: out.append("Broad and adaptable engineering skill set")
    return out[:4]

def gen_gaps(missing: list, resume: str, jd: str) -> list:
    r, j = resume.lower(), jd.lower()
    gaps = []
    if len(missing) > 3:
        top = [fmt_skill(s) for s in missing[:3]]
        gaps.append(f"Missing {len(missing)} required skills (e.g. {', '.join(top)})")
    elif missing:
        gaps.append(f"Lacks: {', '.join(fmt_skill(s) for s in missing)}")
    if 'senior' in j and not re.search(r'\b(senior|lead|staff|architect)\b', r):
        gaps.append("May not meet the required seniority level")
    if ('startup' in j or '0 to 1' in j) and \
       not re.search(r'\b(startup|founder|early.stage)\b', r):
        gaps.append("No clear startup or 0-to-1 experience")
    if ('llm' in j or 'openai' in j) and \
       not re.search(r'\b(openai|anthropic|gemini|llm|gpt)\b', r):
        gaps.append("Limited LLM / AI API experience")
    if not gaps: gaps.append("Resume coverage looks solid for this role")
    return gaps[:3]

def gen_summary(name: str, rec: str, matched: list, missing: list, yrs: int) -> str:
    first   = name.split()[0] if name else "This candidate"
    n_match = len(matched)
    n_miss  = len(missing)
    if rec == "Strong Match":
        return (f"{first} is an excellent fit — aligns on {n_match} key skills"
                f"{' with ' + str(yrs) + '+ years of experience' if yrs else ''}.")
    if rec == "Good Match":
        gap = f", though gaps remain in {', '.join(fmt_skill(s) for s in missing[:2])}" if missing else ""
        return f"{first} covers most requirements ({n_match} matched skills){gap}. Worth a strong look."
    if rec == "Possible Match":
        return (f"{first} shows relevant promise but is missing {n_miss} required "
                f"skill{'s' if n_miss != 1 else ''}. Could work with some upskilling.")
    return (f"{first} currently lacks several key requirements and would need "
            "significant development to be a strong fit for this role.")

# ─────────────────────────────────────────────
#  API ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/rank", methods=["POST"])
def rank():
    body       = request.get_json(force=True, silent=True) or {}
    jd         = body.get("job_description", "").strip()
    candidates = body.get("candidates", [])

    if not jd:
        return jsonify({"error": "job_description is required"}), 400
    if not candidates:
        return jsonify({"error": "candidates list is required"}), 400

    jd_skills = extract_skills(jd)
    results   = []

    for cand in candidates:
        name   = (cand.get("name")   or "Candidate").strip()
        resume = (cand.get("resume") or "").strip()
        if not resume:
            continue

        # ── AI PIPELINE ──
        sim        = tfidf_similarity(jd, resume)
        res_skills = extract_skills(resume)
        matched    = sorted(set(jd_skills) & set(res_skills))
        missing    = sorted(set(jd_skills) - set(res_skills))

        skills_sc  = compute_skills_score(sim, jd_skills, res_skills)
        exp_sc     = compute_experience_score(jd, resume)
        edu_sc     = compute_education_score(resume)

        # Weighted final: skills 40% | experience 40% | education 20%
        overall = min(100, int(skills_sc * 0.40 + exp_sc * 0.40 + edu_sc * 0.20))

        if   overall >= 78: rec = "Strong Match"
        elif overall >= 60: rec = "Good Match"
        elif overall >= 42: rec = "Possible Match"
        else:               rec = "Weak Match"

        yrs = extract_years(resume)
        results.append({
            "candidate_name":     name,
            "overall_score":      overall,
            "skills_score":       skills_sc,
            "experience_score":   exp_sc,
            "education_score":    edu_sc,
            "tfidf_similarity":   round(sim * 100, 1),
            "matched_skills":     [fmt_skill(s) for s in matched],
            "missing_skills":     [fmt_skill(s) for s in missing],
            "strengths":          gen_strengths(resume, matched, yrs, jd),
            "gaps":               gen_gaps(missing, resume, jd),
            "summary":            gen_summary(name, rec, matched, missing, yrs),
            "recommendation":     rec,
        })

    results.sort(key=lambda x: x["overall_score"], reverse=True)
    return jsonify({
        "rankings":            results,
        "jd_skills_extracted": [fmt_skill(s) for s in jd_skills],
        "model":               "TF-IDF + Cosine Similarity (scikit-learn)",
    })


@app.route("/api/parse-pdfs", methods=["POST"])
def parse_pdfs():
    """
    Accept multiple PDF files via multipart/form-data.
    Returns a list of {name, text} objects extracted from each PDF.
    Requires: PyMuPDF  (pip install PyMuPDF)
    """
    if not PDF_SUPPORT:
        return jsonify({"error": "PyMuPDF not installed. Run: pip install PyMuPDF"}), 501

    files = request.files.getlist("resumes")
    if not files:
        return jsonify({"error": "No files uploaded. Send PDF files under the field name 'resumes'."}), 400

    results = []
    errors  = []

    for f in files:
        filename = f.filename or "unknown.pdf"
        # Derive a candidate name from the filename (strip extension, replace _ / - with space)
        raw_name = re.sub(r'\.pdf$', '', filename, flags=re.I)
        candidate_name = re.sub(r'[_\-]+', ' ', raw_name).strip().title()

        try:
            pdf_bytes = f.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text("text"))
            doc.close()
            full_text = "\n".join(pages_text).strip()

            if not full_text:
                errors.append(f"{filename}: no readable text found (may be a scanned image PDF)")
                continue

            results.append({
                "filename": filename,
                "name":     candidate_name,
                "text":     full_text,
                "pages":    len(pages_text),
            })
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")

    return jsonify({
        "parsed":  results,
        "errors":  errors,
        "total":   len(results),
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "TF-IDF + Cosine Similarity (scikit-learn)"})


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║        🧠  TalentRank AI  v2.1           ║")
    print("  ╠══════════════════════════════════════════╣")
    print("  ║  AI Engine : TF-IDF + Cosine Similarity  ║")
    print("  ║  Library   : scikit-learn + Flask         ║")
    print("  ║  📄 PDF Upload : PyMuPDF                  ║")
    print("  ║  ✅ Free   ✅ No API key  ✅ Offline      ║")
    print("  ╚══════════════════════════════════════════╝")
    print()
    print("  → Open http://localhost:5000")
    print()
    app.run(debug=False, port=5000, host="0.0.0.0")

