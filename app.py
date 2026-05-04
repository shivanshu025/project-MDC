import streamlit as st
import re
import time

# ------------------ Helper Functions ------------------

def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text).lower()


def extract_skills(text, skill_list):
    text = clean_text(text)
    return [skill for skill in skill_list if skill.lower() in text]


def calculate_score(jd_skills, resume_skills):
    if len(jd_skills) == 0:
        return 0
    matched = len(set(jd_skills) & set(resume_skills))
    return int((matched / len(jd_skills)) * 100)


def get_label(score):
    if score >= 80:
        return "Highly Suitable"
    elif score >= 60:
        return "Good Fit"
    elif score >= 40:
        return "Average"
    else:
        return "Low Fit"

# ------------------ UI ------------------

st.set_page_config(page_title="CV Reader", layout="wide")

st.markdown("""
    <style>
    body { background-color: #0e1117; }
    .title { text-align:center; font-size:40px; font-weight:bold; color:white; }
    .subtitle { text-align:center; color:gray; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🤖 AI Resume Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Hiring Assistant</div>", unsafe_allow_html=True)

skills_db = ["Python", "Java", "SQL", "Machine Learning", "Communication", "HTML", "CSS"]

st.write("---")

jd_input = st.text_area("📄 Enter Job Description")
uploaded_files = st.file_uploader("📂 Upload Multiple Resumes (txt)", accept_multiple_files=True, type=["txt"])

col1, col2 = st.columns(2)
run_demo = col1.button("🧪 Run Demo")
analyze = col2.button("▶️ Analyze")

results = []

# ------------------ Demo Mode ------------------
if run_demo:
    jd_input = "Looking for Python, SQL, Machine Learning, Communication"

    resumes = {
        "Rahul Sharma": "Python SQL Machine Learning",
        "Aman Verma": "Python Communication",
        "Priya Singh": "Java HTML CSS"
    }

    jd_skills = extract_skills(jd_input, skills_db)

    with st.spinner("Analyzing resumes..."):
        time.sleep(2)

        for name, text in resumes.items():
            res_skills = extract_skills(text, skills_db)
            score = calculate_score(jd_skills, res_skills)
            results.append((name, score))

# ------------------ Uploaded Files Mode ------------------
if analyze and uploaded_files:
    jd_skills = extract_skills(jd_input, skills_db)

    with st.spinner("Analyzing resumes..."):
        time.sleep(2)

        for file in uploaded_files:
            content = file.read().decode("utf-8")
            res_skills = extract_skills(content, skills_db)
            score = calculate_score(jd_skills, res_skills)
            results.append((file.name, score))

# ------------------ Display Results ------------------

if results:
    results = sorted(results, key=lambda x: x[1], reverse=True)

    st.write("---")
    st.subheader("🏆 Candidate Rankings")

    for i, (name, score) in enumerate(results, start=1):
        st.markdown(f"### #{i} {name}")
        st.progress(score / 100)
        st.write(f"Score: {score}%")
        st.write(f"Status: {get_label(score)}")
        st.write("---")

# Footer
st.markdown("<div style='text-align:center; color:gray;'>Developed for AI Project</div>", unsafe_allow_html=True)
