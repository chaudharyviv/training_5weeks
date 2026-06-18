
"""
Resume Scanner Pro — AI-Powered Candidate Evaluation (Fixed & Improved)
====================================================
Now with Clear "Why Selected/Rejected" Section
"""

import json
import re
from datetime import datetime
from openai import OpenAI
import streamlit as st

# ── Page Config & CSS ───────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Scanner Pro", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .gradient-text { background: linear-gradient(120deg, #155799, #159957); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .score-circle { width: 140px; height: 140px; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 48px; font-weight: 800; box-shadow: 0 8px 30px rgba(0,0,0,0.15); }
    .resume-box { background: #f8f9fa; border-radius: 10px; padding: 16px; max-height: 420px; overflow-y: auto; white-space: pre-wrap; font-size: 13.5px; line-height: 1.65; border: 1px solid #e0e0e0; }
    .decision-box { padding: 20px; border-radius: 12px; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ───────────────────────────────────────────────────────────
if "resume_text" not in st.session_state: st.session_state.resume_text = ""
if "jd_text" not in st.session_state: st.session_state.jd_text = ""
if "last_sample" not in st.session_state: st.session_state.last_sample = None

# ── OpenAI Client ───────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        st.sidebar.warning("⚠️ Add OPENAI_API_KEY in .streamlit/secrets.toml")
        return None

oai = get_client()

# ── Job Descriptions ─────────────────────────────────────────────────
JOB_DESCRIPTIONS = {
    "Linux Engineer": {
        "title": "🐧 Linux Engineer",
        "icon": "🐧",
        "description": """
**Role Summary:**
We are seeking an experienced Linux Engineer to manage and optimize our Linux-based infrastructure. The ideal candidate will have deep expertise in system administration, automation, and troubleshooting.

**Key Responsibilities:**
- Design, implement, and maintain Linux server environments (RHEL, Ubuntu, CentOS)
- Automate system administration tasks using shell scripting and configuration management tools
- Monitor system performance, troubleshoot issues, and ensure high availability
- Implement security best practices and manage system patches
- Collaborate with development teams to support application deployment

**Required Skills & Experience:**
- 5+ years of Linux system administration experience
- Expert knowledge of shell scripting (Bash, Python)
- Experience with configuration management (Ansible, Puppet, or Chef)
- Strong understanding of networking, firewalls, and security
- Experience with virtualization and containerization (Docker, Kubernetes)
- Knowledge of monitoring tools (Nagios, Zabbix, Prometheus)
- Bachelor's degree in Computer Science or related field

**Nice to Have:**
- Red Hat Certified Engineer (RHCE)
- Experience with cloud platforms (AWS, Azure, GCP)
- Knowledge of CI/CD pipelines
"""
    },
    "Software Engineer": {
        "title": "💻 Software Engineer",
        "icon": "💻",
        "description": """
**Role Summary:**
We are looking for a talented Software Engineer to join our development team. The ideal candidate will write clean, efficient, and maintainable code while contributing to all phases of the software development lifecycle.

**Key Responsibilities:**
- Design, develop, and maintain high-quality software applications
- Write clean, testable, and well-documented code following best practices
- Participate in code reviews and contribute to team knowledge sharing
- Collaborate with product managers and designers to define requirements
- Troubleshoot and debug complex software issues

**Required Skills & Experience:**
- 4+ years of professional software development experience
- Strong proficiency in at least two programming languages (Python, Java, JavaScript, Go)
- Experience with version control (Git)
- Knowledge of data structures, algorithms, and object-oriented programming
- Experience with SQL and relational databases
- Understanding of REST APIs and microservices architecture
- Bachelor's degree in Computer Science or related field

**Nice to Have:**
- Experience with cloud services (AWS, Azure, GCP)
- Knowledge of Docker and Kubernetes
- Experience with agile development methodologies
"""
    }
}

# ── Sample Resumes (FULL) ───────────────────────────────────────────────────
SAMPLE_RESUMES = {
    "Strong Linux Engineer": """
JOHN SMITH
Senior Linux Engineer
Email: john.smith@email.com | Phone: (555) 123-4567

PROFESSIONAL SUMMARY
Senior Linux System Administrator with 8 years of experience managing enterprise Linux environments. Expert in automation, security, and infrastructure optimization. Red Hat Certified Engineer with a passion for performance tuning and high-availability systems.

TECHNICAL SKILLS
• Operating Systems: RHEL (6/7/8), Ubuntu, CentOS, Debian
• Scripting: Advanced Bash, Python, Perl
• Configuration Management: Ansible, Puppet
• Virtualization: VMware vSphere, KVM, Docker, Kubernetes
• Monitoring: Nagios, Prometheus, Zabbix, Grafana
• Cloud: AWS (EC2, S3, VPC), Azure
• Databases: PostgreSQL, MySQL

PROFESSIONAL EXPERIENCE

Senior Linux Engineer | TechCorp Inc. | 2018-Present
• Manage 500+ Linux servers across multiple data centers, maintaining 99.99% uptime
• Implemented Ansible automation reducing deployment time by 75%
• Led migration of legacy systems to containerized environments using Docker and Kubernetes
• Designed and implemented security hardening following CIS benchmarks

Linux System Administrator | DataDynamics | 2015-2018
• Administered 200+ RHEL and Ubuntu servers
• Automated system maintenance tasks using Python and Bash scripts
• Reduced mean time to resolution (MTTR) by 40%

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2015

CERTIFICATIONS
• Red Hat Certified Engineer (RHCE) - RHEL 8
• AWS Certified Solutions Architect - Associate
• Certified Kubernetes Administrator (CKA)
""",

    "Weak Linux Engineer": """
DAVID WILSON
IT Administrator
Email: david.w@email.com | Phone: (555) 987-6543

PROFESSIONAL SUMMARY
IT professional with experience in various systems. Recently completed CompTIA Linux+ certification. Looking to transition into a Linux Engineer role.

TECHNICAL SKILLS
• Windows Server, Windows 10
• Basic Linux (Ubuntu, CentOS)
• Microsoft Office Suite
• Active Directory
• Basic networking

PROFESSIONAL EXPERIENCE

IT Support Specialist | Small Business Solutions | 2020-Present
• Provide tier 1 and tier 2 IT support
• Manage Windows workstations and basic Linux servers
• Created user accounts in Active Directory

Junior IT Administrator | Local Government | 2018-2020
• Maintained desktop computers
• Provided phone and email support

EDUCATION
Associate of Applied Science in Information Technology
Community College | 2018

CERTIFICATIONS
• CompTIA Linux+ (2022)
• CompTIA A+
""",

    "Strong Software Engineer": """
SARAH JOHNSON
Senior Software Engineer
Email: sarah.j@email.com | Phone: (555) 234-5678

PROFESSIONAL SUMMARY
Senior Software Engineer with 6 years of experience developing scalable web applications. Expert in Python, Java, and modern web frameworks.

TECHNICAL SKILLS
• Languages: Python, Java, JavaScript, Go
• Frameworks: Django, Spring Boot, React
• Databases: PostgreSQL, MongoDB
• Cloud: AWS, GCP
• DevOps: Docker, Kubernetes, CI/CD

PROFESSIONAL EXPERIENCE

Senior Software Engineer | CloudTech Solutions | 2019-Present
• Led development of microservices architecture serving 1M+ users
• Implemented CI/CD pipeline reducing deployment time significantly
• Mentored junior developers

Software Engineer | Digital Innovations | 2017-2019
• Developed full-stack web applications using Python Django and React

EDUCATION
Master of Science in Computer Science
Stanford University | 2017
""",

    "Weak Software Engineer": """
MARK PATEL
Junior Developer / IT Support
Email: mark.patel@email.com | Phone: (555) 876-5432

PROFESSIONAL SUMMARY
Enthusiastic individual with a passion for technology. Self-taught programmer with some personal project experience.

TECHNICAL SKILLS
• Languages: Basic Python, HTML, CSS
• Tools: Microsoft Office, WordPress
• Basic JavaScript

PROFESSIONAL EXPERIENCE

IT Support Technician | Retail Chain HQ | 2021-Present
• Resolved printer and desktop issues

Data Entry Clerk | Insurance Company | 2019-2021
• Entered policy data into CRM

EDUCATION
Bachelor of Arts in Business Administration
Regional State University | 2019

CERTIFICATIONS
• Udemy — Python for Beginners
"""
}

# ── System Prompt ───────────────────────────────────────────────────────────
RESUME_SCANNER_SYSTEM = """
You are an expert HR screening consultant with 20 years of experience. Your task is to evaluate a candidate's resume against a specific job description using a rigorous scoring system.

**Scoring System (1-10):**
- 10: Exceptional - Exceeds all requirements
- 8-9: Excellent / Very Strong
- 7: Strong
- 5-6: Adequate / Below Average
- 1-4: Weak / Poor

**Decision Framework:**
- 8-10: SELECTED
- 7: SELECTED
- 5-6: MANAGER REVIEW
- 1-4: NOT SELECTED

**Evaluation Categories:**
1. Technical Skills & Knowledge
2. Experience Relevance & Depth
3. Education & Certifications
4. Soft Skills & Communication

Provide response in this exact format:
SCORE SUMMARY:
Overall Score: X/10
Verdict: [SELECTED/MANAGER REVIEW/NOT SELECTED]

CATEGORY SCORES:
- Technical Skills: X/10
- Experience: X/10
- Education & Certifications: X/10
- Soft Skills: X/10

DETAILED JUSTIFICATION:
[Explanation]

STRENGTHS:
• point 1
• point 2

GAPS & CONCERNS:
• gap 1
• gap 2

RECOMMENDATION:
[Final recommendation]
"""

# ── Helper Functions (Improved Parsing) ─────────────────────────────────────
def ask_gpt(system: str, user: str, temperature: float = 0.1):
    if not oai:
        return "❌ OpenAI client not initialized."
    try:
        r = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=1600
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

def parse_evaluation_response(response: str):
    result = {
        "raw": response,
        "overall_score": 0.0,
        "verdict": "ERROR",
        "category_scores": {},
        "strengths": [],
        "gaps": [],
        "recommendation": "No recommendation available."
    }
    
    score_match = re.search(r"Overall Score:\s*([\d.]+)", response, re.I)
    if score_match:
        result["overall_score"] = float(score_match.group(1))
    
    verdict_match = re.search(r"Verdict:\s*(SELECTED|MANAGER REVIEW|NOT SELECTED)", response, re.I)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).upper()
    
    # Category scores
    for match in re.finditer(r"([A-Za-z &\-]+?):\s*([\d.]+)/10", response):
        cat = match.group(1).strip()
        if "Overall" not in cat:
            result["category_scores"][cat] = float(match.group(2))
    
    # Strengths & Gaps
    strengths_sec = re.search(r"STRENGTHS:(.*?)(?:GAPS & CONCERNS:|RECOMMENDATION:|$)", response, re.DOTALL | re.I)
    if strengths_sec:
        result["strengths"] = [s.strip() for s in re.findall(r"[•-]\s*(.+)", strengths_sec.group(1)) if s.strip()]
    
    gaps_sec = re.search(r"GAPS.*?:(.*?)(?:RECOMMENDATION:|$)", response, re.DOTALL | re.I)
    if gaps_sec:
        result["gaps"] = [g.strip() for g in re.findall(r"[•-]\s*(.+)", gaps_sec.group(1)) if g.strip()]
    
    # Recommendation (Key Improvement)
    rec_match = re.search(r"RECOMMENDATION:(.*?)(?=$)", response, re.DOTALL | re.I)
    if rec_match:
        result["recommendation"] = rec_match.group(1).strip()
    
    return result

def get_score_color(score: float):
    if score >= 7: return "#00a65a"
    elif score >= 5: return "#f39c12"
    return "#e74c3c"

def get_verdict_icon(verdict: str):
    return {"SELECTED": "✅", "MANAGER REVIEW": "⚠️", "NOT SELECTED": "❌"}.get(verdict, "❓")

def scan_resume(jd_text: str, resume_text: str, temperature: float = 0.1):
    prompt = f"JOB DESCRIPTION:\n{jd_text}\n\nCANDIDATE RESUME:\n{resume_text}"
    raw_response = ask_gpt(RESUME_SCANNER_SYSTEM, prompt, temperature)
    if "❌ Error" in raw_response:
        return {"error": raw_response}
    
    result = parse_evaluation_response(raw_response)
    result["raw_output"] = raw_response
    return result

# ── Sidebar (Same) ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Resume Scanner Pro")
    st.caption("AI-Powered Candidate Evaluation")
    st.divider()

    st.markdown("### 📋 Job Role")
    role_list = [f"{v['icon']} {k}" for k, v in JOB_DESCRIPTIONS.items()] + ["✏️ Custom JD"]
    selected_role = st.radio("Select role", role_list, index=0, label_visibility="collapsed")

    st.divider()
    st.markdown("### 📄 Candidate Resume")
    resume_option = st.selectbox("Load sample", ["Select..."] + list(SAMPLE_RESUMES.keys()) + ["Custom / Paste"], label_visibility="collapsed")

    if st.button("📥 Load Selected Sample", use_container_width=True):
        if resume_option in SAMPLE_RESUMES:
            st.session_state.resume_text = SAMPLE_RESUMES[resume_option]
            st.session_state.last_sample = resume_option
            st.success(f"✅ Loaded: {resume_option}")
            st.rerun()

    st.divider()
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
    st.divider()
    evaluate_btn = st.button("🔍 Evaluate Candidate", type="primary", use_container_width=True)

# ── Main UI ─────────────────────────────────────────────────────────────────
st.markdown('<h1 style="text-align:center;"><span class="gradient-text">🎯 Resume Scanner Pro</span></h1>', unsafe_allow_html=True)
st.caption("Now with Clear Decision Reasoning")
st.divider()

col_jd, col_res = st.columns(2)

with col_jd:
    st.markdown("#### 📋 Job Description")
    if "Custom" in selected_role:
        jd_text = st.text_area("Paste Job Description", value=st.session_state.jd_text, height=420)
        st.session_state.jd_text = jd_text
    else:
        role_name = selected_role.split(" ", 1)[1] if " " in selected_role else selected_role
        jd = JOB_DESCRIPTIONS.get(role_name, {})
        jd_text = jd.get("description", "")
        st.caption(f"**{jd.get('title', role_name)}**")
        st.markdown(f'<div class="resume-box">{jd_text}</div>', unsafe_allow_html=True)

with col_res:
    st.markdown("#### 📄 Candidate Resume")
    if st.session_state.get("last_sample"):
        st.caption(f"**Loaded:** {st.session_state.last_sample}")
    resume_text = st.text_area("Resume Content", value=st.session_state.resume_text, height=420)
    st.session_state.resume_text = resume_text

st.divider()

# ── Results ─────────────────────────────────────────────────────────────────
st.markdown("### 📊 Evaluation Results")

if evaluate_btn:
    if not jd_text.strip() or not resume_text.strip():
        st.error("Please provide both Job Description and Resume")
    else:
        with st.spinner("🔍 Analyzing with AI..."):
            result = scan_resume(jd_text, resume_text, temperature)

        if "error" in result:
            st.error(result["error"])
        else:
            score = result.get("overall_score", 0)
            verdict = result.get("verdict", "UNKNOWN")
            score_color = get_score_color(score)

            # Score & Verdict
            col_score, col_verdict = st.columns([1, 2])
            with col_score:
                st.markdown(f"""
                <div style="text-align:center;">
                    <div class="score-circle" style="background: conic-gradient({score_color} 0% {score*10}%, #f0f0f0 {score*10}% 100%); border: 6px solid {score_color}; color: {score_color};">{score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_verdict:
                icon = get_verdict_icon(verdict)
                if verdict == "SELECTED":
                    st.success(f"### {icon} SELECTED")
                elif verdict == "MANAGER REVIEW":
                    st.warning(f"### {icon} MANAGER REVIEW")
                else:
                    st.error(f"### {icon} NOT SELECTED")

            st.divider()

            # ── NEW: Clear Decision Reasoning ─────────────────────────────
            st.markdown("#### 🎯 Why This Decision?")
            with st.container():
                decision_color = "#d4edda" if verdict == "SELECTED" else "#f8d7da" if verdict == "NOT SELECTED" else "#fff3cd"
                st.markdown(f"""
                <div class="decision-box" style="background:{decision_color}; border-left: 6px solid {score_color};">
                    {result.get('recommendation', 'No specific recommendation generated.')}
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Category Scores, Strengths, Gaps
            if result.get("category_scores"):
                st.markdown("#### 📊 Category Scores")
                cols = st.columns(len(result["category_scores"]))
                for i, (cat, sc) in enumerate(result["category_scores"].items()):
                    with cols[i]:
                        st.metric(cat, f"{sc:.1f}/10")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 💪 Strengths")
                for s in result.get("strengths", []):
                    st.success(f"• {s}")
            with col2:
                st.markdown("#### ⚠️ Gaps & Concerns")
                for g in result.get("gaps", []):
                    st.error(f"• {g}")

            st.divider()
            with st.expander("📝 Full Detailed Justification", expanded=True):
                st.markdown(result.get("raw_output", "No details available."))

            # Download
            export_data = {**result, "timestamp": datetime.now().isoformat(), "job_title": selected_role}
            st.download_button("📥 Download Full Report (JSON)", data=json.dumps(export_data, indent=2), file_name=f"resume_evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json")

else:
    st.info("👈 Select role & resume from sidebar, then click **Evaluate Candidate**")

st.divider()
st.markdown("<div style='text-align:center;color:#777;'>Resume Scanner Pro • Clear Decision Reasoning </div>", unsafe_allow_html=True)
