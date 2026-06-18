"""
Resume Scanner Pro — AI-Powered Candidate Evaluation
====================================================
Evaluate candidates against job descriptions using GPT-4o-mini
with a sophisticated 1-10 scoring system.

Run:
    pip install streamlit openai anthropic python-dotenv
    streamlit run w02_app.py
"""

import os
import json
import re
from datetime import datetime
from openai import OpenAI
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Scanner Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for Eye-Catching Design ──────────────────────────────────────
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stCard, .metric-card {
        background: white; border-radius: 20px; padding: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    .gradient-text {
        background: linear-gradient(120deg, #155799, #159957);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .score-circle {
        width: 130px; height: 130px; border-radius: 50%; margin: 0 auto;
        display: flex; align-items: center; justify-content: center;
        font-size: 42px; font-weight: 800; box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    .score-circle:hover { transform: scale(1.08); }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: 600; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        return None

oai = get_client()

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
- Bachelor's degree in Computer Science or related field (or equivalent experience)

**Nice to Have:**
- Red Hat Certified Engineer (RHCE) or similar certification
- Experience with cloud platforms (AWS, Azure, GCP)
- Knowledge of CI/CD pipelines
- Experience with database administration

**Soft Skills:**
- Strong problem-solving and analytical abilities
- Excellent documentation skills
- Ability to work in a 24/7 on-call rotation
- Good communication skills for team collaboration
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
- Contribute to system architecture and technical design discussions

**Required Skills & Experience:**
- 4+ years of professional software development experience
- Strong proficiency in at least two programming languages (Python, Java, C++, JavaScript, or Go)
- Experience with version control (Git)
- Knowledge of data structures, algorithms, and object-oriented programming
- Experience with SQL and relational databases
- Understanding of REST APIs and microservices architecture
- Bachelor's degree in Computer Science, Engineering, or related field

**Nice to Have:**
- Experience with cloud services (AWS, Azure, GCP)
- Knowledge of Docker and containerization
- Experience with agile development methodologies
- Contributions to open-source projects

**Soft Skills:**
- Strong analytical and problem-solving skills
- Excellent communication and collaboration abilities
- Self-motivated with ability to work independently
- Continuous learning mindset
"""
    }
}

# ── Default Weights  ───────────────────────────────────────
DEFAULT_WEIGHTS = {
    "Technical Skills": 40,
    "Experience": 25,
    "Education & Certifications": 20,
    "Soft Skills": 15
}

# ── Sample Resumes ──────────────────────────────────────────────────────────
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
• Networking: TCP/IP, DNS, DHCP, Firewalls, Load Balancers

PROFESSIONAL EXPERIENCE

Senior Linux Engineer | TechCorp Inc. | 2018-Present
• Manage 500+ Linux servers across multiple data centers, maintaining 99.99% uptime
• Implemented Ansible automation reducing deployment time by 75%
• Led migration of legacy systems to containerized environments using Docker and Kubernetes
• Designed and implemented security hardening following CIS benchmarks
• Created comprehensive monitoring solution using Prometheus and Grafana
• Mentored junior team members on Linux administration best practices

Linux System Administrator | DataDynamics | 2015-2018
• Administered 200+ RHEL and Ubuntu servers in production environment
• Automated system maintenance tasks using Python and Bash scripts
• Implemented backup and disaster recovery procedures
• Managed Apache, Nginx, and Tomcat web servers
• Reduced mean time to resolution (MTTR) by 40% through improved monitoring

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2015
GPA: 3.8/4.0

CERTIFICATIONS
• Red Hat Certified Engineer (RHCE) - RHEL 8
• AWS Certified Solutions Architect - Associate
• Certified Kubernetes Administrator (CKA)

ACHIEVEMENTS
• Published article on "High-Availability Linux Clusters" in SysAdmin Magazine
• Presented at LinuxCon 2022 on automation best practices
• Awarded "Employee of the Year" 2021 at TechCorp Inc.
""",

    "Weak Linux Engineer": """
DAVID WILSON
IT Administrator
Email: david.w@email.com | Phone: (555) 987-6543

PROFESSIONAL SUMMARY
IT professional with experience in various systems. Recently completed CompTIA Linux+ certification. Looking to transition into a Linux Engineer role.

TECHNICAL SKILLS
• Windows Server 2012/2016, Windows 10
• Basic Linux (Ubuntu, CentOS)
• Microsoft Office Suite
• Active Directory
• Basic networking
• Help desk support

PROFESSIONAL EXPERIENCE

IT Support Specialist | Small Business Solutions | 2020-Present
• Provide tier 1 and tier 2 IT support to 50+ users
• Manage Windows workstations and basic network troubleshooting
• Set up and maintain simple Linux servers for internal use
• Created user accounts and managed permissions in Active Directory
• Documented IT procedures and created knowledge base articles

Junior IT Administrator | Local Government | 2018-2020
• Maintained desktop computers and peripheral devices
• Assisted with hardware upgrades and replacements
• Managed inventory of IT equipment
• Provided phone and email support to end users

EDUCATION
Associate of Applied Science in Information Technology
Community College | 2018

CERTIFICATIONS
• CompTIA Linux+ (2022)
• CompTIA A+
• CompTIA Network+

PROJECTS
• Set up Ubuntu server for file sharing at home
• Configured a basic web server using Apache
""",

    "Strong Software Engineer": """
SARAH JOHNSON
Senior Software Engineer
Email: sarah.j@email.com | Phone: (555) 234-5678

PROFESSIONAL SUMMARY
Senior Software Engineer with 6 years of experience developing scalable web applications. Expert in Python, Java, and modern web frameworks. Passionate about clean code, system architecture, and technical mentorship.

TECHNICAL SKILLS
• Languages: Python, Java, JavaScript, Go
• Frameworks: Django, Spring Boot, React, Node.js
• Databases: PostgreSQL, MongoDB, Redis
• Cloud: AWS (EC2, Lambda, RDS, S3), GCP
• DevOps: Docker, Kubernetes, CI/CD (Jenkins, GitLab CI)
• Version Control: Git, GitHub, GitLab
• Testing: PyTest, JUnit, Selenium
• Architecture: REST APIs, Microservices, Event-Driven Architecture

PROFESSIONAL EXPERIENCE

Senior Software Engineer | CloudTech Solutions | 2019-Present
• Led development of microservices architecture serving 1M+ users
• Designed and implemented RESTful APIs with 99.9% uptime
• Reduced average response time from 200ms to 45ms
• Mentored 5 junior developers through code reviews and pair programming
• Implemented CI/CD pipeline reducing deployment time from 2 hours to 15 minutes
• Achieved 85% test coverage across all services

Software Engineer | Digital Innovations | 2017-2019
• Developed full-stack web applications using Python Django and React
• Collaborated with product team to define and implement features
• Built data processing pipelines for analytics reporting
• Participated in sprint planning and agile ceremonies

EDUCATION
Master of Science in Computer Science
Stanford University | 2017
GPA: 3.9/4.0

Bachelor of Science in Software Engineering
University of California, Berkeley | 2015
GPA: 3.7/4.0

CONTRIBUTIONS
• Open-source contributor to Django and React
• Published 3 technical articles on Medium about microservices
• Speaker at PyCon 2022
""",

"Weak Software Engineer": """
MARK PATEL
Junior Developer / IT Support
Email: mark.patel@email.com | Phone: (555) 876-5432

PROFESSIONAL SUMMARY
Enthusiastic individual with a passion for technology looking to transition into software development. Self-taught programmer with some personal project experience. Recently completed an online bootcamp in web development.

TECHNICAL SKILLS
• Languages: Basic Python (beginner), HTML, CSS
• Tools: Microsoft Office, Google Workspace
• Basic understanding of JavaScript
• Familiar with WordPress

PROFESSIONAL EXPERIENCE

IT Support Technician | Retail Chain HQ | 2021-Present
• Resolved printer, network, and desktop issues for 80+ office staff
• Reset passwords and managed basic user accounts in Active Directory
• Assisted in setting up new employee workstations

Data Entry Clerk | Insurance Company | 2019-2021
• Entered policy data into internal CRM system
• Generated weekly Excel reports for the operations team
• Coordinated with field agents via email

EDUCATION
Bachelor of Arts in Business Administration
Regional State University | 2019
GPA: 2.6/4.0

CERTIFICATIONS
• Udemy — "Python for Beginners" (2023, online course)
• Google IT Support Professional Certificate (2022)

PROJECTS
• Personal blog built with WordPress
• A basic to-do list app in Python using print() statements (GitHub: 1 commit)
""",
}

# ── System Prompts ──────────────────────────────────────────────────────────
RESUME_SCANNER_SYSTEM = """
You are an expert HR screening consultant with 20 years of experience. Your task is to evaluate a candidate's resume against a specific job description using a rigorous scoring system.

**Your Role:**
- Analyze the resume in depth, comparing it to the job description requirements
- Assign scores (1-10) for each evaluation category
- Calculate a weighted total score
- Make a clear recommendation based on the score

**Scoring System (1-10):**
- 10: Exceptional - Exceeds all requirements
- 9: Excellent - Meets all requirements with some extras
- 8: Very Strong - Meets all core requirements
- 7: Strong - Meets most requirements
- 6: Adequate - Meets some requirements
- 5: Below Average - Meets few requirements
- 4: Weak - Significant gaps
- 3: Very Weak - Major deficiencies
- 2: Poor - Mostly not meeting requirements
- 1: Very Poor - Barely meets any requirements

**Decision Framework:**
- Score 8-10: SELECTED - Highly recommended for interview
- Score 7: SELECTED - Recommended for interview
- Score 5-6: MANAGER REVIEW - Potential fit, needs discussion
- Score 1-4: NOT SELECTED - Does not meet requirements

**Evaluation Categories (with typical weights for tech roles):**
1. Technical Skills & Knowledge (30-45%): Core skills matching JD
2. Experience Relevance & Depth (25-35%): Years and quality of experience
3. Education & Certifications (15-20%): Degrees, certifications, training
4. Soft Skills & Communication (10-15%): Leadership, teamwork, communication

**Format Your Response:**
Provide a structured evaluation with:

SCORE SUMMARY:
Overall Score: X/10
Verdict: [SELECTED/MANAGER REVIEW/NOT SELECTED]

CATEGORY SCORES:
- Technical Skills: X/10
- Experience: X/10
- Education & Certifications: X/10
- Soft Skills: X/10

DETAILED JUSTIFICATION:
[2-3 paragraphs explaining each category score with specific references to the resume]

STRENGTHS:
• Specific strength 1
• Specific strength 2

GAPS & CONCERNS:
• Specific gap 1
• Specific gap 2

RECOMMENDATION:
[Final recommendation with clear rationale - 2-3 sentences]

Be fair, objective, and evidence-based. Reference specific points from the resume in your justification.
"""
# ── Helper Functions ──────────────────────────────────────────────────────
def ask_gpt(system: str, user: str, temperature: float = 0.1):
    try:
        r = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=1200
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"

def parse_evaluation_response(response: str):
    result = {"raw": response, "overall_score": 0.0, "verdict": "ERROR", "category_scores": {}, "strengths": [], "gaps": []}
    
    score_match = re.search(r"Overall Score:\s*([\d.]+)", response, re.I)
    if score_match:
        result["overall_score"] = float(score_match.group(1))
    
    verdict_match = re.search(r"Verdict:\s*(SELECTED|MANAGER REVIEW|NOT SELECTED)", response, re.I)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).upper()
    
    for match in re.finditer(r"([A-Za-z &\-]+?):\s*([\d.]+)/10", response):
        cat = match.group(1).strip()
        if "Overall" not in cat:
            result["category_scores"][cat] = float(match.group(2))
    
    strengths_sec = re.search(r"STRENGTHS:(.*?)(?:GAPS|RECOMMENDATION|$)", response, re.DOTALL | re.I)
    if strengths_sec:
        result["strengths"] = [s.strip() for s in re.findall(r"[•-]\s*(.+)", strengths_sec.group(1)) if s.strip()]
    
    gaps_sec = re.search(r"GAPS.*?:(.*?)(?:RECOMMENDATION|$)", response, re.DOTALL | re.I)
    if gaps_sec:
        result["gaps"] = [g.strip() for g in re.findall(r"[•-]\s*(.+)", gaps_sec.group(1)) if g.strip()]
    
    return result

def get_score_color(score: float):
    if score >= 7: return "#00a65a"
    elif score >= 5: return "#f39c12"
    return "#e74c3c"

def get_verdict_icon(verdict: str):
    return {"SELECTED": "✅", "MANAGER REVIEW": "⚠️", "NOT SELECTED": "❌"}.get(verdict, "❓")

def get_verdict_color(verdict: str):
    return {"SELECTED": "#155724", "MANAGER REVIEW": "#856404", "NOT SELECTED": "#721c24"}.get(verdict, "#333")

def scan_resume(jd_text: str, resume_text: str, temperature: float = 0.1):
    prompt = f"JOB DESCRIPTION:\n{jd_text}\n\nCANDIDATE RESUME:\n{resume_text}"
    raw_response = ask_gpt(RESUME_SCANNER_SYSTEM, prompt, temperature)
    if "❌ Error" in raw_response:
        return {"error": raw_response}
    
    result = parse_evaluation_response(raw_response)
    result["raw_output"] = raw_response
    
    just_match = re.search(r"DETAILED JUSTIFICATION:(.*?)(?:STRENGTHS:|GAPS & CONCERNS:|RECOMMENDATION:|$)", raw_response, re.DOTALL | re.I)
    if just_match:
        result["justification"] = just_match.group(1).strip()
    else:
        result["justification"] = "See raw output for details."
        
    return result

# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="text-align: center;"><span class="gradient-text">🎯 Resume Scanner Pro</span></h1>', unsafe_allow_html=True)
st.caption("Training Demo • Visual & Interactive")

# ── Top control bar: role selector + resume selector + temperature + button ──
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.4, 1.4, 0.6, 0.8])

with ctrl1:
    role_list = [f"{v['icon']} {k}" for k, v in JOB_DESCRIPTIONS.items()] + ["✏️ Custom JD"]
    selected_role = st.selectbox("📋 Job Role", role_list, index=0)

with ctrl2:
    resume_option = st.selectbox("📄 Sample Resume", ["Select..."] + list(SAMPLE_RESUMES.keys()) + ["Custom"])

with ctrl3:
    temperature = st.slider("🌡️ Temp", 0.0, 1.0, 0.1)

with ctrl4:
    st.markdown("<br>", unsafe_allow_html=True)
    evaluate_btn = st.button("🔍 Evaluate Candidate", type="primary", use_container_width=True)

st.divider()

# ── Resolve JD & Resume text from selections ─────────────────────────────────
if "Custom" in selected_role:
    weights = {}
    jd_title = "Custom JD"
    jd_text = ""   # filled below in the JD pane
else:
    role_name = selected_role.split(" ", 1)[1]
    jd = JOB_DESCRIPTIONS[role_name]
    jd_text = jd["description"]
    weights = jd.get("weighted_criteria", DEFAULT_WEIGHTS)
    jd_title = jd.get("title", role_name)

if resume_option in SAMPLE_RESUMES:
    resume_text = SAMPLE_RESUMES[resume_option]
else:
    resume_text = ""   # filled below in Resume pane

# ── Three-pane layout ─────────────────────────────────────────────────────────
col_jd, col_res, col_result = st.columns([1, 1, 1.4])

# ── Pane 1: Job Description ───────────────────────────────────────────────────
with col_jd:
    st.markdown("""
    <div style="background:white; border-radius:16px; padding:18px;
                box-shadow:0 4px 20px rgba(0,0,0,0.08); min-height:580px;">
    <h4 style="margin-top:0; color:#155799;">📋 Job Description</h4>
    """, unsafe_allow_html=True)

    if "Custom" in selected_role:
        jd_text = st.text_area("Paste custom JD", height=460, label_visibility="collapsed",
                               placeholder="Paste your job description here…")
    else:
        st.caption(f"**{jd_title}**")
        if weights:
            for k, v in weights.items():
                st.progress(v / 100, text=f"{k} ({v}%)")
            st.markdown("---")
        st.markdown(
            f'<div style="height:400px; overflow-y:auto; font-size:13px; '
            f'color:#444; line-height:1.6;">{jd_text}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ── Pane 2: Candidate Resume ──────────────────────────────────────────────────
with col_res:
    st.markdown("""
    <div style="background:white; border-radius:16px; padding:18px;
                box-shadow:0 4px 20px rgba(0,0,0,0.08); min-height:580px;">
    <h4 style="margin-top:0; color:#159957;">📄 Candidate Resume</h4>
    """, unsafe_allow_html=True)

    if resume_option in SAMPLE_RESUMES:
        st.caption(f"**{resume_option}**")
        st.markdown(
            f'<div style="height:480px; overflow-y:auto; font-size:13px; '
            f'color:#444; white-space:pre-wrap; line-height:1.6;">{resume_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        resume_text = st.text_area("Paste resume", height=460, label_visibility="collapsed",
                                   placeholder="Paste candidate resume here…")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Pane 3: Results ───────────────────────────────────────────────────────────
with col_result:
    st.markdown("""
    <div style="background:white; border-radius:16px; padding:18px;
                box-shadow:0 4px 20px rgba(0,0,0,0.08); min-height:580px;">
    <h4 style="margin-top:0; color:#764ba2;">📊 Evaluation Results</h4>
    """, unsafe_allow_html=True)
    
    # Check if we should run evaluation
    if evaluate_btn:
        if not jd_text.strip():
            st.warning("⚠️ Please provide a job description")
        elif not resume_text.strip():
            st.warning("⚠️ Please provide a candidate resume")
        else:
            with st.spinner("🔍 Analyzing candidate against job requirements..."):
                result = scan_resume(jd_text, resume_text, temperature)
                
                if result.get("error"):
                    st.error(f"❌ Evaluation error: {result['error']}")
                else:
                    # ── Score Display ──────────────────────────────────
                    score = result.get("overall_score", 0)
                    verdict = result.get("verdict", "UNKNOWN")
                    
                    # Big score circle
                    score_color = get_score_color(score)
                    
                    col_score, col_verdict = st.columns([1, 2])
                    
                    with col_score:
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <div class="score-circle" style="
                                background: conic-gradient(
                                    {score_color} 0% {score * 10}%, 
                                    #f0f0f0 {score * 10}% 100%
                                );
                                border: 4px solid {score_color};
                                color: {score_color};
                            ">
                                {score:.1f}
                            </div>
                            <p style="font-size: 12px; color: #999; margin-top: 5px;">out of 10</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_verdict:
                        icon = get_verdict_icon(verdict)
                        color = get_verdict_color(verdict)
                        
                        if verdict == "SELECTED":
                            st.success(f"### {icon} SELECTED")
                            st.markdown(f"""
                            <div style="background: #d4edda; border-radius: 10px; padding: 10px;">
                                <p style="color: #155724; margin: 0;">
                                    🎉 Highly recommended for interview
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            # Balloons for selected
                            st.balloons()
                        elif verdict == "MANAGER REVIEW":
                            st.warning(f"### {icon} MANAGER REVIEW")
                            st.markdown(f"""
                            <div style="background: #fff3cd; border-radius: 10px; padding: 10px;">
                                <p style="color: #856404; margin: 0;">
                                    👤 Requires hiring manager discussion
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"### {icon} NOT SELECTED")
                            st.markdown(f"""
                            <div style="background: #f8d7da; border-radius: 10px; padding: 10px;">
                                <p style="color: #721c24; margin: 0;">
                                    ❌ Does not meet minimum requirements
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # ── Category Scores ──────────────────────────────
                    if result.get("category_scores"):
                        st.markdown("#### 📊 Category Scores")
                        
                        cats = list(result["category_scores"].items())
                        num_cols = min(4, len(cats))
                        cols = st.columns(num_cols)
                        
                        for i, (cat, score_val) in enumerate(cats[:num_cols]):
                            with cols[i % num_cols]:
                                cat_color = get_score_color(score_val)
                                st.markdown(f"""
                                <div style="text-align: center; padding: 10px; 
                                            background: #f8f9fa; border-radius: 10px;
                                            border-left: 4px solid {cat_color};">
                                    <p style="font-size: 12px; color: #666; margin: 0;">
                                        {cat[:20]}
                                    </p>
                                    <p style="font-size: 24px; font-weight: 700; 
                                              color: {cat_color}; margin: 5px 0;">
                                        {score_val:.1f}
                                    </p>
                                    <p style="font-size: 10px; color: #999; margin: 0;">/10</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # ── Strengths and Gaps ────────────────────────────
                    col_sg1, col_sg2 = st.columns(2)
                    
                    with col_sg1:
                        if result.get("strengths"):
                            st.markdown("#### 💪 Strengths")
                            for strength in result["strengths"][:5]:
                                st.success(f"• {strength}")
                    
                    with col_sg2:
                        if result.get("gaps"):
                            st.markdown("#### ⚠️ Gaps & Concerns")
                            for gap in result["gaps"][:5]:
                                st.error(f"• {gap}")
                    
                    st.divider()
                    
                    # ── Full Justification ────────────────────────────
                    st.markdown("#### 📝 Detailed Justification")
                    with st.expander("View full evaluation", expanded=True):
                        st.markdown(result.get("justification", "No justification available"))
                    
                    # ── Raw Output ──────────────────────────────────
                    with st.expander("🔍 View raw AI output"):
                        st.code(result.get("raw_output", ""), language="text")
                    
                    # ── Download Result ──────────────────────────────
                    st.divider()
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        # Export as JSON
                        export_data = {
                            "timestamp": datetime.now().isoformat(),
                            "job_title": jd_title,
                            "score": score,
                            "verdict": verdict,
                            "category_scores": result.get("category_scores", {}),
                            "strengths": result.get("strengths", []),
                            "gaps": result.get("gaps", []),
                            "justification": result.get("justification", ""),
                            "raw_output": result.get("raw_output", "")
                        }
                        json_str = json.dumps(export_data, indent=2)
                        st.download_button(
                            "📥 Download Report (JSON)",
                            data=json_str,
                            file_name=f"candidate_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        # Copy to clipboard
                        st.button(
                            "📋 Copy Summary",
                            on_click=lambda: st.write("Copied!"),
                            use_container_width=True
                        )
    
    else:
        # Placeholder when no evaluation done
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <p style="font-size: 64px; margin: 0;">🔍</p>
            <h3 style="color: #666; margin: 20px 0;">Ready to Evaluate</h3>
            <p style="color: #999; font-size: 16px;">
                Select a job role, load a resume, and click<br>
                <strong>"Evaluate Candidate"</strong> to get started.
            </p>
            <div style="background: #f8f9fa; border-radius: 10px; padding: 15px; 
                        margin-top: 20px; text-align: left;">
                <p style="color: #666; font-size: 14px; margin: 0;">
                    💡 <strong>Tip:</strong> Try loading a <strong>Strong Linux Engineer</strong> 
                    sample and compare with a <strong>Weak Linux Engineer</strong> to see the scoring in action.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()

# ── Training Section ──────────────────────────────────────────────────────

with st.expander("💡 Why This is a Great Prompt Engineering Example", expanded=False):
    col_teach1, col_teach2, col_teach3 = st.columns(3)
    
    with col_teach1:
        st.markdown("""
        ### 📚 Context Engineering
        
        **Structured Context Matters:**
        - Job Description defines evaluation criteria
        - Resume provides evidence to evaluate
        - AI synthesizes both for decisions
        
        **Weighted Scoring System:**
        - Different roles have different priorities
        - AI considers multiple dimensions
        - Transparent, reproducible evaluations
        """)
    
    with col_teach2:
        st.markdown("""
        ### 🎯 Prompt Design Principles
        
        **Applied in This App:**
        ✅ **Clear Instructions**: Detailed scoring rubric
        ✅ **Structured Output**: Score + categories + justification
        ✅ **Role Assignment**: Expert HR consultant persona
        ✅ **Schema Design**: Parsable evaluation format
        ✅ **Constraints**: Evidence-based justification required
        
        **Decision Framework:**
        - **7-10**: SELECTED → Auto-advance
        - **5-6**: MANAGER REVIEW → Human judgment
        - **1-4**: NOT SELECTED → Auto-reject
        """)
    
    with col_teach3:
        st.markdown("""
        ### 🚀 Real-World Impact
        
        **Why This Matters:**
        1. **Consistency**: Same rubric for all candidates
        2. **Scalability**: Evaluate hundreds of resumes
        3. **Transparency**: Every score has justification
        4. **Reduced Bias**: AI applies same criteria fairly
        5. **Time Savings**: HR teams focus on top candidates
        
        **Key Takeaway:**
        The quality of the AI's evaluation depends entirely
        on the quality of the prompt engineering.
        """)

st.divider()

# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <p style="color: #999; font-size: 14px;">
        🚀 Built with GPT-4o-mini • Context Engineering • Prompt Design
    </p>
    <p style="color: #bbb; font-size: 12px;">
        Resume Scanner Pro — AI-Powered Candidate Evaluation
    </p>
</div>
""", unsafe_allow_html=True)