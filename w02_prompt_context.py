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
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Card styling */
    .stCard {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
    }
    
    /* Gradient headers */
    .gradient-text {
        background: linear-gradient(120deg, #155799, #159957);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Score display */
    .score-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        font-size: 36px;
        font-weight: 800;
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    .score-circle:hover {
        transform: scale(1.05);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
    }
    
    /* Verdict badges */
    .verdict-badge {
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 18px;
        text-align: center;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Animated stats */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stat-card {
        animation: slideIn 0.5s ease-out;
        background: white;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        border-radius: 20px;
        height: 12px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Custom expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2c3e50;
        border-radius: 10px !important;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f0f2f6 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Divider styling */
    hr {
        margin: 30px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Badge animations */
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Success/Error/Warning styling */
    .stAlert {
        border-radius: 15px;
        border-left: 6px solid;
        padding: 15px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Clients ───────────────────────────────────────────────────────────────────
if "oai" not in st.session_state:
    from openai import OpenAI
    st.session_state.oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

oai = st.session_state.oai

# ── Job Descriptions ──────────────────────────────────────────────────────────
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
""",
        "weighted_criteria": {
            "Technical Skills": 40,
            "Experience": 25,
            "Education & Certifications": 20,
            "Soft Skills": 15
        }
    },
    
    "Coding Engineer": {
        "title": "💻 Coding Engineer",
        "icon": "💻",
        "description": """
**Role Summary:**
We are looking for a talented Coding Engineer to join our development team. The ideal candidate will write clean, efficient, and maintainable code while contributing to all phases of the software development lifecycle.

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
""",
        "weighted_criteria": {
            "Technical Skills": 45,
            "Experience": 25,
            "Education": 15,
            "Soft Skills": 15
        }
    },
    
    "Project Manager": {
        "title": "📊 Project Manager",
        "icon": "📊",
        "description": """
**Role Summary:**
We are seeking an experienced Project Manager to lead and deliver complex technology projects. The ideal candidate will have a proven track record of managing cross-functional teams and delivering projects on time and within budget.

**Key Responsibilities:**
- Lead end-to-end project delivery from initiation to closure
- Develop comprehensive project plans, timelines, and budgets
- Coordinate cross-functional teams (development, QA, operations, business stakeholders)
- Manage stakeholder expectations and provide regular project updates
- Identify and mitigate project risks and issues proactively
- Ensure project documentation and reporting are maintained
- Drive continuous improvement in project management processes

**Required Skills & Experience:**
- 7+ years of project management experience in technology
- Proven track record of delivering complex projects
- Strong understanding of project management methodologies (Agile, Waterfall, Hybrid)
- Experience with project management tools (JIRA, MS Project, Asana)
- Budget management and resource allocation expertise
- Risk management and contingency planning skills
- Bachelor's degree in Business, Computer Science, or related field

**Nice to Have:**
- PMP, Prince2, or Agile certification
- Experience with AI/ML projects
- Background in software development
- International project experience

**Soft Skills:**
- Exceptional leadership and team management abilities
- Excellent communication and presentation skills
- Strong negotiation and stakeholder management skills
- Adaptability and ability to work under pressure
""",
        "weighted_criteria": {
            "Leadership & Management": 30,
            "Project Delivery": 30,
            "Communication": 25,
            "Certifications & Education": 15
        }
    },
    
    "Service Delivery Manager": {
        "title": "🔄 Service Delivery Manager",
        "icon": "🔄",
        "description": """
**Role Summary:**
We are seeking a proactive Service Delivery Manager to ensure exceptional IT service delivery to our clients. The ideal candidate will manage service level agreements (SLAs), drive service improvement, and maintain strong client relationships.

**Key Responsibilities:**
- Manage service delivery operations to meet or exceed agreed SLAs
- Lead incident management, problem management, and change management processes
- Develop and maintain strong client relationships as a trusted advisor
- Monitor service performance metrics and generate management reports
- Drive continuous service improvement initiatives
- Manage service delivery team including resource allocation and performance
- Conduct service reviews and present to executive stakeholders

**Required Skills & Experience:**
- 6+ years of service delivery or IT operations management experience
- Proven track record of managing SLAs and client relationships
- Deep understanding of ITIL framework and ITSM best practices
- Experience with incident, problem, and change management
- Strong vendor management and negotiation skills
- Experience with service management tools (ServiceNow, JIRA, etc.)
- Bachelor's degree in Business, IT, or related field

**Nice to Have:**
- ITIL certification (V3 or V4)
- Experience with managed services or consulting
- Knowledge of cloud services and infrastructure
- Experience in financial services or healthcare sectors

**Soft Skills:**
- Excellent client-facing and communication skills
- Strong leadership and team motivation abilities
- Strategic thinking with operational focus
- Problem-solving and crisis management skills
""",
        "weighted_criteria": {
            "Service Delivery Experience": 35,
            "Client & Stakeholder Management": 30,
            "Process & ITIL Knowledge": 20,
            "Leadership & Communication": 15
        }
    }
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

    "Strong Project Manager": """
MICHAEL CHEN
Senior Project Manager
Email: michael.c@email.com | Phone: (555) 345-6789

PROFESSIONAL SUMMARY
Senior Project Manager with 10 years of experience leading technology projects for Fortune 500 companies. Certified PMP with expertise in Agile methodologies and digital transformation initiatives.

TECHNICAL SKILLS
• Project Management: Agile, Scrum, Kanban, Waterfall, Hybrid
• Tools: JIRA, MS Project, Asana, Trello, Confluence
• Budget Management: Program-level budgeting up to $5M
• Risk Management: MITRE, ISO 31000
• Stakeholder Management: Executive reporting, client relations
• Project Lifecycle: Initiation to closure, change management
• Communication: Technical and business reporting, presentations

PROFESSIONAL EXPERIENCE

Senior Project Manager | GlobalTech Systems | 2018-Present
• Managed 15+ technology projects ranging from $500K to $5M annually
• Led 30-person cross-functional teams (developers, QA, operations, business)
• Achieved 95% on-time delivery rate across all projects
• Implemented Agile transformation across 3 departments
• Drove $2M cost savings through process optimization
• Recognized as "Top Performer" 2022 for exceptional project delivery

Project Manager | InnovateCorp | 2015-2018
• Managed software development projects for financial services clients
• Coordinated with 5+ simultaneous project streams
• Maintained project budgets and resource allocation
• Conducted stakeholder meetings and status reporting
• Successfully delivered 12 projects ahead of schedule

Project Manager | TechStart Inc. | 2013-2015
• Led product development for SaaS platform
• Managed vendor relationships and procurement
• Created project schedules and resource plans
• Facilitated daily stand-ups and sprint planning

EDUCATION
Master of Business Administration
Harvard Business School | 2013
GPA: 3.8/4.0

Bachelor of Science in Computer Engineering
MIT | 2011
GPA: 3.6/4.0

CERTIFICATIONS
• Project Management Professional (PMP)
• Certified Scrum Master (CSM)
• SAFe Agilist

ACHIEVEMENTS
• PMI Project of the Year Nominee 2022
• Published article on "Agile at Scale" in Project Management Journal
""",

    "Weak Project Manager": """
LISA ADAMS
Project Coordinator
Email: lisa.a@email.com | Phone: (555) 456-7890

PROFESSIONAL SUMMARY
Recent MBA graduate seeking project management opportunities. Experience in administrative coordination and team support. Passionate about helping teams succeed.

TECHNICAL SKILLS
• Microsoft Office Suite (Excel, Word, PowerPoint)
• Team collaboration tools (Slack, Teams)
• Basic project tracking (JIRA, Trello)
• Administrative coordination
• Event planning
• Basic data analysis in Excel

PROFESSIONAL EXPERIENCE

Project Coordinator | StartupX | 2021-Present
• Assist project managers with scheduling and documentation
• Track project milestones and update tracking tools
• Coordinate team meetings and prepare materials
• Maintain project files and documentation
• Respond to stakeholder inquiries

Administrative Assistant | Corporate Inc. | 2019-2021
• Provided administrative support to executive team
• Managed calendars and scheduled appointments
• Prepared meeting minutes and distributed to attendees
• Coordinated office events and activities
• Managed general office operations

EDUCATION
Master of Business Administration
State University | 2022
GPA: 3.5/4.0

Bachelor of Arts in Communications
State University | 2019
GPA: 3.6/4.0

CERTIFICATIONS
None

INTERNSHIP
Summer Intern | TechStart | 2018
• Assisted with market research for new product
• Created presentations for investor meetings
""",

    "Strong Service Delivery Manager": """
ROBERT GARCIA
Service Delivery Manager
Email: robert.g@email.com | Phone: (555) 567-8901

PROFESSIONAL SUMMARY
Service Delivery Manager with 10 years of experience ensuring exceptional IT service delivery for enterprise clients. Expert in ITIL framework with proven ability to maintain high SLA performance and client satisfaction.

TECHNICAL SKILLS
• Service Management: ITIL V3/V4 (Incident, Problem, Change, Knowledge)
• Tools: ServiceNow, BMC Remedy, JIRA, Confluence
• SLA Management: Performance monitoring, reporting
• Client Management: Enterprise accounts, stakeholder relations
• Operations: Service desk management, escalation procedures
• Metrics: KPI development, dashboard reporting, CSAT

PROFESSIONAL EXPERIENCE

Service Delivery Manager | EnterpriseIT Solutions | 2018-Present
• Manage service delivery for 3 enterprise clients (annual revenue $12M)
• Maintain 99.9% SLA compliance across all service lines
• Implemented ITIL processes resulting in 40% faster incident resolution
• Led 25-person service delivery team across multiple locations
• Achieved 95% client satisfaction (CSAT) - highest in organization
• Reduced critical incidents by 30% through proactive monitoring

Service Delivery Lead | TechServices Global | 2015-2018
• Managed service delivery for 6 accounts totaling $8M in revenue
• Designed and implemented service improvement plans
• Conducted quarterly service reviews with executive stakeholders
• Managed 15-person team across help desk and service desk operations
• Successfully onboarded 3 new clients in 12 months

IT Operations Manager | DataSecure Corp | 2012-2015
• Led IT operations for 1000+ user organization
• Managed incident, problem, and change management processes
• Reduced system downtime by 50% through improved processes
• Implemented monitoring and alerting systems

EDUCATION
Master of Science in Information Systems
Carnegie Mellon University | 2012
GPA: 3.7/4.0

Bachelor of Science in Computer Science
University of Texas | 2010
GPA: 3.5/4.0

CERTIFICATIONS
• ITIL V4 Managing Professional
• ITIL V3 Expert
• Certified Service Manager (CSM)
• TOGAF Certified

ACHIEVEMENTS
• Awarded "Service Excellence" by Clients 2021
• Published whitepaper on "Digital Service Transformation"
• Speaker at ServiceNow Knowledge 2023
""",

    "Weak Service Delivery Manager": """
PATRICIA LEE
Service Desk Supervisor
Email: patricia.l@email.com | Phone: (555) 678-9012

PROFESSIONAL SUMMARY
Service desk professional with experience in customer support and team supervision. Seeking growth opportunity in service delivery management.

TECHNICAL SKILLS
• Service desk operations
• Customer service and support
• Team supervision (4-5 people)
• Help desk ticketing systems (Zendesk, Freshservice)
• Microsoft Office Suite
• Basic IT knowledge (hardware, software)

PROFESSIONAL EXPERIENCE

Service Desk Supervisor | HelpTech Inc. | 2020-Present
• Supervise 5 help desk agents providing tier 1 support
• Monitor ticket queues and manage workloads
• Handle escalations and complex customer issues
• Create shift schedules and manage attendance
• Prepare basic service reports

Senior Support Agent | CustomerCare Solutions | 2018-2020
• Provided technical support for various software products
• Responded to 30+ customer tickets daily
• Escalated unresolved issues to tier 2 support
• Maintained knowledge base articles

Customer Support Representative | TeleSupport Inc. | 2016-2018
• Answered customer calls and emails
• Resolved basic technical issues
• Documented customer interactions in ticketing system
• Handled customer complaints professionally

EDUCATION
Bachelor of Arts in Communications
State College | 2016
GPA: 3.4/4.0

CERTIFICATIONS
• ITIL V3 Foundation (2021)
• Help Desk Institute (HDI) Certified

PROJECTS
• Implemented new call routing system
• Created customer service training materials
"""
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

def ask_gpt(system: str, user: str, temperature: float = 0, max_tokens: int = 1000) -> str:
    """Call GPT-4o-mini with the given prompts."""
    try:
        r = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"

def parse_evaluation_response(response: str) -> dict:
    """Parse the AI evaluation response to extract structured data."""
    result = {
        "raw": response,
        "overall_score": 0,
        "verdict": "ERROR",
        "category_scores": {},
        "strengths": [],
        "gaps": [],
        "justification": response
    }
    
    # Extract overall score
    score_pattern = r"Overall Score:\s*([\d.]+)/10"
    score_match = re.search(score_pattern, response)
    if score_match:
        result["overall_score"] = float(score_match.group(1))
    
    # Extract verdict
    verdict_pattern = r"Verdict:\s*(\w+)"
    verdict_match = re.search(verdict_pattern, response)
    if verdict_match:
        result["verdict"] = verdict_match.group(1)
    
    # Extract category scores
    cat_pattern = r"([^-]+):\s*([\d.]+)/10"
    for match in re.finditer(cat_pattern, response):
        category = match.group(1).strip()
        score = float(match.group(2))
        if category and category not in ["Overall Score"]:
            result["category_scores"][category] = score
    
    # Extract strengths
    strengths_section = re.search(r"STRENGTHS:(.*?)(?:GAPS|$)", response, re.DOTALL)
    if strengths_section:
        strengths = re.findall(r"[•-]\s*(.+?)(?:\n|$)", strengths_section.group(1))
        result["strengths"] = [s.strip() for s in strengths if s.strip()]
    
    # Extract gaps
    gaps_section = re.search(r"GAPS.*?:(.*?)(?:RECOMMENDATION|$)", response, re.DOTALL)
    if gaps_section:
        gaps = re.findall(r"[•-]\s*(.+?)(?:\n|$)", gaps_section.group(1))
        result["gaps"] = [g.strip() for g in gaps if g.strip()]
    
    return result

def scan_resume(jd_text: str, resume_text: str, temperature: float = 0.1) -> dict:
    """Evaluate a resume against job description with scoring."""
    
    user_prompt = f"""
JOB DESCRIPTION:
{jd_text}

CANDIDATE RESUME:
{resume_text}

Please evaluate this candidate using the scoring system and format specified. Be thorough and evidence-based.
"""
    
    try:
        response = ask_gpt(RESUME_SCANNER_SYSTEM, user_prompt, temperature, max_tokens=1000)
        parsed = parse_evaluation_response(response)
        parsed["raw_output"] = response
        return parsed
    except Exception as e:
        return {
            "error": str(e),
            "raw_output": "",
            "verdict": "ERROR",
            "overall_score": 0
        }

def get_verdict_icon(verdict: str) -> str:
    """Get icon for verdict display."""
    icons = {
        "SELECTED": "✅",
        "MANAGER REVIEW": "⚠️",
        "NOT SELECTED": "❌",
        "ERROR": "⚙️"
    }
    return icons.get(verdict.upper(), "❓")

def get_verdict_color(verdict: str) -> str:
    """Get color for verdict display."""
    colors = {
        "SELECTED": "#00a65a",
        "MANAGER REVIEW": "#f39c12",
        "NOT SELECTED": "#e74c3c",
        "ERROR": "#95a5a6"
    }
    return colors.get(verdict.upper(), "#95a5a6")

def get_score_color(score: float) -> str:
    """Get color based on score."""
    if score >= 7:
        return "#00a65a"
    elif score >= 5:
        return "#f39c12"
    else:
        return "#e74c3c"

# ── UI Header ────────────────────────────────────────────────────────────────

# Hero section with gradient
st.markdown("""
<div style="text-align: center; padding: 20px 0 30px 0;">
    <h1 style="font-size: 48px; margin-bottom: 0;">
        <span style="background: linear-gradient(120deg, #155799, #159957); 
                     -webkit-background-clip: text; 
                     -webkit-text-fill-color: transparent;
                     font-weight: 800;">
            🎯 Resume Scanner Pro
        </span>
    </h1>
    <p style="font-size: 18px; color: #666; margin-top: 10px;">
        AI-Powered Candidate Evaluation with 1-10 Scoring System
    </p>
    <p style="font-size: 14px; color: #999;">
        Built with GPT-4o-mini • Context Engineering • Prompt Design
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Main Layout ─────────────────────────────────────────────────────────────

# Left: Input Section, Right: Results
col_left, col_right = st.columns([1.2, 1.8], gap="large")

with col_left:
    st.markdown("""
    <div style="background: white; border-radius: 20px; padding: 20px; 
                box-shadow: 0 10px 40px rgba(0,0,0,0.08);">
    """, unsafe_allow_html=True)
    
    # ── Job Description ─────────────────────────────────────────────────
    st.markdown("### 📋 Job Description")
    
    # Role selection with icons
    selected_role = st.radio(
        "Select role:",
        [f"{JOB_DESCRIPTIONS['Linux Engineer']['icon']} Linux Engineer",
         f"{JOB_DESCRIPTIONS['Coding Engineer']['icon']} Coding Engineer",
         f"{JOB_DESCRIPTIONS['Project Manager']['icon']} Project Manager",
         f"{JOB_DESCRIPTIONS['Service Delivery Manager']['icon']} Service Delivery Manager",
         "✏️ Custom JD"],
        index=0,
        horizontal=False
    )
    
    # Determine which JD to use
    if "Custom" in selected_role:
        jd_text = st.text_area(
            "Enter custom Job Description:",
            height=200,
            placeholder="Paste job description here...",
            key="custom_jd"
        )
        jd_title = "Custom JD"
        weights = {}
    else:
        # Extract the role name without icon
        role_name = selected_role.split(" ", 1)[1] if " " in selected_role else selected_role
        # Remove icon and clean
        for role in JOB_DESCRIPTIONS:
            if role in role_name:
                jd_text = JOB_DESCRIPTIONS[role]["description"]
                jd_title = role
                weights = JOB_DESCRIPTIONS[role]["weighted_criteria"]
                break
        else:
            jd_text = JOB_DESCRIPTIONS["Linux Engineer"]["description"]
            jd_title = "Linux Engineer"
            weights = JOB_DESCRIPTIONS["Linux Engineer"]["weighted_criteria"]
        
        with st.expander("📄 View Job Description"):
            st.markdown(jd_text)
    
    # Weighted criteria display
    if weights:
        st.caption("**🎯 Weighted Criteria:**")
        for cat, weight in weights.items():
            st.progress(weight / 100, text=f"{cat}: {weight}%")
    
    st.divider()
    
    # ── Resume Input ──────────────────────────────────────────────────
    st.markdown("### 📄 Candidate Resume")
    
    resume_option = st.selectbox(
        "Load sample resume:",
        ["Select a sample..."] + list(SAMPLE_RESUMES.keys()) + ["Custom Resume"],
        index=0
    )
    
    if resume_option != "Select a sample..." and resume_option != "Custom Resume":
        resume_text = SAMPLE_RESUMES[resume_option]
        with st.expander("📄 View Resume"):
            st.markdown(resume_text)
        st.caption(f"📎 Loaded: {resume_option}")
    else:
        resume_text = st.text_area(
            "Paste candidate resume:",
            height=250,
            placeholder="Paste the candidate's resume here...",
            key="custom_resume"
        )
    
    st.divider()
    
    # ── Evaluation Settings ───────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    
    col_temp, col_btn = st.columns([1, 1.5])
    
    with col_temp:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Lower = consistent, Higher = creative"
        )
    
    with col_btn:
        st.markdown("###")
        evaluate_btn = st.button(
            "🔍 Evaluate Candidate",
            type="primary",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ── Right Column: Results ──────────────────────────────────────────────────

with col_right:
    st.markdown("""
    <div style="background: white; border-radius: 20px; padding: 20px; 
                box-shadow: 0 10px 40px rgba(0,0,0,0.08); min-height: 500px;">
    """, unsafe_allow_html=True)
    
    # ── Results Header ─────────────────────────────────────────────────
    st.markdown("### 📊 Evaluation Results")
    
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

# ── Teaching Section ──────────────────────────────────────────────────────

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