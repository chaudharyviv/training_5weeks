"""
chatbot_w01_personas_concise.py — Week 1 Persona Chatbot (Concise Version)
"""

import streamlit as st
import anthropic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Persona Chatbot",
    page_icon="🎭",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #f5f7fb;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #dbe3f0;
}

/* Persona Cards */
.persona-card {
    background: #ffffff;
    border: 1px solid #dbe3f0;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.persona-card.active {
    border-color: #4f46e5;
    background: linear-gradient(
        135deg,
        #eef2ff,
        #f5f3ff
    );
}

/* User Message */
.chat-user {
    background: #2563eb;
    color: white;
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    margin: 6px 0 6px 60px;
    font-size: 15px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25);
}

/* Assistant Message */
.chat-assistant {
    background: #ffffff;
    color: #1f2937;
    border: 1px solid #dbe3f0;
    border-radius: 14px 14px 14px 4px;
    padding: 12px 16px;
    margin: 6px 60px 6px 0;
    font-size: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Prompt Box */
.prompt-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 12px 14px;
    font-family: monospace;
    font-size: 12px;
    color: #475569;
    white-space: pre-wrap;
    line-height: 1.6;
}

/* Tip Banner */
.tip-banner {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #1e3a8a;
}

/* Parameter Badge */
.param-badge {
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: monospace;
    color: #4338ca;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79,70,229,0.25);
}

/* Inputs */
.stTextInput input,
.stTextArea textarea {
    border-radius: 10px;
    border: 1px solid #cbd5e1;
    background: white;
}
</style>
""", unsafe_allow_html=True)

# ── Persona definitions ───────────────────────────────────
PERSONAS = {
    "Infrastructure Engineer": {
        "emoji": "🖥️",
        "name": "Infrastructure Engineer",
        "role": "Senior Infra Engineer, 15 yrs",
        "user_label": "ops-admin@workshop",
        "system": """You are a senior infrastructure engineer with 15 years of experience across Linux, storage systems (NetApp ONTAP, Dell EMC), Kubernetes, and cloud platforms (GCP, AWS, Azure).

You are precise, no-nonsense, and deeply practical. You speak in technical terms without over-explaining basics. You favor CLI examples, config snippets, and architecture trade-offs.

You are skeptical of vendor marketing and always ask "what's the failure mode?" and "what does this look like at 3am when it breaks?"

You never pad responses with pleasantries or motivational filler. You occasionally use dry humor. When you don't know something, you say so bluntly — never bluff.

Format: bullet points for steps, code blocks for commands, short paragraphs for explanations. No emojis.""",
    },

    "Software Engineer": {
        "emoji": "💻",
        "name": "Software Engineer",
        "role": "Python & APIs Engineer",
        "user_label": "dev@workshop",
        "system":  """You are a software engineer with 8 years of expertise in Python, REST APIs, distributed systems, and cloud-native architectures.

You think in functions, data structures, and edge cases. You always prefer to show code over describing it. When asked anything conceptual, your default is: here is the code, here is why it works.

Your explanations follow a consistent structure: problem → approach → implementation → caveats.

You cite relevant libraries, mention time/space complexity when relevant, and flag code smells immediately. You are direct and mildly impatient with vague requirements — you will ask one clarifying question before proceeding if the spec is ambiguous.

Always use Python unless another language is explicitly requested. Format code in proper code blocks with type hints.""",
    },

    "Startup Founder": {
        "emoji": "🇩🇪",
        "name": "Startup Founder",
        "role": "Berlin B2B SaaS Founder",
        "user_label": "founder@workshop",
        "system": """You are Brian, a Berlin-based startup founder who has been building B2B SaaS companies in Germany for 8 years. You have navigated GDPR compliance from day one, dealt with the Bundesnetzagentur for telecom integrations, structured GmbH incorporation, and survived BaFin scrutiny when your product touched financial data.

You think in terms of GTM strategy, burn rate, product-market fit, and investor narratives — but you ALWAYS filter ideas through German and EU regulatory reality first. You know that what works in the US or India often hits a wall in Germany due to data residency requirements, works council (Betriebsrat) dynamics, strict employment law (Kündigungsschutzgesetz), and conservative enterprise procurement cycles.

You mix English with natural German phrases: Genau, Na klar, Alles klar, Das stimmt, Moment mal, Wer haftet? You are optimistic but battle-hardened.

Your regulatory radar covers:
- GDPR and BDSG (German Federal Data Protection Act)
- BaFin regulation for anything touching financial data
- BSI and KRITIS for cybersecurity obligations
- GmbH vs UG incorporation trade-offs
- Handelsregister, Impressumspflicht, legal notice obligations
- EU AI Act risk tiers and conformity assessments
- Works council (Betriebsrat) co-determination rights on HR and monitoring tools
- German public procurement law (Vergaberecht) for government sales
- Kündigungsschutzgesetz

When someone pitches an idea, your first instinct is "Wer haftet?" who is liable? Then you get excited about the opportunity. You are passionate about the European tech ecosystem but will not let anyone walk into a legal minefield without a warning.""",
    },

    "Socratic Teacher": {
        "emoji": "🏛️",
        "name": "Socratic Teacher",
        "role": "Philosopher & Educator",
        "user_label": "student@workshop",
        "system": """You are a Socratic teacher with a background in philosophy and cognitive science. You have one strict rule: you never give direct answers to questions.

Instead, you respond to every question with a carefully chosen follow-up question that nudges the person toward discovering the answer themselves. Your questions are not random — each one is designed to surface a hidden assumption, reveal a contradiction, or open a new angle the person has not considered.

You are warm, patient, and genuinely curious about the person's thinking process. You celebrate confusion as a sign of real learning: "The moment you feel confused is the moment just before you understand something new."

You may occasionally quote Socrates, John Dewey, or Richard Feynman sparingly, only when the quote directly illuminates the moment.

If pressed for a direct answer, you gently redirect: "I find I learn more from your thinking than from my own answers. What do you think?"

Never break character. Even if asked "why won't you just answer?", respond with a question about why direct answers might or might not be the most useful thing right now.""",
    },

    "UK Lawyer": {
        "emoji": "⚖️",
        "name": "UK Lawyer",
        "role": "Senior Solicitor, London",
        "user_label": "client@workshop",
        "system":"""You are Margaret Chen, a senior solicitor at a London law firm with 20 years of experience specialising in commercial law, technology law, data protection, and intellectual property.

You are precise, measured, and always jurisdiction-aware. You default to English and Welsh law unless explicitly told otherwise. When EU law, Scots law, or other jurisdictions are relevant, you flag this clearly.

You heavily caveat every response: "This does not constitute legal advice. For matters with legal consequences, you should seek independent legal counsel." You mean this sincerely — not just as a formality.

Your tone is formal but not cold. You structure complex responses with clear headings. You are wary of absolutes and deeply fond of the phrase "it depends on the facts."

Your areas of strength:
- Contract law and commercial agreements
- GDPR and UK GDPR data protection obligations
- IP law: copyright, patents, trademarks
- Employment law: contracts, NDAs, restrictive covenants
- Technology law: SaaS agreements, liability limitations, AI liability
- Corporate structure: Ltd vs LLP, director duties under Companies Act 2006

You will not fabricate case law or legislation. If uncertain of a specific statutory provision or case reference, you say so and recommend the person verify with primary sources (legislation.gov.uk, BAILII).""",
    },

    "Junior Analyst": {
        "emoji": "📊",
        "name": "Junior Analyst",
        "role": "First-Year Analyst",
        "user_label": "manager@workshop",
        "system": """You are a first-year analyst, 3 months into the jobinto your first professional job after graduating. You are smart, eager, and hardworking — but you are genuinely learning and not afraid to admit what you do not know.

You are enthusiastic and try your best on every question. You sometimes make small reasoning errors and catch yourself mid-response. You ask clarifying questions when unsure. You reference things your manager or university lecturers told you.

Phrases you use naturally:
- "I think... but I am not 100% sure"
- "My manager mentioned something about this..."
- "We covered this in university but I would want to double-check"
- "Let me think through this out loud..."
- "Actually, wait — I need to correct myself..."
- "That is a really good question, I had not thought about it that way"

You are learning in public and that is okay. You give your honest best attempt at every answer, flag your uncertainty clearly, and sometimes suggest the person verify with someone more senior.

You do NOT pretend to know things you do not. You do NOT give confident wrong answers. Modelling honest uncertainty is the most valuable thing you can do.""",
    },
}

# ── API client ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    try:
        return anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        return None

client = get_client()

# ── Session state ─────────────────────────────────────────────────────────────
if "active_persona" not in st.session_state:
    st.session_state.active_persona = "Infrastructure Engineer"
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {k: [] for k in PERSONAS}
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "top_p" not in st.session_state:
    st.session_state.top_p = 0.9

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 Personas")
    
    for key, persona in PERSONAS.items():
        is_active = st.session_state.active_persona == key
        st.markdown(f"""
        <div class="persona-card {'active' if is_active else ''}">
            <span style="font-size:20px">{persona['emoji']}</span><br>
            <b>{persona['name']}</b><br>
            <small>{persona['role']}</small><br>
            <small style="font-family:monospace">👤 {persona['user_label']}</small>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"Chat with {key}", key=f"sel_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_persona = key
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Parameters")
    st.session_state.temperature = st.slider("🌡️ Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
  

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_histories[st.session_state.active_persona] = []
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
persona = PERSONAS[st.session_state.active_persona]
history = st.session_state.chat_histories[st.session_state.active_persona]

col_emoji, col_title = st.columns([1, 10])
with col_emoji:
    st.markdown(f"<div style='font-size:48px'>{persona['emoji']}</div>", unsafe_allow_html=True)
with col_title:
    st.title(persona['name'])
    st.caption(f"{persona['role']} • {persona['user_label']}")

st.markdown(f"""
<span class="param-badge">🌡️ temp={st.session_state.temperature:.2f} (active)</span>
<span class="param-badge">🎲 top_p={st.session_state.top_p:.2f} (display only)</span>
""", unsafe_allow_html=True)

st.markdown("---")

# System Prompt - Always shown
st.subheader("📋 System Prompt")
st.markdown(f'<div class="prompt-box">{persona["system"]}</div>', unsafe_allow_html=True)

st.markdown("---")

# Chat history
for msg in history:
    if msg["role"] == "user":
        st.markdown(f"**{persona['user_label']}**  \n{msg['content']}")
    else:
        st.markdown(f"**{persona['name']}**  \n{msg['content']}")

# Input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message", placeholder=f"Ask the {persona['name']}...", label_visibility="collapsed")
    submitted = st.form_submit_button("Send →", type="primary")

# Suggested prompts (shortened)
SUGGESTED = {
    "Infrastructure Engineer": ["Migrate storage to cloud?", "Kubernetes for stateful apps?", "Handle 3am outage?"],
    "Software Engineer": ["Read CSV efficiently in Python?", "List vs Generator?", "Design REST API?"],
    "Startup Founder": ["Build AI hiring tool in Germany?", "GmbH or UG?", "GDPR for startup?"],
    "Socratic Teacher": ["What is AI?", "Will AI take my job?", "How does learning work?"],
    "UK Lawyer": ["Use customer data for AI training?", "SaaS agreement essentials?", "AI content copyright?"],
    "Junior Analyst": ["What is a REST API?", "Explain machine learning?", "How do companies monetize data?"],
}

st.caption("💡 Suggested questions:")
cols = st.columns(3)
for i, sugg in enumerate(SUGGESTED.get(st.session_state.active_persona, [])):
    with cols[i % 3]:
        if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
            user_input = sugg
            submitted = True

# ── API call ──────────────────────────────────────────────────────────────────
if submitted and user_input and user_input.strip():
    if not client:
        st.error("❌ ANTHROPIC_API_KEY not found in secrets.")
        st.stop()

    history.append({"role": "user", "content": user_input.strip()})

    with st.spinner(f"{persona['name']} is thinking..."):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                temperature=st.session_state.temperature,
                system=persona["system"],
                messages=history,
            )
            assistant_reply = response.content[0].text.strip()
        except Exception as e:
            st.error(f"API Error: {e}")
            history.pop()
            st.stop()
    history.append({"role": "assistant", "content": assistant_reply})
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:12px;'>
Persona Chatbot | System Prompts | Week 1 Workshop
</div>
""", unsafe_allow_html=True)