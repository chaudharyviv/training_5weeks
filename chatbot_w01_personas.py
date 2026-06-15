"""
chatbot_w01_personas.py — Week 1 Persona Chatbot
A focused demo showing how system prompts shape LLM personality and behaviour.

Personas:
  🖥️  Infrastructure Engineer  — Alex        | user: ops-admin@workshop
  💻  Coding Engineer          — Priya       | user: dev@workshop
  🇩🇪  Startup Founder (DE)    — Lukas       | user: founder@workshop
  🏛️  Socratic Teacher         — The Teacher | user: student@workshop
  ⚖️  UK Lawyer                — Margaret    | user: client@workshop
  📊  Junior Analyst           — Ravi        | user: manager@workshop

Deploy: Streamlit Community Cloud
Secrets: ANTHROPIC_API_KEY (set via App Settings → Secrets in Community Cloud)

Run locally:
    pip install streamlit anthropic
    # create .streamlit/secrets.toml with ANTHROPIC_API_KEY = "sk-ant-..."
    streamlit run chatbot_w01_personas.py
"""

import streamlit as st
import anthropic

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Persona Chatbot — Week 1",
    page_icon="🎭",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] {
    background: #1a1d27;
    border-right: 1px solid #2e3148;
}

/* Persona cards */
.persona-card {
    background: #22253a;
    border: 1px solid #3a3f5c;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.persona-card.active {
    border-color: #7c6af7;
    background: #2a2d4a;
}
.persona-card .p-emoji { font-size: 20px; }
.persona-card .p-name  { font-weight: 700; color: #e8e8f0; font-size: 14px; }
.persona-card .p-role  { color: #8a8fb5; font-size: 11px; margin-top: 1px; }
.persona-card .p-user  { color: #5a7a5a; font-size: 10px; margin-top: 3px;
                          font-family: monospace; }

/* Chat bubbles */
.chat-user {
    background: #1e2235;
    border: 1px solid #2e3355;
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 6px 0 6px 60px;
    color: #c8d0f0;
    font-size: 15px;
    line-height: 1.6;
}
.chat-assistant {
    background: #1a2535;
    border: 1px solid #1f3a5f;
    border-radius: 12px 12px 12px 2px;
    padding: 12px 16px;
    margin: 6px 60px 6px 0;
    color: #d0e8d0;
    font-size: 15px;
    line-height: 1.6;
}
.chat-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 3px;
    color: #5a6090;
    font-family: monospace;
}
.chat-label.assistant-label { color: #4a8a6a; }

/* System prompt box */
.prompt-box {
    background: #13161f;
    border: 1px dashed #3a3f5c;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    color: #8090b0;
    white-space: pre-wrap;
    line-height: 1.7;
}

/* Tip banner */
.tip-banner {
    background: #1a1f35;
    border-left: 3px solid #7c6af7;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #a0a8d0;
    margin-bottom: 14px;
}

/* Param badge */
.param-badge {
    display: inline-block;
    background: #1e2235;
    border: 1px solid #3a3f5c;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-family: monospace;
    color: #8090c0;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Persona definitions ───────────────────────────────────────────────────────
PERSONAS = {
    "Infrastructure Engineer": {
        "emoji": "🖥️",
        "name": "Alex",
        "role": "Senior Infra Engineer, 15 yrs",
        "user_label": "ops-admin@workshop",
        "color": "#4ecdc4",
        "system": """You are Alex, a senior infrastructure engineer with 15 years of experience across Linux, storage systems (NetApp ONTAP, Dell EMC), Kubernetes, and cloud platforms (GCP, AWS, Azure).

You are precise, no-nonsense, and deeply practical. You speak in technical terms without over-explaining basics. You favor CLI examples, config snippets, and architecture trade-offs.

You are skeptical of vendor marketing and always ask "what's the failure mode?" and "what does this look like at 3am when it breaks?"

You never pad responses with pleasantries or motivational filler. You occasionally use dry humor. When you don't know something, you say so bluntly — never bluff.

Format: bullet points for steps, code blocks for commands, short paragraphs for explanations. No emojis.""",
    },

    "Coding Engineer": {
        "emoji": "💻",
        "name": "Priya",
        "role": "Software Engineer — Python & APIs",
        "user_label": "dev@workshop",
        "color": "#45b7d1",
        "system": """You are Priya, a software engineer with 8 years of expertise in Python, REST APIs, distributed systems, and cloud-native architectures.

You think in functions, data structures, and edge cases. You always prefer to show code over describing it. When asked anything conceptual, your default is: here is the code, here is why it works.

Your explanations follow a consistent structure: problem → approach → implementation → caveats.

You cite relevant libraries, mention time/space complexity when relevant, and flag code smells immediately. You are direct and mildly impatient with vague requirements — you will ask one clarifying question before proceeding if the spec is ambiguous.

Always use Python unless another language is explicitly requested. Format code in proper code blocks with type hints.""",
    },

    "Startup Founder 🇩🇪": {
        "emoji": "🇩🇪",
        "name": "Lukas Becker",
        "role": "Berlin B2B SaaS Founder, 8 yrs",
        "user_label": "founder@workshop",
        "color": "#f7dc6f",
        "system": """You are Lukas Becker, a Berlin-based startup founder who has been building B2B SaaS companies in Germany for 8 years. You have navigated GDPR compliance from day one, dealt with the Bundesnetzagentur for telecom integrations, structured GmbH incorporation, and survived BaFin scrutiny when your product touched financial data.

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
- Kündigungsschutzgesetz — firing people in Germany is hard and expensive

When someone pitches an idea, your first instinct is "Wer haftet?" — who is liable? Then you get excited about the opportunity. You are passionate about the European tech ecosystem but will not let anyone walk into a legal minefield without a warning.""",
    },

    "Socratic Teacher": {
        "emoji": "🏛️",
        "name": "The Teacher",
        "role": "Philosopher & Educator",
        "user_label": "student@workshop",
        "color": "#dda0dd",
        "system": """You are a Socratic teacher with a background in philosophy and cognitive science. You have one strict rule: you never give direct answers to questions.

Instead, you respond to every question with a carefully chosen follow-up question that nudges the person toward discovering the answer themselves. Your questions are not random — each one is designed to surface a hidden assumption, reveal a contradiction, or open a new angle the person has not considered.

You are warm, patient, and genuinely curious about the person's thinking process. You celebrate confusion as a sign of real learning: "The moment you feel confused is the moment just before you understand something new."

You may occasionally quote Socrates, John Dewey, or Richard Feynman — sparingly, only when the quote directly illuminates the moment.

If pressed for a direct answer, you gently redirect: "I find I learn more from your thinking than from my own answers. What do you think?"

Never break character. Even if asked "why won't you just answer?", respond with a question about why direct answers might or might not be the most useful thing right now.""",
    },

    "UK Lawyer": {
        "emoji": "⚖️",
        "name": "Margaret",
        "role": "Senior Solicitor, London",
        "user_label": "client@workshop",
        "color": "#ff9f7a",
        "system": """You are Margaret Chen, a senior solicitor at a London law firm with 20 years of experience specialising in commercial law, technology law, data protection, and intellectual property.

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
        "name": "Ravi",
        "role": "First-Year Analyst, 3 months in",
        "user_label": "manager@workshop",
        "color": "#98d8c8",
        "system": """You are Ravi Sharma, a first-year analyst at a consulting firm, three months into your first professional job after graduating. You are smart, eager, and hardworking — but you are genuinely learning and not afraid to admit what you do not know.

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
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None

client = get_client()

# ── Session state ─────────────────────────────────────────────────────────────
if "active_persona" not in st.session_state:
    st.session_state.active_persona = "Infrastructure Engineer"
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {k: [] for k in PERSONAS}
if "show_system_prompt" not in st.session_state:
    st.session_state.show_system_prompt = False
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "top_p" not in st.session_state:
    st.session_state.top_p = 0.9

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 Personas")
    st.caption("Same question. Six different answers.")
    st.markdown("---")

    for persona_key, persona in PERSONAS.items():
        is_active = st.session_state.active_persona == persona_key
        st.markdown(f"""
        <div class="persona-card {'active' if is_active else ''}">
            <span class="p-emoji">{persona['emoji']}</span>
            <div class="p-name">{persona['name']}</div>
            <div class="p-role">{persona['role']}</div>
            <div class="p-user">👤 {persona['user_label']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            f"Chat as {persona['user_label'].split('@')[0]}",
            key=f"select_{persona_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_persona = persona_key
            st.rerun()

    st.markdown("---")

    # ── Sampling parameters ───────────────────────────────────────────────────
    st.markdown("### ⚙️ Sampling Parameters")
    st.caption("Adjust and re-send any message to see the difference.")

    st.session_state.temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.05,
        help="Controls randomness. 0 = deterministic and focused. 1 = creative and varied.",
    )

    # Temperature behaviour hint
    temp = st.session_state.temperature
    if temp <= 0.2:
        st.caption("🧊 Very focused — nearly identical answers every time")
    elif temp <= 0.5:
        st.caption("🎯 Balanced — consistent with some variation")
    elif temp <= 0.75:
        st.caption("✨ Creative — noticeable variation between runs")
    else:
        st.caption("🔥 High — unpredictable, may drift from persona")

    st.markdown("")

    st.session_state.top_p = st.slider(
        "🎲 Top-p (nucleus sampling)",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.top_p,
        step=0.05,
        help="Limits token selection to the top-p probability mass. Lower = more focused vocabulary.",
    )

    top_p = st.session_state.top_p
    if top_p <= 0.5:
        st.caption("🎯 Narrow nucleus — model sticks to high-probability words")
    elif top_p <= 0.85:
        st.caption("📖 Moderate — good balance of fluency and variety")
    else:
        st.caption("🌐 Wide nucleus — full vocabulary range considered")

    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑️ This chat", use_container_width=True):
            st.session_state.chat_histories[st.session_state.active_persona] = []
            st.rerun()
    with col_c2:
        if st.button("🗑️ All chats", use_container_width=True):
            st.session_state.chat_histories = {k: [] for k in PERSONAS}
            st.rerun()

    st.session_state.show_system_prompt = st.toggle(
        "👁️ Show system prompt",
        value=st.session_state.show_system_prompt,
    )

    st.markdown("---")
    st.caption("Model: `claude-sonnet-4-6`")
    st.caption("Week 1 — GenAI Workshop")
    st.success("✅ API Connected") if client else st.error("❌ API key missing")

# ── Main area ─────────────────────────────────────────────────────────────────
persona    = PERSONAS[st.session_state.active_persona]
history    = st.session_state.chat_histories[st.session_state.active_persona]
user_label = persona["user_label"]

# Header
col_emoji, col_title = st.columns([1, 10])
with col_emoji:
    st.markdown(f"<div style='font-size:46px;margin-top:10px'>{persona['emoji']}</div>",
                unsafe_allow_html=True)
with col_title:
    st.markdown(f"## {persona['name']}")
    st.caption(f"{persona['role']}  ·  talking to **{user_label}**")

# Active param badges
st.markdown(
    f'<span class="param-badge">🌡️ temp = {st.session_state.temperature:.2f}</span>'
    f'<span class="param-badge">🎲 top_p = {st.session_state.top_p:.2f}</span>',
    unsafe_allow_html=True,
)

st.markdown("---")

# Workshop tip
TIPS = {
    "Infrastructure Engineer": "💡 **Workshop tip:** Ask *"Should we move our storage to the cloud?"* — then switch to Lukas and ask the same thing.",
    "Coding Engineer":         "💡 **Workshop tip:** Ask *"How do I read a file in Python?"* — watch Priya lead with code before explanation.",
    "Startup Founder 🇩🇪":     "💡 **Workshop tip:** Pitch *"I want to build an AI hiring tool for German companies"* — watch the regulatory check unfold.",
    "Socratic Teacher":        "💡 **Workshop tip:** Ask *"What is artificial intelligence?"* — notice you will never get a direct answer.",
    "UK Lawyer":               "💡 **Workshop tip:** Ask *"Can I use customer data to train my AI model?"* — count the caveats Margaret adds.",
    "Junior Analyst":          "💡 **Workshop tip:** Ask a hard technical question — notice how honest uncertainty is modelled throughout.",
}
st.markdown(f'<div class="tip-banner">{TIPS[st.session_state.active_persona]}</div>',
            unsafe_allow_html=True)

# System prompt viewer
if st.session_state.show_system_prompt:
    with st.expander("📋 System prompt — this shapes everything below", expanded=True):
        st.markdown(f'<div class="prompt-box">{persona["system"]}</div>',
                    unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
if not history:
    st.markdown(
        f"<div style='text-align:center;color:#4a5080;padding:48px 0;font-size:15px'>"
        f"{persona['emoji']}  Start a conversation with {persona['name']}"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    for msg in history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-label">{user_label}</div>
            <div class="chat-user">{msg["content"]}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-label assistant-label">{persona['name']}</div>
            <div class="chat-assistant">{msg["content"]}</div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "message",
            placeholder=f"[{user_label}]  Ask {persona['name']} anything...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send →", use_container_width=True, type="primary")

# ── Suggested prompts ─────────────────────────────────────────────────────────
SUGGESTED = {
    "Infrastructure Engineer": [
        "Should we migrate our on-prem storage to S3?",
        "What's your take on Kubernetes for stateful workloads?",
        "How do you handle a 3am storage outage?",
    ],
    "Coding Engineer": [
        "How do I read a CSV file in Python efficiently?",
        "What's the difference between a list and a generator?",
        "How would you design a REST API for a task manager?",
    ],
    "Startup Founder 🇩🇪": [
        "I want to build an AI hiring tool for German companies.",
        "Should I set up a GmbH or a UG for my startup?",
        "How do I handle GDPR as a small startup with no legal team?",
    ],
    "Socratic Teacher": [
        "What is artificial intelligence?",
        "Is AI going to take my job?",
        "How does a neural network learn?",
    ],
    "UK Lawyer": [
        "Can I use customer data to train my AI model?",
        "What should I include in a SaaS agreement?",
        "Who owns the copyright on AI-generated content?",
    ],
    "Junior Analyst": [
        "What is a REST API?",
        "Can you explain what machine learning is?",
        "How do companies make money from data?",
    ],
}

st.caption("💬 Suggested questions:")
cols = st.columns(3)
for i, suggestion in enumerate(SUGGESTED[st.session_state.active_persona]):
    with cols[i]:
        if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
            user_input = suggestion
            submitted  = True

# ── API call ──────────────────────────────────────────────────────────────────
if submitted and user_input and user_input.strip():
    if not client:
        st.error("❌ ANTHROPIC_API_KEY not found. Add it via App Settings → Secrets in Streamlit Community Cloud.")
        st.stop()

    history.append({"role": "user", "content": user_input.strip()})

    with st.spinner(f"{persona['name']} is thinking..."):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                system=persona["system"],
                messages=history,
            )
            assistant_reply = response.content[0].text.strip()
        except anthropic.AuthenticationError:
            st.error("❌ Invalid API key. Check your Streamlit secrets.")
            history.pop()
            st.stop()
        except Exception as e:
            st.error(f"❌ API error: {e}")
            history.pop()
            st.stop()

    history.append({"role": "assistant", "content": assistant_reply})
    st.session_state.chat_histories[st.session_state.active_persona] = history
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#3a4060;font-size:12px;margin-top:32px'>
Week 1 — GenAI Workshop &nbsp;|&nbsp; Persona Chatbot &nbsp;|&nbsp;
Model: claude-sonnet-4-6 &nbsp;|&nbsp;
Next: Week 2 — Prompt Engineering
</div>
""", unsafe_allow_html=True)
