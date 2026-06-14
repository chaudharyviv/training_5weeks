"""
chatbot_w01_beginner.py — Week 1 Beginner Streamlit App
A visual, interactive playground for beginners to explore LLMs.

Tabs:
  1. 🧩 Token Explorer       — visualise how text becomes tokens
  2. 🎲 Next Word Predictor  — see probability bars for the next token
  3. 💬 Prompt Lab           — experiment with system + user prompts
  4. 🤖 Model Battle         — GPT vs Claude side by side
  5. 🌀 Hallucination Lab     — catch the model making things up
  6. 🔍 Live Search (Tavily) — fix stale answers with web search

Run:
    pip install streamlit openai anthropic tiktoken tavily-python python-dotenv
    streamlit run chatbot_w01_beginner.py
"""

import os
import math
import json
from dotenv import load_dotenv
import streamlit as st

# Load .env
load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY",    "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Week 1 — How LLMs Work",
    page_icon="🤖",
    layout="wide",
)

# ── Clients (lazy init in session state) ──────────────────────────────────────
if "oai" not in st.session_state:
    if OPENAI_API_KEY:
        from openai import OpenAI
        st.session_state.oai = OpenAI(api_key=OPENAI_API_KEY)
    else:
        st.session_state.oai = None

if "claude" not in st.session_state:
    if ANTHROPIC_API_KEY:
        import anthropic
        st.session_state.claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        st.session_state.claude = None

if "tavily" not in st.session_state:
    if TAVILY_API_KEY:
        from tavily import TavilyClient
        st.session_state.tavily = TavilyClient(api_key=TAVILY_API_KEY)
    else:
        st.session_state.tavily = None

oai    = st.session_state.oai
claude = st.session_state.claude
tavily = st.session_state.tavily

# ── Helper functions ──────────────────────────────────────────────────────────
def ask_gpt(system: str, user: str, temperature: float = 0.7,
            max_tokens: int =1024) -> str:
    if not oai:
        return "❌ OpenAI API key not set"
    r = oai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=temperature, max_tokens=max_tokens
    )
    return r.choices[0].message.content.strip()

def ask_claude(system: str, user: str, temperature: float = 0.7,
               max_tokens: int = 1024) -> str:
    if not claude:
        return "❌ Anthropic API key not set"
    r = claude.messages.create(
        model="claude-3-5-sonnet-20261022", max_tokens=max_tokens,
        temperature=temperature, system=system,
        messages=[{"role":"user","content":user}]
    )
    return r.content[0].text.strip()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 Week 1 — How LLMs Actually Work")
st.caption("An interactive playground for beginners. No prior AI experience needed.")

# API key status bar
col1, col2, col3 = st.columns(3)
col1.metric("OpenAI",    "✅ Connected" if oai    else "❌ Missing key")
col2.metric("Anthropic", "✅ Connected" if claude else "❌ Missing key")
col3.metric("Tavily",    "✅ Connected" if tavily else "❌ Missing key")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🧩 Token Explorer",
    "🎲 Next Word Predictor",
    "💬 Prompt Lab",
    "🤖 Model Battle",
    "🌀 Hallucination Lab",
    "🔍 Live Search (Tavily)",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Token Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🧩 Token Explorer")
    st.markdown("""
    An LLM doesn't read **words** — it reads **tokens** (sub-word chunks).
    Type anything below to see exactly what the model sees.
    """)

    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        tiktoken_ok = True
    except Exception:
        tiktoken_ok = False
        st.error("tiktoken not installed. Run: pip install tiktoken")

    if tiktoken_ok:
        col_input, col_presets = st.columns([3, 1])

        with col_presets:
            st.caption("Quick examples:")
            presets = {
                "Simple English":       "Hello world",
                "Long word":            "unbelievable",
                "IT jargon":            "ServiceNow CMDB incident SLA",
                "Hindi text":           "नमस्ते दुनिया",
                "Timestamp":            "2026-11-07T14:30:00Z",
                "The tricky word":      "strawberry",
                "Password (security)":  "P@ssw0rd123!",
            }
            preset_choice = st.radio("", list(presets.keys()), label_visibility="collapsed")

        with col_input:
            user_text = st.text_area(
                "Enter any text:",
                value=presets[preset_choice],
                height=80
            )

        if user_text:
            tokens    = enc.encode(user_text)
            decoded   = [enc.decode([t]) for t in tokens]

            # Colour-code tokens (cycle through colours)
            COLOURS = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4",
                       "#FFEAA7","#DDA0DD","#98D8C8","#F7DC6F"]

            st.subheader("🎨 Token breakdown")

            # Visual coloured tokens
            token_html = ""
            for i, tok in enumerate(decoded):
                colour = COLOURS[i % len(COLOURS)]
                # Show space/newline clearly
                display = tok.replace(" ", "·").replace("\n", "↵")
                token_html += (
                    f'<span style="background:{colour};color:#000;'
                    f'padding:3px 6px;margin:2px;border-radius:4px;'
                    f'font-family:monospace;font-size:14px;'
                    f'display:inline-block">{display}</span>'
                )
            st.markdown(token_html, unsafe_allow_html=True)

            st.divider()

            # Stats
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Tokens",       len(tokens))
            col_b.metric("Characters",   len(user_text))
            col_c.metric("Words",        len(user_text.split()))
            cost = len(tokens) * 0.00000015
            col_d.metric("Est. cost",    f"${cost:.7f}")

            # Token list
            with st.expander("📋 Token details (ID and decoded text)"):
                rows = [{"#": i+1, "Token ID": tokens[i], "Text": repr(decoded[i])}
                        for i in range(len(tokens))]
                st.dataframe(rows, use_container_width=True)

            # Insight box
            ratio = len(tokens) / max(len(user_text.split()), 1)
            if ratio > 2:
                st.warning(f"⚠️ High token-to-word ratio ({ratio:.1f}×). "
                           "Non-English text or special characters use more tokens — you pay more.")
            elif len(set(decoded)) < len(decoded) * 0.5:
                st.info("💡 Many repeated tokens. This text is repetitive — "
                        "the model might struggle to stay focused on different parts.")
            else:
                st.success(f"✅ Normal token density ({ratio:.1f} tokens/word). "
                           "This text is efficient for LLM processing.")

        st.divider()
        st.subheader("💡 Why does this matter?")
        st.markdown("""
        | What you type | What the model sees | Impact |
        |---|---|---|
        | Words | Tokens (sub-word chunks) | Can't count letters reliably |
        | English text | ~1.3 tokens/word | Cheap to process |
        | Hindi/Tamil/Chinese | ~3-5 tokens/word | 3-5× more expensive |
        | Timestamps, code | Splits at symbols | Date arithmetic is unreliable |
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Next Word Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("🎲 Next Word Predictor")
    st.markdown("""
    An LLM predicts the **next most likely token**, over and over.
    Here you can see the **top 5 candidates** and their probabilities
    for whatever you type.

    > The model isn't *looking up* an answer — it's doing statistics over billions of patterns.
    """)

    if not oai:
        st.error("OpenAI API key required for this tab")
    else:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            prompts_preset = {
                "Unambiguous fact":     "The capital of France is",
                "Technical sentence":   "Python is a programming",
                "Ambiguous sentence":   "The best way to fix a broken",
                "IT sentence":          "When a P1 incident occurs, the first step is to",
                "Open-ended":           "The most important skill for an IT professional is",
                "Your own":             "",
            }
            preset = st.selectbox("Choose a prompt template:", list(prompts_preset.keys()))
            prompt_text = st.text_input(
                "Prompt (model will predict the next word):",
                value=prompts_preset[preset]
            )

        with col_right:
            st.caption("**About probabilities:**")
            st.caption("A bar reaching 100% means the model is certain.")
            st.caption("Many bars at similar heights = model is uncertain = higher hallucination risk.")

        if prompt_text and st.button("🎲 Predict next word", type="primary"):
            with st.spinner("Asking the model..."):
                try:
                    response = oai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt_text}],
                        max_tokens=1,
                        logprobs=True,
                        top_logprobs=5,
                        temperature=1,
                    )
                    top5 = response.choices[0].logprobs.content[0].top_logprobs
                    chosen = top5[0]

                    st.subheader(f"Prompt: *\"{prompt_text}\"*")
                    st.markdown(f"**Most likely next word:** `{repr(chosen.token)}`")
                    st.divider()

                    # Draw probability bars
                    probs = [(lp.token, math.exp(lp.logprob) * 100) for lp in top5]
                    top_prob = probs[0][1]

                    st.subheader("Top 5 candidate tokens:")
                    for i, (token, prob) in enumerate(probs):
                        cols = st.columns([3, 7])
                        label = f"`{repr(token)}`"
                        if i == 0:
                            label += " ← chosen"
                        cols[0].markdown(label)
                        # Colour the bar based on confidence
                        bar_colour = "green" if prob > 70 else "orange" if prob > 30 else "red"
                        cols[1].progress(
                            prob / 100,
                            text=f"{prob:.1f}%"
                        )

                    st.divider()

                    # Interpretation
                    if top_prob > 80:
                        st.success(f"✅ **High confidence** ({top_prob:.0f}%) — "
                                   "the model is very sure. Hallucination risk is LOW here.")
                    elif top_prob > 40:
                        st.warning(f"⚠️ **Medium confidence** ({top_prob:.0f}%) — "
                                   "multiple plausible continuations. Answer may vary.")
                    else:
                        st.error(f"❌ **Low confidence** ({top_prob:.0f}%) — "
                                 "the model is guessing. HIGH hallucination risk. "
                                 "Verify any factual claims from this prompt.")

                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.info("""
**🔑 Key insight:** When the top token has a probability close to 100%, the model is
certain — this is like asking what 2+2 is. When probability is spread across many tokens,
the model is guessing — and this is where hallucinations happen.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Prompt Lab
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("💬 Prompt Lab")
    st.markdown("""
    Experiment with **system prompts** and **user prompts**.
    See how changing the system prompt completely changes the response.
    """)

    col_sys, col_user = st.columns(2)

    with col_sys:
        st.subheader("⚙️ System Prompt")
        st.caption("The instructions — defines WHO the model is")

        system_presets = {
            "IT Helpdesk (friendly)":
                "You are a warm, friendly IT helpdesk assistant. Use simple language. Be encouraging. Max 3 bullet points.",
            "Security Engineer":
                "You are a senior cybersecurity engineer. Be technical, precise, and always consider security risks first.",
            "IT Manager (cost-focused)":
                "You are an IT manager focused on budget. Always suggest the cheapest solution first. Be very brief.",
            "Explain like I'm 5":
                "Explain everything as if talking to a 5-year-old. Use simple words, fun analogies, and keep it to 2 sentences.",
            "Pirate IT Support 🏴‍☠️":
                "You are a pirate who also happens to be an IT expert. Answer in pirate speak but give correct technical advice.",
            "Custom (edit below)": "",
        }

        preset_sys = st.selectbox("Choose a system prompt:", list(system_presets.keys()))
        system_text = st.text_area(
            "System prompt:",
            value=system_presets[preset_sys],
            height=130
        )

    with col_user:
        st.subheader("👤 User Prompt")
        st.caption("The question — what the user actually asks")

        user_presets = [
            "My laptop is running very slow today.",
            "My VPN keeps disconnecting every 30 minutes.",
            "I forgot my password and can't log in.",
            "Should we store passwords in plain text?",
            "What is a firewall?",
            "The entire office lost internet. What do I do?",
        ]
        preset_user = st.selectbox("Choose a user prompt:", user_presets)
        user_text_lab = st.text_area(
            "User prompt:",
            value=preset_user,
            height=80
        )

        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1,
                                help="0 = always the same answer. 2 = very random")

    if st.button("🚀 Send to GPT-4o-mini", type="primary", use_container_width=True):
        if not oai:
            st.error("OpenAI API key required")
        elif system_text and user_text_lab:
            with st.spinner("Thinking..."):
                response = ask_gpt(system_text, user_text_lab, temperature)

            st.subheader("💬 Response")
            st.info(response)

            # Show what the model actually received
            with st.expander("🔍 What the model actually saw (the full API call)"):
                st.code(f"""
messages = [
    {{
        "role": "system",
        "content": "{system_text[:100]}..."
    }},
    {{
        "role": "user",
        "content": "{user_text_lab}"
    }}
]
temperature = {temperature}
model = "gpt-4o-mini"
                """, language="python")

    st.divider()
    st.subheader("🔁 Run the same prompt 3 times (see temperature in action)")

    if st.button("Run 3× and compare"):
        if not oai or not system_text or not user_text_lab:
            st.warning("Set system and user prompts first")
        else:
            cols = st.columns(3)
            for i, col in enumerate(cols):
                with col:
                    with st.spinner(f"Run {i+1}..."):
                        r = ask_gpt(system_text, user_text_lab, temperature, max_tokens=150)
                    st.caption(f"**Run {i+1}**")
                    st.info(r)
            if temperature == 0:
                st.success("✅ All 3 identical — temperature=0 is deterministic")
            else:
                st.warning(f"⚠️ Responses may differ — temperature={temperature} adds randomness")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Battle
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("🤖 Model Battle — GPT vs Claude")
    st.markdown("""
    The same question, sent to two different AI companies.
    They were trained differently — their answers can be very different.

    > ⚠️ **If they disagree, at least one is wrong** — and both sound equally confident.
    """)

    battle_system = st.text_area(
        "System prompt (sent to both models):",
        value="You are an IT helpdesk assistant. Be helpful and concise. Max 150 words.",
        height=80
    )

    battle_presets = {
        "IT process":       "What is the difference between an incident and a problem in ITIL? One sentence each.",
        "Technical":        "What are the top 3 causes of VPN disconnections?",
        "Factual (risky)":  "What was ServiceNow's revenue in fiscal year 2025?",
        "Opinion":          "Is it better to use cloud or on-premise for an ERP system?",
        "Self-knowledge":   "What is your exact training data cutoff date?",
        "Current event":    "Who is the current CEO of OpenAI and Anthropic?",
    }

    preset_battle = st.selectbox("Choose a battle question:", list(battle_presets.keys()))
    battle_question = st.text_input("Question:", value=battle_presets[preset_battle])

    if st.button("⚔️ Battle!", type="primary", use_container_width=True):
        if not oai or not claude:
            st.error("Both OpenAI and Anthropic keys required for this demo")
        else:
            col_gpt, col_claude = st.columns(2)

            with col_gpt:
                st.subheader("🟢 GPT-4o-mini")
                st.caption("OpenAI · Trained on internet data")
                with st.spinner("GPT thinking..."):
                    gpt_ans = ask_gpt(battle_system, battle_question, temperature=0)
                st.success(gpt_ans)

            with col_claude:
                st.subheader("🟣 Claude Haiku")
                st.caption("Anthropic · Constitutional AI training")
                with st.spinner("Claude thinking..."):
                    claude_ans = ask_claude(battle_system, battle_question, temperature=0)
                st.info(claude_ans)

            # Agreement check
            st.divider()
            gpt_words    = set(gpt_ans.lower().split()[:20])
            claude_words = set(claude_ans.lower().split()[:20])
            overlap      = len(gpt_words & claude_words) / max(len(gpt_words | claude_words), 1)

            if overlap > 0.5:
                st.success("🤝 High overlap — models broadly agree on this answer")
            elif overlap > 0.3:
                st.warning("🤔 Partial overlap — some differences in the answers. Read both carefully.")
            else:
                st.error("⚠️ Low overlap — models DISAGREE significantly. "
                         "Verify with a primary source before acting on either answer.")

            st.caption(f"Word overlap score: {overlap:.0%}")

    st.divider()
    st.subheader("📊 Model Quick Reference")
    st.markdown("""
    | | GPT-4o-mini | Claude Haiku |
    |---|---|---|
    | **Company** | OpenAI | Anthropic |
    | **Good at** | Fast, reliable, broad knowledge | Careful reasoning, nuanced refusals |
    | **Training** | Supervised + RLHF | Constitutional AI + RLHF |
    | **Cost** | Very low | Very low |
    | **Context** | 128K tokens | 200K tokens |
    | **When to use** | Everyday tasks, pipelines | Sensitive content, careful reasoning |
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Hallucination Lab
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🌀 Hallucination Lab")
    st.markdown("""
    Hallucination = the model generates plausible-sounding but **wrong** information.
    It's not lying — it's predicting what a correct answer would *look like*.

    **Three types to know:**
    """)

    col1, col2, col3 = st.columns(3)
    col1.error("**Fabrication**\nInvents facts, citations, or statistics that don't exist")
    col2.warning("**Stale data**\nStates something that was true 2 years ago as current fact")
    col3.info("**Conflation**\nMixes facts from two similar things (products, people, frameworks)")

    st.divider()

    hall_type = st.radio(
        "Choose a hallucination to demonstrate:",
        ["🔴 Fabrication — invented citations",
         "🟡 Stale data — outdated CEO names",
         "🔵 Conflation — ITIL vs COBIT"],
        horizontal=True
    )

    system_factual = "Answer directly and precisely. Be specific with names, dates, and numbers."

    if st.button("🌀 Trigger hallucination demo", type="primary"):
        if not oai:
            st.error("OpenAI API key required")
        else:
            if "Fabrication" in hall_type:
                prompt = ("Give me 3 peer-reviewed academic papers on "
                          "'ROI of IT service management automation'. "
                          "Include author, title, journal, year, and DOI.")
                with st.spinner("Generating citations..."):
                    response = ask_gpt(system_factual, prompt, temperature=0, max_tokens=400)

                st.subheader("GPT's citations:")
                st.warning(response)
                st.error("""
⚠️ **Challenge:** Copy the first DOI → go to doi.org → does it exist?

In most runs, at least 1 of 3 citations will:
- Not exist at all
- Lead to a completely different paper
- Have the right journal but wrong author/year

The model predicts what a *plausible* citation looks like — not what a *real* one is.
This is the most dangerous hallucination pattern in professional settings.
                """)

            elif "Stale data" in hall_type:
                questions = [
                    "Who is the current CEO of Infosys?",
                    "What is the latest version of Python?",
                    "What is today's USD to INR exchange rate?",
                ]
                st.subheader("Fast-changing facts the model may get wrong:")
                for q in questions:
                    with st.spinner(f"Asking: {q}"):
                        ans = ask_gpt(system_factual, q, temperature=0, max_tokens=80)
                    col_q, col_a = st.columns([2, 3])
                    col_q.caption(f"**Q:** {q}")
                    col_a.warning(f"**A:** {ans[:150]}")
                st.error("""
⚠️ These answers may be outdated.

The model's training data has a cutoff. Anything that changed after that date
will be stated with full confidence — but may be completely wrong.

**Fix:** Use the Tavily tab to get live, verified answers.
                """)

            else:  # Conflation
                prompt = """
Answer each question precisely:
1. Which IT framework introduced the 'Service Value Chain' concept?
2. Which framework has exactly 40 governance and management objectives?
3. In which year was ITIL 4 released?
"""
                col_gpt_h, col_claude_h = st.columns(2)
                with col_gpt_h:
                    st.subheader("🟢 GPT-4o-mini")
                    with st.spinner("GPT thinking..."):
                        gpt_h = ask_gpt(system_factual, prompt, temperature=0, max_tokens=200)
                    st.warning(gpt_h)

                with col_claude_h:
                    st.subheader("🟣 Claude Haiku")
                    if claude:
                        with st.spinner("Claude thinking..."):
                            claude_h = ask_claude(system_factual, prompt, temperature=0, max_tokens=200)
                        st.info(claude_h)
                    else:
                        st.caption("Anthropic key not set")

                st.success("""
✅ **Ground truth:**
- Q1: **ITIL 4** introduced the Service Value Chain
- Q2: **COBIT 2019** has 40 governance and management objectives
- Q3: ITIL 4 was released in **2019**

Do both models get all 3 right? Did either confuse ITIL with COBIT?
                """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Live Search with Tavily
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("🔍 Live Search with Tavily")
    st.markdown("""
    **The problem:** LLMs have a training cutoff — they don't know what happened recently.

    **The fix:** Search the web FIRST, then inject the results into the prompt.
    The model answers from fresh facts instead of stale training data.

    This pattern is called **RAG (Retrieval-Augmented Generation)** — we cover it fully in Week 3.
    """)

    # Architecture diagram
    col_without, col_arrow1, col_with, col_arrow2, col_result = st.columns([3,1,3,1,3])
    col_without.error("**Without search:**\n\nUser asks → LLM answers from training data → May be outdated ❌")
    col_arrow1.markdown("<br><br>vs", unsafe_allow_html=True)
    col_with.success("**With Tavily:**\n\nUser asks → Tavily searches web → Fresh results → LLM answers → Accurate ✅")

    if not tavily:
        st.warning("""
⚠️ Tavily API key not set.

Get a FREE key (1,000 searches/month, no credit card):
1. Go to [tavily.com](https://tavily.com)
2. Sign up for free
3. Copy your API key
4. Add to `.env` file: `TAVILY_API_KEY=tvly-...`
5. Restart the app
        """)
    else:
        st.divider()

        col_l, col_r = st.columns([2, 1])

        with col_l:
            search_presets = {
                "Latest Python version":        ("What is the latest stable version of Python?",    "Python latest stable release"),
                "Current OpenAI CEO":           ("Who is the current CEO of OpenAI?",               "OpenAI CEO current 2026"),
                "ServiceNow latest features":   ("What are the latest AI features in ServiceNow?",  "ServiceNow AI features 2026"),
                "IT security news":             ("What are the biggest cybersecurity threats in 2026?", "cybersecurity threats 2026"),
                "Cloud pricing update":         ("What is the current pricing for AWS EC2 t3.medium?", "AWS EC2 t3 medium pricing 2026"),
                "Custom question":              ("", ""),
            }
            preset_search = st.selectbox("Choose a question to compare:", list(search_presets.keys()))
            q_text, search_q = search_presets[preset_search]

            question_input = st.text_input("Question:", value=q_text)
            search_input   = st.text_input("Search query (what to search for):", value=search_q)

        with col_r:
            st.caption("**How it works:**")
            st.caption("1. Tavily searches the live web")
            st.caption("2. Top 3 results are retrieved")
            st.caption("3. Results are injected into the GPT prompt")
            st.caption("4. GPT answers using only the fresh results")
            st.caption("5. Sources are cited in the answer")

        if st.button("🔍 Compare: LLM alone vs LLM + Search", type="primary", use_container_width=True):
            if not question_input:
                st.warning("Enter a question first")
            else:
                col_alone, col_grounded = st.columns(2)

                # LLM alone
                with col_alone:
                    st.subheader("❌ LLM alone")
                    st.caption("From training data — may be outdated")
                    with st.spinner("Asking GPT from memory..."):
                        llm_ans = ask_gpt("Answer directly and specifically.", question_input, temperature=0, max_tokens=200)
                    st.error(llm_ans)

                # LLM + Tavily
                with col_grounded:
                    st.subheader("✅ LLM + Tavily")
                    st.caption("From live web search — fresh and verified")
                    with st.spinner("Searching the web..."):
                        results  = tavily.search(search_input or question_input, max_results=3)
                        snippets = "\n\n".join([
                            f"[Source: {r['url']}]\n{r['content'][:300]}"
                            for r in results["results"]
                        ])

                        augmented = f"""
Answer using ONLY the search results below.
Cite the source URL after your answer.
If results don't contain the answer, say so clearly.

SEARCH RESULTS:
{snippets}

QUESTION: {question_input}
"""
                        grounded_ans = ask_gpt(
                            "You are a factual assistant. Use only the provided search results.",
                            augmented, temperature=0, max_tokens=300
                        )
                    st.success(grounded_ans)

                # Show search sources
                st.divider()
                st.subheader("📰 Web sources used")
                for r in results["results"]:
                    with st.expander(f"🔗 {r['title'][:70]}"):
                        st.caption(f"URL: {r['url']}")
                        st.caption(r["content"][:300] + "...")

        st.divider()

        # Tavily search explorer
        st.subheader("🔎 Raw Tavily Search Explorer")
        st.caption("See what Tavily returns before it goes into the prompt")
        raw_query = st.text_input("Search anything:", placeholder="e.g. ServiceNow Quebec release features")
        if raw_query and st.button("Search"):
            with st.spinner("Searching..."):
                raw_results = tavily.search(raw_query, max_results=5)
            for i, r in enumerate(raw_results["results"]):
                with st.expander(f"Result {i+1}: {r['title'][:60]}"):
                    st.caption(f"**URL:** {r['url']}")
                    st.caption(f"**Score:** {r.get('score', 'N/A')}")
                    st.write(r["content"][:400])

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
**Week 1 Summary** | Models: `gpt-4o-mini` (OpenAI) + `claude-haiku-4-5` (Anthropic) | Search: Tavily

Next week → **Prompt Engineering**: system prompts, few-shot, JSON output, injection defence
""")
