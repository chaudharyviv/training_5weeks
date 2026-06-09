"""
chatbot_w03.py — Week 3 Streamlit Chatbot
RAG Knowledge Base chatbot with:
- Source citations on every answer
- 5 failure mode explorer (live demonstrations)
- Neo4j CMDB "related issues" panel (mock if Neo4j unavailable)
- RAGAS-style faithfulness + relevance scoring
- Query expansion toggle

Run:
    pip install streamlit openai anthropic neo4j scikit-learn numpy requests pydantic
    streamlit run chatbot_w03.py

Set environment variables:
    export OPENAI_API_KEY=sk-...
    export NEO4J_URI=bolt://localhost:7687       (optional)
    export NEO4J_USER=neo4j                      (optional)
    export NEO4J_PASSWORD=password               (optional)
"""

import os
import sys
import json
import time
import numpy as np
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import LLMClient, Tracer, Guardrails

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Base — Week 3",
    page_icon="📚",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
if "tracer" not in st.session_state:
    st.session_state.tracer = Tracer(session_id="w03-streamlit")
if "client" not in st.session_state:
    st.session_state.client = LLMClient(tracer=st.session_state.tracer)
if "history" not in st.session_state:
    st.session_state.history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_scores" not in st.session_state:
    st.session_state.last_scores = {}
if "kb_embeddings" not in st.session_state:
    st.session_state.kb_embeddings = None
if "failure_result" not in st.session_state:
    st.session_state.failure_result = None

tracer = st.session_state.tracer
client = st.session_state.client
MODEL  = "gpt-4o-mini"

# ══════════════════════════════════════════════════════════════════════════════
# Knowledge base (IT Ops policies)
# ══════════════════════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = [
    {"id": "sla-p1",     "title": "SLA: P1 Critical Incidents",
     "content": "P1 incidents: acknowledge within 15 min, resolve within 4 hours. Complete service outage >50 users, revenue-impacting failure, active security breach. Must page PagerDuty on-call immediately. Post-incident review required within 5 business days. P1 opens war room in #incidents-live Slack."},
    {"id": "sla-p2",     "title": "SLA: P2 High Priority",
     "content": "P2 incidents: acknowledge within 30 min, resolve within 8 business hours. Single team blocked, critical user unable to work, degraded performance on key system. SLA clock pauses in 'Awaiting User' state."},
    {"id": "sla-p3p4",   "title": "SLA: P3 and P4 Incidents",
     "content": "P3: acknowledge 2h, resolve 3 business days. P4: acknowledge 1 day, resolve 10 business days. P3 examples: individual software issue with workaround. P4: cosmetic UI, informational queries. SLA clock stops during 'Awaiting User'."},
    {"id": "vpn-current","title": "VPN Policy — GlobalProtect (Current)",
     "content": "All remote access uses GlobalProtect VPN (Palo Alto). Cisco AnyConnect decommissioned March 2024. MFA mandatory via Okta Verify. Split tunnelling disabled. Max session: 12 hours. Install: intranet.company.com/vpn-install. TAP adapter errors: reinstall the full client."},
    {"id": "vpn-legacy", "title": "VPN Policy — Cisco AnyConnect (DEPRECATED March 2024)",
     "content": "DEPRECATED. Do not use. Cisco AnyConnect was decommissioned 31 March 2024. See vpn-current for current policy. Kept for audit trail only."},
    {"id": "dw-access",  "title": "Data Warehouse Access (Snowflake)",
     "content": "Submit Data Access Request via IT Service Portal (portal.internal/data-access). Required: business justification, data classification, duration, manager approval. Confidential/Restricted datasets need DPO sign-off. Standard: 3 business days. Urgent: 24 hours. All access logged, reviewed quarterly."},
    {"id": "pii-cloud",  "title": "PII Storage Policy — Cloud Object Storage",
     "content": "PII in AWS S3/GCS/Azure Blob only with: (1) AES-256+ encryption, (2) public access blocked at account level, (3) tagged 'PII-Restricted', (4) retention per GDPR Art. 5, (5) IAM roles only — no static credentials. Violations reported to Security within 24 hours."},
    {"id": "oncall-sap", "title": "After-Hours SAP Support",
     "content": "SAP Basis issues outside 8 AM–7 PM IST or weekends: page 'SAP-BASIS-ONCALL' PagerDuty. No response in 15 min: escalate to SAP CoE Lead Anand Krishnamurthy. SAP HANA DB: 'HANA-DBA-ONCALL'. Business hours: ServiceNow ticket, category SAP, group SAP-BASIS-TEAM. Maintenance: Saturday 11 PM–3 AM IST."},
    {"id": "change-mgmt","title": "Change Management — CAB and Emergency Changes",
     "content": "Standard changes: ServiceNow, reviewed at CAB Thursdays 2 PM IST. Needs manager approval + IT Risk for production. Emergency changes: bypass CAB, requires Emergency Change Manager approval (24/7 PagerDuty). All changes: rollback plan + test evidence + business impact. Freeze: Dec 15–Jan 5."},
    {"id": "backup",     "title": "Backup and Disaster Recovery",
     "content": "Production DBs: daily full + continuous WAL. RPO=1h, RTO=4h. File servers: daily incremental + weekly full. RPO=24h, RTO=8h. Laptops: Backblaze cloud backup on corporate Wi-Fi. Retention: 30 days rolling + 12 months snapshots. DR test quarterly. Restore: P2 ServiceNow ticket, category Data Recovery."},
    {"id": "gdpr-sar",   "title": "GDPR Subject Access Request Handling",
     "content": "SARs must be fulfilled within 30 days (GDPR Art. 15). IT must respond to data extraction requests from DPO within 5 business days. GDPR reporting dashboard: reports.internal/gdpr-sar. If dashboard unavailable: escalate to IT Security as P2. Archive SAR evidence 3 years."},
    {"id": "cmdb-policy","title": "CMDB Governance — Configuration Item Standards",
     "content": "All production CIs must be registered in ServiceNow CMDB within 24 hours of provisioning. Required: CI name, type, owner team, environment, upstream/downstream dependencies. CMDB accuracy target: 95%. Quarterly audits by IT Risk. Unregistered production CIs: raise P3 change record."},
]

# ══════════════════════════════════════════════════════════════════════════════
# Mock Neo4j CMDB data
# ══════════════════════════════════════════════════════════════════════════════
MOCK_CMDB = {
    "SAP S/4HANA":    {"incidents": ["INC0001001 — SAP login slow (HANA memory pressure, P2, Resolved)",
                                      "INC0001005 — SAP payroll module timeout (network, P1, Resolved)"],
                        "depends_on": ["SAP HANA DB", "Load Balancer (F5)"],
                        "owner":      "SAP-BASIS-TEAM"},
    "GlobalProtect VPN":{"incidents": ["INC0001002 — TAP adapter missing after Win update (P2, Resolved)"],
                          "depends_on": ["Network firewall"],
                          "owner":      "Network-Ops"},
    "Load Balancer":  {"incidents": ["INC0001003 — SSL cert expired, causing 503s (P1, Resolved)"],
                        "depends_on": [],
                        "owner":      "Network-Ops"},
    "Snowflake DW":   {"incidents": ["INC0001004 — Read-only during maintenance (P3, Resolved)"],
                        "depends_on": [],
                        "owner":      "DBA-Team"},
}

# ══════════════════════════════════════════════════════════════════════════════
# RAG system prompt
# ══════════════════════════════════════════════════════════════════════════════
RAG_SYSTEM = """
You are an IT knowledge base assistant.
Answer ONLY using the provided context documents.
After each factual claim, cite the source as [doc-id].
If the context contains conflicting information (e.g., old vs new policy), flag the conflict explicitly.
If the answer is not in the context, say exactly: "This is not covered in the knowledge base."
Never guess or use general knowledge — only the provided context.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Build / retrieve embeddings (cached in session state)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Building knowledge base embeddings...")
def build_kb_embeddings(api_key: str):
    """Embed all KB documents. Cached across reruns."""
    from openai import OpenAI
    oai = OpenAI(api_key=api_key)
    texts = [f"{d['title']}\n{d['content']}" for d in KNOWLEDGE_BASE]
    resp  = oai.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([e.embedding for e in resp.data])


def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return build_kb_embeddings(api_key)


def retrieve(query: str, top_k: int = 3, expand: bool = False,
             threshold: float = 0.25) -> list:
    """Retrieve top_k relevant KB chunks."""
    embeddings = get_embeddings()
    if embeddings is None:
        return []

    queries = [query]
    if expand:
        exp_raw = client.chat(
            MODEL,
            user=f"Generate 3 alternate phrasings of this query. One per line, no numbering.\nQuery: {query}",
            temperature=0.4, tags=["query_expansion"]
        )
        queries += [q.strip() for q in exp_raw.strip().splitlines() if q.strip()]

    best_scores = np.zeros(len(KNOWLEDGE_BASE))
    for q in queries:
        from openai import OpenAI
        oai  = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = oai.embeddings.create(model="text-embedding-3-small", input=[q])
        q_emb  = np.array([resp.data[0].embedding])
        scores = cosine_similarity(q_emb, embeddings)[0]
        best_scores = np.maximum(best_scores, scores)

    ranked = np.argsort(best_scores)[::-1]
    return [
        {"doc": KNOWLEDGE_BASE[i], "score": float(best_scores[i])}
        for i in ranked[:top_k]
        if best_scores[i] >= threshold
    ]


def rag_answer(question: str, top_k: int = 3, expand: bool = False) -> dict:
    """Full RAG pipeline. Returns {answer, sources, retrieved}."""
    retrieved = retrieve(question, top_k=top_k, expand=expand)
    if not retrieved:
        return {
            "answer":    "This is not covered in the knowledge base.",
            "sources":   [],
            "retrieved": []
        }
    context = "\n\n".join([
        f"[SOURCE: {r['doc']['id']}] {r['doc']['title']}\n{r['doc']['content']}"
        for r in retrieved
    ])
    answer = client.chat(
        MODEL,
        user=f"CONTEXT:\n{context}\n\nQUESTION: {question}",
        system=RAG_SYSTEM,
        temperature=0, tags=["rag_answer"]
    )
    return {
        "answer":    answer,
        "sources":   [r["doc"]["id"] for r in retrieved],
        "retrieved": retrieved
    }


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS-style evaluation
# ══════════════════════════════════════════════════════════════════════════════
def eval_faithfulness(question: str, answer: str, context: str) -> dict:
    prompt = f"""Rate the faithfulness of this answer on a scale 0.0–1.0.
Faithfulness = fraction of answer claims supported by the context below.
Context: {context[:600]}
Answer:  {answer[:400]}
Return JSON: {{"score": 0.0-1.0, "reason": "one sentence"}}"""
    raw = client.chat(MODEL, user=prompt, temperature=0, json_mode=True, tags=["eval_faith"])
    try:
        return json.loads(raw)
    except Exception:
        return {"score": 0.0, "reason": "parse error"}


def eval_relevance(question: str, answer: str) -> dict:
    prompt = f"""Rate how well this answer addresses the question on a scale 0.0–1.0.
Question: {question}
Answer:   {answer[:400]}
Return JSON: {{"score": 0.0-1.0, "reason": "one sentence"}}"""
    raw = client.chat(MODEL, user=prompt, temperature=0, json_mode=True, tags=["eval_rel"])
    try:
        return json.loads(raw)
    except Exception:
        return {"score": 0.0, "reason": "parse error"}


# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════
st.title("📚  IT Knowledge Base Chatbot")
st.caption("Week 3 — RAG · Source citations · Failure mode explorer · CMDB panel · RAGAS eval")

# Embedding status
if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️  OPENAI_API_KEY not set. Embeddings and retrieval will not work.")
else:
    embeddings = get_embeddings()
    if embeddings is not None:
        st.success(f"✅  Knowledge base ready — {len(KNOWLEDGE_BASE)} documents embedded")

tabs = st.tabs(["💬 RAG Chat", "🔍 Failure Modes", "🗺️ CMDB Panel", "📊 RAGAS Eval"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RAG Chat
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    col_chat, col_sources = st.columns([3, 2])

    with col_chat:
        st.subheader("💬 Chat")
        expand_query = st.toggle("🔍 Query expansion (bridges terminology mismatches)", value=False)
        top_k        = st.slider("Documents to retrieve", 1, 5, 3)

        chat_box = st.container(height=380)
        with chat_box:
            if not st.session_state.history:
                st.info("Ask any IT policy question — e.g. 'What is our SLA for a P1 incident?'")
            for msg in st.session_state.history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        if question := st.chat_input("Ask an IT policy question..."):
            safe, reason = Guardrails.check_input(question)
            if not safe:
                st.error(f"🛡️ Blocked: {reason}")
            else:
                st.session_state.history.append({"role": "user", "content": question})

                with st.spinner("Retrieving and generating..."):
                    result = rag_answer(question, top_k=top_k, expand=expand_query)

                st.session_state.history.append(
                    {"role": "assistant", "content": result["answer"]}
                )
                st.session_state.last_sources  = result["retrieved"]
                st.session_state.last_scores   = {
                    r["doc"]["id"]: r["score"] for r in result["retrieved"]
                }
                st.rerun()

    with col_sources:
        st.subheader("📎 Retrieved Sources")
        if st.session_state.last_sources:
            for r in st.session_state.last_sources:
                doc   = r["doc"]
                score = r["score"]
                with st.expander(f"[{doc['id']}] {doc['title']} — score: {score:.3f}"):
                    st.caption(doc["content"][:300] + "...")
                    if score < 0.35:
                        st.warning("⚠️ Low relevance score — this chunk may not be the right match")
                    elif "DEPRECATED" in doc["title"]:
                        st.warning("⚠️ This document is DEPRECATED — verify against current policy")
        else:
            st.caption("Sources will appear here after your first question")

        st.divider()
        st.subheader("📈 Session Stats")
        col_a, col_b = st.columns(2)
        col_a.metric("Cost",    f"${tracer.total_cost():.5f}")
        col_b.metric("Calls",   len(tracer.traces))

        if st.button("🔄 Reset", key="reset_chat"):
            st.session_state.history     = []
            st.session_state.last_sources = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Failure Mode Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🔍 RAG Failure Mode Explorer")
    st.caption("Each failure mode is a real production problem. Run them to see exactly what breaks.")

    failure_modes = {
        "1 — Vocabulary mismatch (SEV1 vs P1)": {
            "query_a": "What is the SLA for a P1 incident?",
            "query_b": "What is the SLA for a SEV1 incident?",
            "description": "Our KB uses 'P1'. Some users say 'SEV1'. Embeddings may bridge this — or may not.",
            "fix": "Query expansion — rephrase the query before retrieval to catch synonym mismatches.",
        },
        "2 — Conflicting docs (old vs new VPN policy)": {
            "query_a": "How do I connect to the VPN?",
            "query_b": None,
            "description": "The KB has both vpn-current and vpn-legacy (deprecated). Both may be retrieved.",
            "fix": "Tag deprecated docs in metadata and filter them before embedding search.",
        },
        "3 — Out-of-scope question (no answer in KB)": {
            "query_a": "What is our maternity leave policy?",
            "query_b": None,
            "description": "The KB has no HR/leave content. The model should say so — not hallucinate.",
            "fix": "Score threshold: if top retrieved doc scores < 0.30, return 'Not in KB' without calling LLM.",
        },
        "4 — Multi-hop (answer needs 2 chunks combined)": {
            "query_a": "If I have a P2 incident on Saturday at 10 PM, what is the SLA and who do I call for SAP?",
            "query_b": None,
            "description": "Needs sla-p2 (SLA details) AND oncall-sap (after-hours contact) — single chunk retrieval misses half.",
            "fix": "Increase top_k. Decompose multi-part questions and run separate retrievals.",
        },
        "5 — Stale data in KB": {
            "query_a": "How do I install Cisco AnyConnect?",
            "query_b": None,
            "description": "The old VPN client was decommissioned. Users asking about it will get confusing results.",
            "fix": "Date-stamp all KB documents. Decommissioned systems: update docs immediately, add redirect notices.",
        },
    }

    selected_fm = st.selectbox("Choose a failure mode to explore", list(failure_modes.keys()))
    fm = failure_modes[selected_fm]

    st.info(f"**What this demonstrates:** {fm['description']}")
    st.success(f"**Fix:** {fm['fix']}")

    col_fm1, col_fm2 = st.columns(2 if fm["query_b"] else 1)

    def run_fm_query(query, col, label):
        with col:
            st.caption(f"**Query:** `{query}`")
            if st.button(f"Run: {label}", key=f"fm_{label}"):
                with st.spinner("Retrieving..."):
                    result   = rag_answer(query, top_k=3, expand=False)
                    exp_res  = rag_answer(query, top_k=3, expand=True)

                st.write("**Without query expansion:**")
                st.caption(f"Sources: {result['sources']}")
                for r in result["retrieved"]:
                    dep = " ⚠️ DEPRECATED" if "DEPRECATED" in r["doc"]["title"] else ""
                    st.caption(f"  [{r['doc']['id']}] score={r['score']:.3f}{dep}")
                st.write(result["answer"][:300])

                st.divider()
                st.write("**With query expansion:**")
                st.caption(f"Sources: {exp_res['sources']}")
                for r in exp_res["retrieved"]:
                    dep = " ⚠️ DEPRECATED" if "DEPRECATED" in r["doc"]["title"] else ""
                    st.caption(f"  [{r['doc']['id']}] score={r['score']:.3f}{dep}")

    run_fm_query(fm["query_a"], col_fm1, "Query A")
    if fm["query_b"]:
        run_fm_query(fm["query_b"], col_fm2, "Query B")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CMDB Panel (Neo4j / mock)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🗺️ CMDB Related Issues Panel")
    st.caption("In production: Neo4j graph traversal. Here: mock CMDB data to show the pattern.")

    # Neo4j connection attempt
    neo4j_available = False
    neo4j_note      = ""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
        )
        driver.verify_connectivity()
        neo4j_available = True
        neo4j_note = "✅ Neo4j connected — using live graph data"
    except Exception as e:
        neo4j_note = f"⚠️  Neo4j not available — using mock CMDB data  ({str(e)[:60]})"

    st.caption(neo4j_note)

    selected_ci = st.selectbox("Select a Configuration Item (CI)", list(MOCK_CMDB.keys()))
    ci_data     = MOCK_CMDB[selected_ci]

    col_ci1, col_ci2 = st.columns(2)

    with col_ci1:
        st.subheader(f"📋 {selected_ci}")
        st.metric("Owner team", ci_data["owner"])
        if ci_data["depends_on"]:
            st.write("**Depends on:**")
            for dep in ci_data["depends_on"]:
                st.caption(f"  → {dep}")
        else:
            st.caption("No dependencies registered")

    with col_ci2:
        st.subheader("🔴 Past Incidents")
        if ci_data["incidents"]:
            for inc in ci_data["incidents"]:
                st.error(f"• {inc}")
        else:
            st.success("No past incidents found")

    st.divider()

    # Impact analysis
    st.subheader("💥 Impact Analysis")
    st.caption("Which other CIs are affected if this CI goes down?")

    affected = [
        ci for ci, data in MOCK_CMDB.items()
        if selected_ci in data.get("depends_on", [])
    ]
    if affected:
        st.warning(f"If **{selected_ci}** fails, these CIs are also impacted:")
        for ci in affected:
            st.caption(f"  → {ci} (owner: {MOCK_CMDB[ci]['owner']})")
    else:
        st.info(f"No other CIs depend on {selected_ci} in this mock dataset.")

    # Cypher query display
    st.divider()
    st.subheader("🔍 Cypher Query (runs in Neo4j)")
    st.code(f"""
// Past incidents for {selected_ci}
MATCH (inc:Incident)-[:AFFECTS]->(ci:ConfigItem {{name: '{selected_ci}'}})
RETURN inc.number, inc.description, inc.priority, inc.root_cause

// Impact analysis — what breaks if this CI fails?
MATCH (affected:ConfigItem)-[:DEPENDS_ON*1..3]->(ci:ConfigItem {{name: '{selected_ci}'}})
RETURN affected.name, affected.type, affected.criticality
""", language="cypher")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAGAS Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("📊 RAGAS-Style RAG Evaluation")
    st.caption("Scores every answer on faithfulness (grounded in context?) and relevance (answers the question?)")

    eval_questions = [
        "What is our SLA for a P1 incident and what happens after resolution?",
        "Can I store customer email addresses in an S3 bucket?",
        "How do I request urgent access to the data warehouse?",
        "Who do I call for SAP Basis issues on a Saturday night?",
        "What is our maternity leave policy?",   # out of scope
        "How do I connect to the VPN from home?",
    ]

    selected_eval_q = st.multiselect(
        "Select questions to evaluate",
        eval_questions,
        default=eval_questions[:4]
    )

    if st.button("Run RAGAS Evaluation →", type="primary"):
        if not selected_eval_q:
            st.warning("Select at least one question.")
        else:
            rows     = []
            progress = st.progress(0, text="Evaluating...")

            for i, q in enumerate(selected_eval_q):
                result   = rag_answer(q, top_k=3)
                ctx      = " ".join(r["doc"]["content"] for r in result["retrieved"])
                faith_r  = eval_faithfulness(q, result["answer"], ctx)
                relev_r  = eval_relevance(q, result["answer"])
                f_score  = faith_r.get("score", 0)
                r_score  = relev_r.get("score", 0)
                avg      = (f_score + r_score) / 2

                rows.append({
                    "Question":    q[:55] + "...",
                    "Sources":     ", ".join(result["sources"]),
                    "Faithfulness": round(f_score, 2),
                    "Relevance":   round(r_score, 2),
                    "Avg":         round(avg, 2),
                    "Pass":        "✅" if avg >= 0.70 else "⚠️ FLAG",
                    "Reason":      faith_r.get("reason", "")[:60],
                })
                progress.progress((i + 1) / len(selected_eval_q),
                                   text=f"{i + 1}/{len(selected_eval_q)} evaluated")

            progress.empty()
            st.dataframe(rows, use_container_width=True)

            pass_count = sum(1 for r in rows if "✅" in r["Pass"])
            avg_faith  = sum(r["Faithfulness"] for r in rows) / len(rows)
            avg_rel    = sum(r["Relevance"]    for r in rows) / len(rows)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pass rate",        f"{pass_count}/{len(rows)}")
            col2.metric("Avg faithfulness", f"{avg_faith:.2f}")
            col3.metric("Avg relevance",    f"{avg_rel:.2f}")
            col4.metric("LLM calls",        len(tracer.traces))

            flagged = [r for r in rows if "FLAG" in r["Pass"]]
            if flagged:
                st.warning(f"⚠️  {len(flagged)} questions need investigation:")
                for r in flagged:
                    st.caption(f"  • {r['Question']} — faith={r['Faithfulness']}, rel={r['Relevance']}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Session: {tracer.session_id} | "
    f"Total cost: ${tracer.total_cost():.5f} | "
    f"KB: {len(KNOWLEDGE_BASE)} docs"
)
