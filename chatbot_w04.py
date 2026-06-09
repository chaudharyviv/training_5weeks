"""
chatbot_w04.py — Week 4 Streamlit Chatbot
Full Multi-Agent IT Helpdesk with:
- Supervisor → Classifier → Researcher → Executor → Reviewer pipeline
- Live tool-trace panel (step-by-step agent actions)
- Approval UI for P1/P2 actions (human-in-the-loop)
- ServiceNow integration (mock + real mode)
- Audit log export
- Session dashboard with cost, latency, incidents created

Run:
    pip install streamlit openai anthropic requests pydantic
    streamlit run chatbot_w04.py

Set environment variables:
    export OPENAI_API_KEY=sk-...
    export SN_INSTANCE=devXXXXXX     (optional — falls back to mock)
    export SN_USER=admin             (optional)
    export SN_PASSWORD=admin         (optional)
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from requests.auth import HTTPBasicAuth
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import LLMClient, Tracer, Guardrails, TicketClassification
from openai import OpenAI

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IT Helpdesk Agent — Week 4",
    page_icon="🤖",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "tracer":             Tracer(session_id=f"w04-{str(uuid.uuid4())[:6]}"),
        "client":             None,
        "oai":                None,
        "history":            [],        # [{role, content, ts}]
        "tool_trace":         [],        # [{step, agent, tool, args, result, approved}]
        "audit_log":          [],        # full audit entries
        "incidents_created":  [],        # [{number, description, priority}]
        "classification":     None,      # last TicketClassification dict
        "pending_approval":   None,      # {tool, args, resolve_key} when HITL gate fires
        "mock_mode":          True,
        "mock_inc_counter":   [3000],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Lazy init clients
    if st.session_state.client is None:
        st.session_state.client = LLMClient(tracer=st.session_state.tracer)
    if st.session_state.oai is None:
        st.session_state.oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

init_state()

tracer = st.session_state.tracer
client = st.session_state.client
_oai   = st.session_state.oai
MODEL  = "gpt-4o-mini"

# ServiceNow config
SN_INSTANCE = os.getenv("SN_INSTANCE", "devXXXXXX")
SN_USER     = os.getenv("SN_USER",     "admin")
SN_PASSWORD = os.getenv("SN_PASSWORD", "admin")
SN_BASE     = f"https://{SN_INSTANCE}.service-now.com"
SN_AUTH     = HTTPBasicAuth(SN_USER, SN_PASSWORD)
SN_HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}

# Detect real ServiceNow
if st.session_state.mock_mode:
    try:
        r = requests.get(f"{SN_BASE}/api/now/table/incident",
                         auth=SN_AUTH, headers=SN_HEADERS,
                         params={"sysparm_limit": 1}, timeout=4)
        if r.status_code == 200:
            st.session_state.mock_mode = False
    except Exception:
        pass

MOCK_MODE = st.session_state.mock_mode

# ══════════════════════════════════════════════════════════════════════════════
# Mock incident store
# ══════════════════════════════════════════════════════════════════════════════
_MOCK_INCIDENTS = [
    {"number": "INC0001001", "description": "VPN TAP adapter missing after Windows update",
     "priority": "2", "state": "Resolved", "assigned_to": "Network-Ops",
     "created": "2024-11-04 09:12:00"},
    {"number": "INC0001003", "description": "Load balancer SSL certificate expired",
     "priority": "1", "state": "Resolved", "assigned_to": "Network-Ops",
     "created": "2024-10-28 07:45:00"},
    {"number": "INC0001004", "description": "Snowflake data warehouse read-only — maintenance",
     "priority": "3", "state": "In Progress", "assigned_to": "DBA-Team",
     "created": "2024-11-05 22:00:00"},
]

PRIORITY_MAP  = {"P1": "1", "P2": "2", "P3": "3", "P4": "4"}
PRIORITY_LABEL = {"1": "P1 Critical", "2": "P2 High", "3": "P3 Medium", "4": "P4 Low"}
ONCALL = {
    "Network-Ops":     {"name": "Karthik Subramanian", "phone": "+91-98XXXXXXX1", "slack": "@karthik.sub"},
    "Desktop-Support": {"name": "Ananya Iyer",          "phone": "+91-98XXXXXXX2", "slack": "@ananya.iyer"},
    "App-Support":     {"name": "Vijay Nair",            "phone": "+91-98XXXXXXX3", "slack": "@vijay.nair"},
    "Security":        {"name": "Deepa Krishnan",        "phone": "+91-98XXXXXXX4", "slack": "@deepa.k"},
    "Management":      {"name": "Rajesh Patel",          "phone": "+91-98XXXXXXX5", "slack": "@rajesh.patel"},
    "SAP-BASIS-TEAM":  {"name": "Anand Krishnamurthy",   "phone": "+91-98XXXXXXX6", "slack": "@anand.k"},
    "DBA-Team":        {"name": "Meena Rajan",            "phone": "+91-98XXXXXXX7", "slack": "@meena.rajan"},
}

# ══════════════════════════════════════════════════════════════════════════════
# Tool implementations
# ══════════════════════════════════════════════════════════════════════════════
def _audit(agent, tool, args, result, approved=True, approver="auto"):
    st.session_state.audit_log.append({
        "id":       str(uuid.uuid4())[:8],
        "ts":       datetime.utcnow().isoformat(),
        "agent":    agent,
        "tool":     tool,
        "args":     args,
        "result":   result,
        "approved": approved,
        "approver": approver,
    })


def search_incidents(query: str, priority: str = None, limit: int = 5) -> dict:
    if MOCK_MODE:
        kw   = query.lower()
        hits = [i for i in _MOCK_INCIDENTS if kw in i["description"].lower()]
        if priority:
            hits = [i for i in hits if i["priority"] == PRIORITY_MAP.get(priority, priority)]
        result = {"status": "success", "mode": "mock", "count": len(hits[:limit]),
                  "incidents": hits[:limit]}
    else:
        try:
            resp = requests.get(
                f"{SN_BASE}/api/now/table/incident",
                auth=SN_AUTH, headers=SN_HEADERS, timeout=8,
                params={"sysparm_limit": limit,
                        "sysparm_query": f"short_descriptionLIKE{query}",
                        "sysparm_fields": "number,short_description,priority,state,assigned_to,sys_created_on"}
            )
            if resp.status_code == 200:
                rows = resp.json().get("result", [])
                result = {"status": "success", "mode": "real", "count": len(rows),
                          "incidents": [{"number": r["number"],
                                         "description": r["short_description"],
                                         "priority": r["priority"],
                                         "state": r.get("state"),
                                         "assigned_to": r.get("assigned_to", {}).get("display_value", "?")}
                                        for r in rows]}
            else:
                result = {"status": "error", "http": resp.status_code}
        except Exception as e:
            result = {"status": "connection_error", "message": str(e)[:60]}
    _audit("executor", "search_incidents", {"query": query, "priority": priority}, result)
    return result


def create_incident(short_description: str, description: str,
                    priority: str = "P3", category: str = "Software",
                    caller_name: str = "IT Agent") -> dict:
    # Validate
    if priority not in PRIORITY_MAP:
        result = {"status": "validation_error", "message": f"Invalid priority '{priority}'"}
        _audit("executor", "create_incident", {"short_description": short_description}, result)
        return result

    if MOCK_MODE:
        st.session_state.mock_inc_counter[0] += 1
        num = f"INC{st.session_state.mock_inc_counter[0]:07d}"
        _MOCK_INCIDENTS.append({
            "number": num, "description": short_description,
            "priority": PRIORITY_MAP[priority], "state": "New",
            "assigned_to": "Unassigned", "created": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })
        result = {"status": "success", "mode": "mock", "incident_number": num,
                  "priority": priority, "category": category,
                  "url": f"{SN_BASE}/incident.do?number={num}"}
    else:
        try:
            payload = {"short_description": short_description[:80], "description": description,
                       "priority": PRIORITY_MAP[priority], "category": category.lower(),
                       "caller_id": {"display_value": caller_name}}
            resp = requests.post(f"{SN_BASE}/api/now/table/incident",
                                  auth=SN_AUTH, headers=SN_HEADERS, json=payload, timeout=10)
            if resp.status_code in (200, 201):
                r = resp.json().get("result", {})
                result = {"status": "success", "mode": "real",
                          "incident_number": r.get("number"), "sys_id": r.get("sys_id"),
                          "priority": priority, "category": category,
                          "url": f"{SN_BASE}/incident.do?sys_id={r.get('sys_id')}"}
            else:
                result = {"status": "error", "http": resp.status_code, "message": resp.text[:100]}
        except Exception as e:
            result = {"status": "connection_error", "message": str(e)[:60]}

    if result.get("status") == "success":
        st.session_state.incidents_created.append({
            "number":      result["incident_number"],
            "description": short_description,
            "priority":    priority,
            "category":    category,
        })
    _audit("executor", "create_incident",
           {"short_description": short_description, "priority": priority, "category": category},
           result)
    return result


def get_weather(city: str) -> dict:
    COORDS = {"bangalore": (12.97, 77.59), "mumbai": (19.08, 72.88),
              "delhi": (28.61, 77.21), "chennai": (13.08, 80.27),
              "hyderabad": (17.38, 78.49), "pune": (18.52, 73.86)}
    coords = COORDS.get(city.lower())
    if not coords:
        result = {"status": "error", "message": f"City '{city}' not supported"}
    else:
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast",
                              params={"latitude": coords[0], "longitude": coords[1],
                                      "current": "temperature_2m,precipitation,wind_speed_10m",
                                      "timezone": "auto"}, timeout=5)
            d    = r.json().get("current", {})
            rain = float(d.get("precipitation", 0))
            result = {"status": "success", "city": city,
                      "temp_c": d.get("temperature_2m"), "rain_mm": rain,
                      "wind_kmh": d.get("wind_speed_10m"),
                      "condition": "Heavy rain" if rain > 10 else "Moderate rain" if rain > 2 else "Clear",
                      "onsite_safe": rain <= 5}
        except Exception as e:
            result = {"status": "error", "message": str(e)[:60]}
    _audit("executor", "get_weather", {"city": city}, result)
    return result


def lookup_wiki(term: str) -> dict:
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{term.replace(' ', '_')}",
            headers={"User-Agent": "ITAgent/1.0"}, timeout=5)
        if r.status_code == 200:
            d = r.json()
            result = {"status": "success", "term": d.get("title"), "summary": d.get("extract", "")[:400]}
        else:
            result = {"status": "not_found", "message": f"No Wikipedia article for '{term}'"}
    except Exception as e:
        result = {"status": "error", "message": str(e)[:60]}
    _audit("researcher", "lookup_wiki", {"term": term}, result)
    return result


def get_oncall(team: str) -> dict:
    if team in ONCALL:
        result = {"status": "success", "team": team, "oncall": ONCALL[team]}
    else:
        result = {"status": "error", "message": f"Unknown team '{team}'"}
    _audit("executor", "get_oncall", {"team": team}, result)
    return result


TOOL_DISPATCH = {
    "search_incidents": search_incidents,
    "create_incident":  create_incident,
    "get_weather":      get_weather,
    "lookup_wiki":      lookup_wiki,
    "get_oncall":       get_oncall,
}

# ══════════════════════════════════════════════════════════════════════════════
# Tool schemas
# ══════════════════════════════════════════════════════════════════════════════
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_incidents",
        "description": "Search ServiceNow incidents. ALWAYS call before create_incident to check duplicates.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "priority": {"type": "string"},
                                      "limit": {"type": "integer", "default": 5}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_incident",
        "description": "Create a ServiceNow incident after confirming no duplicate exists.",
        "parameters": {"type": "object",
                       "properties": {
                           "short_description": {"type": "string", "description": "Max 80 chars"},
                           "description": {"type": "string"},
                           "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
                           "category": {"type": "string",
                                        "enum": ["Network", "Hardware", "Software", "Access", "Data", "Compliance", "Other"]},
                           "caller_name": {"type": "string"}},
                       "required": ["short_description", "description", "priority", "category"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get weather for on-site hardware work. Only call for physical hardware issues.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string",
                                               "enum": ["bangalore", "mumbai", "delhi", "chennai", "hyderabad", "pune"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "lookup_wiki",
        "description": "Look up an IT concept, ITIL term, or technical standard on Wikipedia.",
        "parameters": {"type": "object",
                       "properties": {"term": {"type": "string"}},
                       "required": ["term"]}}},
    {"type": "function", "function": {
        "name": "get_oncall",
        "description": "Get on-call engineer for a team. Call for P1 and P2 incidents.",
        "parameters": {"type": "object",
                       "properties": {"team": {"type": "string",
                                               "enum": list(ONCALL.keys())}},
                       "required": ["team"]}}},
]

# ══════════════════════════════════════════════════════════════════════════════
# Agent system prompts
# ══════════════════════════════════════════════════════════════════════════════
AGENT_SYSTEM = """
You are an IT helpdesk agent with the following tools:
- search_incidents: search existing ServiceNow tickets
- create_incident: log a new ticket (ALWAYS search first for duplicates)
- get_weather: check weather before scheduling on-site hardware work
- lookup_wiki: look up IT concepts or ITIL terms for the user
- get_oncall: get the on-call contact for P1/P2 incidents

Rules:
1. Always search before creating — avoid duplicate tickets.
2. For hardware issues needing on-site dispatch: check weather for the city first.
3. For P1 or P2: get the on-call contact after creating the ticket.
4. If a tool fails: say so clearly. Never invent data.
5. Always confirm incident numbers in your final response.
6. Keep responses concise and actionable.
"""

CLASSIFIER_SYSTEM = """
Analyse this IT issue and return JSON only:
{"category": "Network|Hardware|Software|Access|Data|Compliance|Other",
 "priority": "P1|P2|P3|P4",
 "affected_users": integer,
 "assignee_team": "Network-Ops|Desktop-Support|App-Support|Security|Management|SAP-BASIS-TEAM|DBA-Team",
 "sla_breach_risk": boolean,
 "confidence": float,
 "needs_review": boolean}
P1=revenue/outage now/>50 users. P2=team blocked/deadline today. P3=individual/workaround. P4=cosmetic.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Agent runner
# ══════════════════════════════════════════════════════════════════════════════
_HITL_TOOLS = {"create_incident"}   # tools that require approval for P1/P2

def run_agent_step(
    user_request:  str,
    require_hitl:  bool = False,
    auto_approve:  bool = True,
) -> dict:
    """
    Single-pass agent run. Returns {answer, trace_steps, tool_calls}.
    Handles HITL gating for create_incident when require_hitl=True.
    """
    messages = [{"role": "system", "content": AGENT_SYSTEM}]
    # Last 10 conversation turns as context
    messages += [{"role": m["role"], "content": m["content"]}
                 for m in st.session_state.history[-10:]]
    messages.append({"role": "user", "content": user_request})

    trace_steps = []
    tool_calls  = []

    for step in range(8):
        resp   = _oai.chat.completions.create(
            model=MODEL, messages=messages,
            tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0
        )
        msg    = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if finish == "tool_calls" and msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                tname = tc.function.name
                targs = json.loads(tc.function.arguments)

                # Input guard-rail on tool args
                safe, reason = Guardrails.check_input(json.dumps(targs))
                if not safe:
                    tool_result = {"status": "blocked", "reason": reason}
                    _audit("agent", f"{tname}_blocked", targs, tool_result,
                           approved=False, approver="guardrail")
                    trace_steps.append({"step": step + 1, "agent": "agent",
                                        "tool": tname, "args": targs,
                                        "result": tool_result, "approved": False})
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(tool_result)})
                    continue

                # HITL gate
                approved = True
                approver = "auto"
                if require_hitl and tname in _HITL_TOOLS:
                    if not auto_approve:
                        # Store pending approval and return — UI will handle it
                        st.session_state.pending_approval = {
                            "tool":       tname,
                            "args":       targs,
                            "tc_id":      tc.id,
                            "messages":   messages,
                        }
                        return {"answer":      "__PENDING_APPROVAL__",
                                "trace_steps": trace_steps,
                                "tool_calls":  tool_calls}
                    else:
                        approver = "auto_approved_demo"

                if tname in TOOL_DISPATCH:
                    tool_result = TOOL_DISPATCH[tname](**targs)
                    tool_calls.append({"tool": tname, "args": targs, "result": tool_result})
                else:
                    tool_result = {"status": "error", "message": f"Unknown tool: {tname}"}

                trace_steps.append({"step": step + 1, "agent": "agent",
                                     "tool": tname, "args": targs,
                                     "result": tool_result, "approved": approved})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(tool_result)})

        elif finish == "stop":
            answer = msg.content or ""
            return {"answer": answer, "trace_steps": trace_steps, "tool_calls": tool_calls}
        else:
            break

    return {"answer": "Agent did not complete.", "trace_steps": trace_steps, "tool_calls": tool_calls}


def classify_request(text: str) -> Optional[dict]:
    raw = client.chat(MODEL, user=text, system=CLASSIFIER_SYSTEM,
                      temperature=0, json_mode=True, tags=["classify"])
    try:
        return json.loads(raw)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════
st.title("🤖  IT Helpdesk — Agentic Chatbot")
st.caption(
    f"Week 4 · Multi-tool agent · ServiceNow {'✅ REAL' if not MOCK_MODE else '🟡 MOCK'} · "
    f"Session: {tracer.session_id}"
)

# ── Sidebar: controls + settings ──────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Settings")
    auto_approve = st.toggle("Auto-approve P1/P2 actions", value=True,
                             help="Turn OFF to see the human-in-the-loop approval gate")
    show_raw_json = st.toggle("Show raw JSON in tool trace", value=False)

    st.divider()
    st.subheader("📊 Session Dashboard")
    col_a, col_b = st.columns(2)
    col_a.metric("Cost",    f"${tracer.total_cost():.5f}")
    col_b.metric("LLM calls", len(tracer.traces))

    col_c, col_d = st.columns(2)
    col_c.metric("Incidents", len(st.session_state.incidents_created))
    col_d.metric("Audit entries", len(st.session_state.audit_log))

    avg_lat = tracer.avg_latency_ms()
    st.metric("Avg latency", f"{avg_lat:.0f} ms")

    st.divider()
    if st.session_state.incidents_created:
        st.subheader("🎫 Incidents Created")
        for inc in st.session_state.incidents_created[-5:]:
            pri_icon = {"P1": "🔴", "P2": "🟠", "P3": "🟡", "P4": "🟢"}.get(inc["priority"], "⚪")
            st.caption(f"{pri_icon} `{inc['number']}` — {inc['description'][:40]}")

    st.divider()
    if st.button("🔄 Reset Session"):
        for k in ["history", "tool_trace", "audit_log", "incidents_created",
                  "classification", "pending_approval"]:
            st.session_state[k] = [] if k in ["history", "tool_trace", "audit_log",
                                               "incidents_created"] else None
        st.rerun()

# ── Main: chat + trace ─────────────────────────────────────────────────────────
chat_col, trace_col = st.columns([3, 2])

with chat_col:
    st.subheader("💬 Chat")

    chat_container = st.container(height=440)
    with chat_container:
        if not st.session_state.history:
            st.info(
                "👋 Hi! I'm your IT helpdesk agent. I can:\n"
                "- Log ServiceNow tickets\n"
                "- Search past incidents\n"
                "- Check weather before on-site work\n"
                "- Look up IT concepts\n"
                "- Get on-call contacts for P1/P2\n\n"
                "Describe your IT issue to get started."
            )
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # ── HITL pending approval ──────────────────────────────────────────────────
    if st.session_state.pending_approval and not auto_approve:
        pa = st.session_state.pending_approval
        st.warning("⏸️ **Agent paused — Human approval required**")
        st.write(f"**Tool:** `{pa['tool']}`")
        st.json(pa["args"])
        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Approve"):
            # Execute the approved tool call
            result = TOOL_DISPATCH[pa["tool"]](**pa["args"])
            pa["messages"].append({"role": "tool", "tool_call_id": pa["tc_id"],
                                   "content": json.dumps(result)})
            _audit("agent", pa["tool"], pa["args"], result, approved=True, approver="human")
            st.session_state.pending_approval = None
            st.rerun()
        if col_no.button("❌ Reject"):
            reject_result = {"status": "rejected_by_human"}
            _audit("agent", pa["tool"], pa["args"], reject_result,
                   approved=False, approver="human")
            st.session_state.pending_approval = None
            st.rerun()

    # ── Chat input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Describe your IT issue..."):
        safe, reason = Guardrails.check_input(prompt)
        if not safe:
            st.error(f"🛡️ Request blocked: {reason}")
        else:
            st.session_state.history.append({"role": "user", "content": prompt})

            # Background classification
            classification = classify_request(prompt)
            st.session_state.classification = classification

            # Determine if HITL needed
            pri = (classification or {}).get("priority", "P3")
            require_hitl = pri in ("P1", "P2") and not auto_approve

            with st.spinner(f"Agent working... (mode: {'MOCK' if MOCK_MODE else 'REAL'})"):
                result = run_agent_step(
                    prompt,
                    require_hitl=require_hitl,
                    auto_approve=auto_approve,
                )

            if result["answer"] != "__PENDING_APPROVAL__":
                st.session_state.history.append(
                    {"role": "assistant", "content": result["answer"]}
                )
                st.session_state.tool_trace = result["trace_steps"]

            st.rerun()

with trace_col:
    # ── Classification badge ───────────────────────────────────────────────────
    st.subheader("📊 Classification")
    c = st.session_state.classification
    if c:
        pri = c.get("priority", "?")
        pri_color = {"P1": "🔴", "P2": "🟠", "P3": "🟡", "P4": "🟢"}.get(pri, "⚪")

        col1, col2 = st.columns(2)
        col1.metric("Category", c.get("category", "?"))
        col2.metric("Priority", f"{pri_color} {pri}")

        col3, col4 = st.columns(2)
        col3.metric("Confidence",    f"{c.get('confidence', 0):.0%}")
        col4.metric("Affected users", c.get("affected_users", 1))

        team = c.get("assignee_team", "?")
        st.caption(f"**Team:** {team}")
        if c.get("sla_breach_risk"):
            st.error("⚠️  SLA breach risk — escalate urgently")
        if c.get("needs_review"):
            st.warning("🔍 Needs P1/P2 review — REVIEWER agent should run")
    else:
        st.caption("Classification appears after first message")

    st.divider()

    # ── Tool trace ─────────────────────────────────────────────────────────────
    st.subheader("🔧 Tool Trace")
    trace = st.session_state.tool_trace

    if trace:
        for step in trace:
            status = step["result"].get("status", "?")
            status_icon = "✅" if status == "success" else "❌" if status in ("error", "connection_error") else "🛡️"
            approved_icon = "✅" if step.get("approved", True) else "❌"

            with st.expander(
                f"{status_icon} Step {step['step']} · **{step['tool']}** · {status}",
                expanded=(status != "success")
            ):
                st.caption(f"Agent: {step['agent']} | Approved: {approved_icon}")

                # Args
                st.caption("**Arguments:**")
                if show_raw_json:
                    st.json(step["args"])
                else:
                    for k, v in step["args"].items():
                        st.caption(f"  `{k}`: {str(v)[:60]}")

                # Key result fields
                st.caption("**Result:**")
                result = step["result"]
                if result.get("status") == "success":
                    if "incident_number" in result:
                        st.success(f"🎫 Created: `{result['incident_number']}`")
                    if "incidents" in result:
                        for inc in result["incidents"][:3]:
                            st.caption(f"  [{inc.get('number')}] {inc.get('description','')[:50]}")
                    if "oncall" in result:
                        oc = result["oncall"]
                        st.info(f"📞 {oc['name']} · {oc['slack']}")
                    if "condition" in result:
                        safe_icon = "✅" if result.get("onsite_safe") else "⚠️"
                        st.caption(f"{safe_icon} {result['condition']} | {result.get('temp_c')}°C")
                    if "summary" in result:
                        st.caption(result["summary"][:300])
                elif status == "connection_error":
                    st.error(f"🔌 {result.get('message', 'Connection failed')}")
                elif status == "blocked":
                    st.warning(f"🛡️ {result.get('reason', 'Blocked by guardrail')}")
                else:
                    st.caption(str(result)[:100])
    else:
        st.caption("Tool calls will appear here as the agent works")

# ══════════════════════════════════════════════════════════════════════════════
# Audit log tab (below main area)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
tabs = st.tabs(["📋 Audit Log", "📤 Export"])

with tabs[0]:
    if st.session_state.audit_log:
        rows = []
        for e in st.session_state.audit_log[-20:]:
            rows.append({
                "ID":       e["id"],
                "Time":     e["ts"][11:19],
                "Agent":    e["agent"],
                "Tool":     e["tool"],
                "Status":   e["result"].get("status", "?"),
                "Approved": "✅" if e.get("approved") else "❌",
                "Approver": e.get("approver", "auto"),
            })
        st.dataframe(rows, use_container_width=True)

        # Anomaly detection
        anomalies = [e for e in st.session_state.audit_log
                     if not e.get("approved") or e["result"].get("status") == "connection_error"]
        if anomalies:
            st.warning(f"⚠️ {len(anomalies)} anomalies detected (rejected actions or connection errors)")
    else:
        st.caption("Audit log will populate as the agent acts")

with tabs[1]:
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("📥 Export audit log (JSON)") and st.session_state.audit_log:
            data = json.dumps(st.session_state.audit_log, indent=2, default=str)
            st.download_button("Download audit.json", data=data,
                               file_name="w04_audit_log.json", mime="application/json")
    with col_exp2:
        if st.button("📥 Export conversation (JSON)") and st.session_state.history:
            data = json.dumps(st.session_state.history, indent=2, default=str)
            st.download_button("Download conversation.json", data=data,
                               file_name="w04_conversation.json", mime="application/json")
