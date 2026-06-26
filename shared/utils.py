"""
shared/utils.py
═══════════════════════════════════════════════════════════════════
Shared utilities for the GenAI + Agentic AI training programme.
Imported by all 4 weekly notebooks.

Contents:
  - MODELS              : friendly model aliases  (use these in notebooks)
  - MODEL_COSTS         : pricing table (USD / 1K tokens)
  - check_api_keys()    : show which APIs are configured
  - quick_start()       : one-liner setup → (LLMClient, Tracer)
  - ask_gpt()           : simple OpenAI chat wrapper for notebooks
  - ask_claude()        : simple Anthropic chat wrapper for notebooks
  - LLMClient           : unified OpenAI + Anthropic client with cost tracking
  - Tracer              : lightweight observability (latency, tokens, cost)
  - PromptRegistry      : versioned prompt store
  - Validator           : Pydantic-based output validation + guardrails
  - EvalFramework       : per-session evaluation harness
  - ModelSelector       : ipywidgets dropdown for interactive model picking
  - pretty_print()      : banner(), section(), observe(), discuss(), compare() …
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import os
import time
import uuid
import hashlib
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict

from pydantic import BaseModel, Field, field_validator
from IPython.core.display import Markdown, display

# ── Auto-load .env (works from any working directory) ─────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    # Walk up from this file's location to find the nearest .env
    _here = os.path.dirname(os.path.abspath(__file__))
    for _candidate in [_here, os.path.dirname(_here)]:
        _env_path = os.path.join(_candidate, ".env")
        if os.path.isfile(_env_path):
            _load_dotenv(_env_path, override=False)
            break
    _DOTENV = True
except ImportError:
    _DOTENV = False

# ── Optional SDK imports (fail gracefully if not installed) ───────────────────
try:
    import tiktoken
    _TIKTOKEN = True
except ImportError:
    _TIKTOKEN = False

try:
    from openai import OpenAI as _OpenAI
    _OPENAI = True
except ImportError:
    _OPENAI = False

try:
    import anthropic as _anthropic
    _ANTHROPIC = True
except ImportError:
    _ANTHROPIC = False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODEL REGISTRY  — friendly aliases + cost table
# ══════════════════════════════════════════════════════════════════════════════

# Use these in notebooks instead of raw model-id strings.
# Change once here; all notebooks pick it up automatically.
MODELS: Dict[str, str] = {
    # OpenAI
    "fast":         "gpt-4o-mini",       # cheap, quick — good for demos & evals
    "smart":        "gpt-4o",            # best reasoning, 33× more expensive
    # Anthropic
    "haiku":        "claude-haiku-4-5",  # fastest Claude — matches gpt-4o-mini tier
    "sonnet":       "claude-sonnet-4-5", # balanced Claude — matches gpt-4o tier
    "opus":         "claude-opus-4-5",   # most capable Claude
    # Embeddings
    "embed_small":  "text-embedding-3-small",
    "embed_large":  "text-embedding-3-large",
}

MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # model_id → {input, output} USD per 1 000 tokens
    "gpt-4o-mini":              {"input": 0.000150, "output": 0.000600},
    "gpt-4o":                   {"input": 0.005000, "output": 0.015000},
    "gpt-4-turbo":              {"input": 0.010000, "output": 0.030000},
    "claude-haiku-4-5":         {"input": 0.000800, "output": 0.004000},
    "claude-sonnet-4-5":        {"input": 0.003000, "output": 0.015000},
    "claude-opus-4-5":          {"input": 0.015000, "output": 0.075000},
    "text-embedding-3-small":   {"input": 0.000020, "output": 0.0},
    "text-embedding-3-large":   {"input": 0.000130, "output": 0.0},
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SETUP HELPERS  — check keys, quick_start
# ══════════════════════════════════════════════════════════════════════════════

def check_api_keys() -> Dict[str, bool]:
    """
    Print which API keys are configured and return a dict.

    Usage (top of any notebook):
        check_api_keys()
    """
    openai_ok    = bool(os.getenv("OPENAI_API_KEY", "").strip())
    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

    _ok = lambda v: "OK  configured" if v else "!!  missing   "
    print("=" * 43)
    print("  API Key Status")
    print("=" * 43)
    print(f"  OPENAI_API_KEY    : {_ok(openai_ok)}")
    print(f"  ANTHROPIC_API_KEY : {_ok(anthropic_ok)}")
    print("=" * 43)
    if not openai_ok and not anthropic_ok:
        print("\n⚠️  No keys found — add them to your .env file or set them as env vars.\n"
              "   See the README or run:  !echo OPENAI_API_KEY=sk-... >> .env")
    return {"openai": openai_ok, "anthropic": anthropic_ok}


def quick_start(
    session_id: str = "",
    default_gpt: str = "",
    default_claude: str = "",
    temperature: float = 0.0,
) -> Tuple["LLMClient", "Tracer"]:
    """
    One-liner notebook setup.  Returns (client, tracer) ready to use.

    Usage:
        client, tracer = quick_start("w01")
        response = client.chat(MODELS['fast'], user="Hello!")

    Parameters
    ----------
    session_id      : label for this run (default: timestamp)
    default_gpt     : override the default GPT model
    default_claude  : override the default Claude model
    temperature     : default temperature for all calls
    """
    keys = check_api_keys()
    tracer = Tracer(session_id=session_id)
    client = LLMClient(tracer=tracer, default_temperature=temperature)

    available = []
    if keys["openai"]:
        gpt = default_gpt or MODELS["fast"]
        available.append(f"GPT → {gpt}")
    if keys["anthropic"]:
        claude = default_claude or MODELS["haiku"]
        available.append(f"Claude → {claude}")

    banner("🚀  Quick Start")
    print(f"  Session ID  : {tracer.session_id}")
    print(f"  Available   : {' | '.join(available) if available else 'none — check keys'}")
    print(f"  Temperature : {temperature}")
    print(f"\n  Handy aliases  →  MODELS = {list(MODELS.keys())[:5]} …")
    print(f"  e.g.  client.chat(MODELS['fast'], user='Hello!')")
    return client, tracer


def ask_gpt(
    system: str,
    user: str,
    model: str = "fast",
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> str:
    """
    Simple OpenAI chat wrapper for notebooks.

    Usage:
        from shared.utils import ask_gpt
        answer = ask_gpt("You are helpful.", "What is ITIL?")
    """
    if not _OPENAI:
        raise RuntimeError("openai package not installed — run: pip install openai")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set — add it to your .env file")
    model = MODELS.get(model, model)
    client = _OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def ask_claude(
    system: str,
    user: str,
    model: str = "haiku",
    temperature: float = 0.7,
    max_tokens: int = 300,
) -> str:
    """
    Simple Anthropic chat wrapper for notebooks.

    Usage:
        from shared.utils import ask_claude
        answer = ask_claude("You are helpful.", "What is ITIL?")
    """
    if not _ANTHROPIC:
        raise RuntimeError("anthropic package not installed — run: pip install anthropic")
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — add it to your .env file")
    model = MODELS.get(model, model)
    client = _anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRACE  (single LLM call record)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trace:
    trace_id:       str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id:     str   = ""
    model:          str   = ""
    provider:       str   = ""           # "openai" | "anthropic"
    prompt_tokens:  int   = 0
    output_tokens:  int   = 0
    latency_ms:     float = 0.0
    cost_usd:       float = 0.0
    temperature:    float = 0.0
    success:        bool  = True
    error:          str   = ""
    tags:           List[str] = field(default_factory=list)
    timestamp:      str   = field(default_factory=lambda: datetime.utcnow().isoformat())

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRACER  (session-level collector)
# ══════════════════════════════════════════════════════════════════════════════

class Tracer:
    """
    Lightweight observability collector.

    Usage:
        tracer = Tracer(session_id="w01-demo")
        client = LLMClient(tracer=tracer)
        tracer.summary()
    """

    def __init__(self, session_id: str = ""):
        self.session_id = session_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.traces: List[Trace] = []

    def record(self, trace: Trace) -> None:
        trace.session_id = self.session_id
        self.traces.append(trace)

    def total_cost(self) -> float:
        return sum(t.cost_usd for t in self.traces)

    def total_tokens(self) -> int:
        return sum(t.total_tokens() for t in self.traces)

    def avg_latency_ms(self) -> float:
        if not self.traces:
            return 0.0
        return sum(t.latency_ms for t in self.traces) / len(self.traces)

    def error_rate(self) -> float:
        if not self.traces:
            return 0.0
        return sum(1 for t in self.traces if not t.success) / len(self.traces)

    def summary(self) -> None:
        banner("📊  Session Tracer Summary")
        print(f"  Session ID    : {self.session_id}")
        print(f"  Total calls   : {len(self.traces)}")
        print(f"  Total tokens  : {self.total_tokens():,}")
        print(f"  Total cost    : ${self.total_cost():.5f}")
        print(f"  Avg latency   : {self.avg_latency_ms():.0f} ms")
        print(f"  Error rate    : {self.error_rate():.1%}")
        if self.traces:
            by_model: Dict[str, int] = {}
            for t in self.traces:
                by_model[t.model] = by_model.get(t.model, 0) + 1
            print(f"  Calls by model: {by_model}")

    def export_jsonl(self, path: str = "traces.jsonl") -> None:
        with open(path, "w") as f:
            for t in self.traces:
                f.write(json.dumps(t.as_dict()) + "\n")
        print(f"Traces written to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LLMClient  (unified OpenAI + Anthropic)
# ══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    Single interface for OpenAI and Anthropic.

    Usage:
        client = LLMClient()
        response = client.chat(MODELS['fast'],  user="Hello!")   # GPT-4o-mini
        response = client.chat(MODELS['haiku'], user="Hello!")   # Claude Haiku

    All calls are traced automatically.
    """

    def __init__(
        self,
        openai_api_key:    str = "",
        anthropic_api_key: str = "",
        tracer: Optional[Tracer] = None,
        default_temperature: float = 0.0,
    ):
        self.tracer = tracer or Tracer()
        self.default_temperature = default_temperature

        if _OPENAI:
            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            self._oai = _OpenAI(api_key=key) if key else None
        else:
            self._oai = None

        if _ANTHROPIC:
            key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
            self._claude = _anthropic.Anthropic(api_key=key) if key else None
        else:
            self._claude = None

    # ── provider detection ─────────────────────────────────────────────────────

    def _provider(self, model: str) -> str:
        # Accept either a raw model id or a MODELS alias
        resolved = MODELS.get(model, model)
        return "anthropic" if resolved.startswith("claude") else "openai"

    def _resolve(self, model: str) -> str:
        """Resolve a friendly alias (e.g. 'haiku') to its full model id."""
        return MODELS.get(model, model)

    # ── cost calculation ───────────────────────────────────────────────────────

    def _cost(self, model: str, prompt_tokens: int, output_tokens: int) -> float:
        rates = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        return (prompt_tokens * rates["input"] + output_tokens * rates["output"]) / 1000

    # ── main chat interface ────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        user: str,
        system: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        json_mode: bool = False,
        tags: Optional[List[str]] = None,
        messages: Optional[List[Dict]] = None,
    ) -> str:
        """
        Unified chat call. Returns the text response.

        `model` can be a friendly alias from MODELS or a raw model id:
            client.chat("fast",          user="...")   # gpt-4o-mini
            client.chat("haiku",         user="...")   # claude-haiku-4-5
            client.chat("gpt-4o-mini",   user="...")   # raw id also works
        """
        model    = self._resolve(model)
        temp     = temperature if temperature is not None else self.default_temperature
        provider = self._provider(model)
        t  = Trace(model=model, provider=provider, temperature=temp, tags=tags or [])
        t0 = time.time()

        try:
            if provider == "openai":
                if self._oai is None:
                    raise RuntimeError("OpenAI client not initialised — check OPENAI_API_KEY")
                result = self._oai_chat(model, system, user, temp, max_tokens, json_mode, messages)
            else:
                if self._claude is None:
                    raise RuntimeError("Anthropic client not initialised — check ANTHROPIC_API_KEY")
                result = self._claude_chat(model, system, user, temp, max_tokens, messages)

            t.prompt_tokens = result["prompt_tokens"]
            t.output_tokens = result["output_tokens"]
            t.cost_usd      = self._cost(model, t.prompt_tokens, t.output_tokens)
            t.success       = True
            return result["text"]

        except Exception as exc:
            t.success = False
            t.error   = str(exc)
            raise

        finally:
            t.latency_ms = (time.time() - t0) * 1000
            self.tracer.record(t)

    # ── OpenAI backend ─────────────────────────────────────────────────────────

    def _oai_chat(self, model, system, user, temp, max_tokens, json_mode, messages):
        msgs = messages if messages else [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        kwargs = dict(model=model, temperature=temp, max_tokens=max_tokens, messages=msgs)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp  = self._oai.chat.completions.create(**kwargs)
        usage = resp.usage
        return {
            "text":          resp.choices[0].message.content,
            "prompt_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
        }

    # ── Anthropic backend ──────────────────────────────────────────────────────

    def _claude_chat(self, model, system, user, temp, max_tokens, messages):
        msgs = [m for m in messages if m["role"] != "system"] if messages else \
               [{"role": "user", "content": user}]
        resp  = self._claude.messages.create(
            model=model, max_tokens=max_tokens, temperature=temp,
            system=system, messages=msgs,
        )
        usage = resp.usage
        return {
            "text":          resp.content[0].text,
            "prompt_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }

    # ── Embedding ──────────────────────────────────────────────────────────────

    def embed(self, texts: List[str] | str, model: str = "text-embedding-3-small"):
        """Return numpy array of embeddings (OpenAI only)."""
        import numpy as np
        model = self._resolve(model)
        if isinstance(texts, str):
            texts = [texts]
        if self._oai is None:
            raise RuntimeError("OpenAI client not initialised — check OPENAI_API_KEY")
        resp = self._oai.embeddings.create(model=model, input=texts)
        emb  = np.array([e.embedding for e in resp.data])
        cost = self._cost(model, sum(len(t.split()) for t in texts), 0)
        trace = Trace(model=model, provider="openai",
                      prompt_tokens=sum(len(t.split()) for t in texts),
                      cost_usd=cost, tags=["embed"])
        self.tracer.record(trace)
        return emb

    # ── Token counting ─────────────────────────────────────────────────────────

    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        model = self._resolve(model)
        if _TIKTOKEN and not model.startswith("claude"):
            try:
                enc = tiktoken.encoding_for_model(model)
                return len(enc.encode(text))
            except Exception:
                pass
        return len(text.split())  # fallback

    def estimate_cost(self, prompt: str, expected_output_tokens: int,
                      model: str = "gpt-4o-mini") -> float:
        model = self._resolve(model)
        pt = self.count_tokens(prompt, model)
        return self._cost(model, pt, expected_output_tokens)

    # ── Multi-model comparison helper ─────────────────────────────────────────

    def compare_models(
        self,
        models: List[str],
        user: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        """
        Run the same prompt on multiple models and print a side-by-side comparison.

        Usage:
            client.compare_models(
                models=["fast", "haiku", "sonnet"],
                user="Summarise this ticket in one sentence: ...",
            )
        """
        banner(f"Model Comparison  ({len(models)} models)")
        print(f"  Prompt: {user[:100]}{'…' if len(user) > 100 else ''}\n")
        for m in models:
            resolved = self._resolve(m)
            label    = f"{m} ({resolved})" if m != resolved else resolved
            try:
                t0   = time.time()
                resp = self.chat(resolved, user=user, system=system,
                                 temperature=temperature, max_tokens=max_tokens,
                                 tags=["compare"])
                ms   = (time.time() - t0) * 1000
                cost = self._cost(resolved,
                                  self.tracer.traces[-1].prompt_tokens,
                                  self.tracer.traces[-1].output_tokens)
                print(f"── {label} ──")
                print(f"   {resp.strip()}")
                print(f"   [{ms:.0f} ms | ${cost:.5f}]\n")
            except Exception as exc:
                print(f"── {label} ── ERROR: {exc}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PROMPT REGISTRY  (versioned prompt store)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PromptVersion:
    version:       str
    system:        str
    created:       str = field(default_factory=lambda: datetime.utcnow().isoformat())
    author:        str = ""
    change_reason: str = ""
    tags:          List[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.system.encode()).hexdigest()[:12]


class PromptRegistry:
    """
    Versioned prompt store.

    Usage:
        registry = PromptRegistry()
        registry.register("ticket_classifier", "v1", system=PROMPT_V1, author="trainer")
        prompt = registry.get("ticket_classifier")           # latest
        prompt = registry.get("ticket_classifier", "v1")     # specific version
        registry.diff("ticket_classifier", "v1", "v2")
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, PromptVersion]] = {}

    def register(self, name: str, version: str, system: str,
                 author: str = "", change_reason: str = "",
                 tags: List[str] = None) -> None:
        if name not in self._store:
            self._store[name] = {}
        self._store[name][version] = PromptVersion(
            version=version, system=system, author=author,
            change_reason=change_reason, tags=tags or []
        )

    def get(self, name: str, version: str = "latest") -> PromptVersion:
        if name not in self._store:
            raise KeyError(f"No prompt named '{name}'")
        versions = self._store[name]
        if version == "latest":
            return versions[sorted(versions)[-1]]
        if version not in versions:
            raise KeyError(f"Version '{version}' not found for prompt '{name}'")
        return versions[version]

    def list_versions(self, name: str) -> List[str]:
        return sorted(self._store.get(name, {}).keys())

    def diff(self, name: str, v_old: str, v_new: str) -> None:
        import difflib
        old  = self.get(name, v_old).system.splitlines(keepends=True)
        new  = self.get(name, v_new).system.splitlines(keepends=True)
        diff = list(difflib.unified_diff(old, new,
                                         fromfile=f"{name}@{v_old}",
                                         tofile=f"{name}@{v_new}"))
        print("".join(diff) if diff else "No differences.")

    def ab_test(self, name: str, v_a: str, v_b: str,
                test_inputs: List[str], client: "LLMClient",
                model: str = "gpt-4o-mini") -> None:
        banner(f"A/B Test: {name}  {v_a} vs {v_b}")
        pa = self.get(name, v_a)
        pb = self.get(name, v_b)
        for i, inp in enumerate(test_inputs, 1):
            out_a = client.chat(model, user=inp, system=pa.system, tags=["ab_test", v_a])
            out_b = client.chat(model, user=inp, system=pb.system, tags=["ab_test", v_b])
            print(f"\n── Input {i}: {inp[:60]} ──")
            print(f"  [{v_a}] {out_a.strip()[:120]}")
            print(f"  [{v_b}] {out_b.strip()[:120]}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PYDANTIC MODELS  (shared output schemas)
# ══════════════════════════════════════════════════════════════════════════════

class TicketClassification(BaseModel):
    category:        str   = Field(..., description="Network|Hardware|Software|Access|Data|Compliance|Other")
    priority:        str   = Field(..., description="P1|P2|P3|P4")
    affected_users:  int   = Field(1, ge=1)
    assignee_team:   str   = Field(..., description="Network-Ops|Desktop-Support|App-Support|Security|Management")
    estimated_hours: int   = Field(1, ge=1)
    sla_breach_risk: bool  = Field(False)
    summary:         str   = Field(..., max_length=120)
    confidence:      float = Field(0.9, ge=0.0, le=1.0)

    VALID_CATEGORIES: ClassVar[Set[str]] = {"Network","Hardware","Software","Access","Data","Compliance","Other"}
    VALID_PRIORITIES: ClassVar[Set[str]] = {"P1","P2","P3","P4"}
    VALID_TEAMS:      ClassVar[Set[str]] = {"Network-Ops","Desktop-Support","App-Support","Security","Management"}

    @field_validator("category")
    @classmethod
    def check_category(cls, v):
        if v not in cls.VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {v}. Must be one of {cls.VALID_CATEGORIES}")
        return v

    @field_validator("priority")
    @classmethod
    def check_priority(cls, v):
        if v not in cls.VALID_PRIORITIES:
            raise ValueError(f"Invalid priority: {v}")
        return v

    @field_validator("assignee_team")
    @classmethod
    def check_team(cls, v):
        if v not in cls.VALID_TEAMS:
            raise ValueError(f"Invalid team: {v}")
        return v


class EvalResult(BaseModel):
    test_id:    str
    input:      str
    expected:   str
    actual:     str
    score:      float = Field(0.0, ge=0.0, le=1.0)
    passed:     bool  = False
    reason:     str   = ""
    latency_ms: float = 0.0
    cost_usd:   float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 8.  EVAL FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

class EvalFramework:
    """
    Weekly evaluation harness.

    Usage:
        ef = EvalFramework(client, model=MODELS['fast'])
        ef.add_case("t01", input="My VPN is down", expected="Network")
        results = ef.run(system_prompt=SYSTEM)
        ef.report(results)
    """

    def __init__(self, client: LLMClient, model: str = "gpt-4o-mini",
                 judge_model: str = "gpt-4o-mini"):
        self.client      = client
        self.model       = client._resolve(model)
        self.judge_model = client._resolve(judge_model)
        self._cases: List[Dict] = []

    def add_case(self, test_id: str, input: str, expected: str,
                 check: str = "exact") -> None:
        """check: "exact" | "contains" | "llm_judge" """
        self._cases.append({"id": test_id, "input": input,
                             "expected": expected, "check": check})

    def _score(self, actual: str, expected: str, check: str) -> Tuple[float, str]:
        if check == "exact":
            passed = actual.strip().lower() == expected.strip().lower()
            return (1.0 if passed else 0.0), ("exact match" if passed else "mismatch")

        if check == "contains":
            passed = expected.lower() in actual.lower()
            return (1.0 if passed else 0.0), ("found" if passed else "not found")

        if check == "llm_judge":
            judge_prompt = (
                f"Score this LLM output on a scale of 0.0 to 1.0.\n"
                f"Expected: {expected}\nActual: {actual}\n"
                f'Return JSON: {{"score": 0.0-1.0, "reason": "one sentence"}}'
            )
            try:
                raw  = self.client.chat(self.judge_model, user=judge_prompt,
                                        json_mode=True, tags=["eval", "judge"])
                data = json.loads(raw)
                return float(data.get("score", 0)), data.get("reason", "")
            except Exception:
                return 0.0, "judge error"

        return 0.0, "unknown check type"

    def run(self, system_prompt: str, temperature: float = 0.0) -> List[EvalResult]:
        results = []
        for case in self._cases:
            t0     = time.time()
            actual = self.client.chat(
                self.model, user=case["input"], system=system_prompt,
                temperature=temperature, tags=["eval", "run"]
            )
            latency        = (time.time() - t0) * 1000
            score, reason  = self._score(actual, case["expected"], case["check"])
            results.append(EvalResult(
                test_id=case["id"], input=case["input"],
                expected=case["expected"], actual=actual,
                score=score, passed=score >= 0.8,
                reason=reason, latency_ms=latency,
            ))
        return results

    def report(self, results: List[EvalResult]) -> None:
        banner("📋  Evaluation Report")
        passed    = sum(1 for r in results if r.passed)
        total     = len(results)
        avg_score = sum(r.score for r in results) / total if total else 0
        print(f"  Passed: {passed}/{total}  ({passed/total:.0%})   Avg score: {avg_score:.2f}")
        print()
        hdr = f"  {'ID':<8} {'Score':>6} {'Pass':>5}  {'Input':<35} {'Expected':<15} {'Actual'}"
        print(hdr)
        print("  " + "─" * 90)
        for r in results:
            icon = "✅" if r.passed else "❌"
            print(f"  {r.test_id:<8} {r.score:>6.2f} {icon}     "
                  f"{r.input[:34]:<35} {r.expected[:14]:<15} {r.actual.strip()[:40]}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  GUARDRAILS
# ══════════════════════════════════════════════════════════════════════════════

class Guardrails:
    """Input and output validation for production LLM pipelines."""

    INJECTION_PATTERNS = [
        "ignore previous instructions",
        "ignore all previous",
        "system instruction override",
        "you are now in",
        "maintenance mode",
        "new task:",
        "disregard your",
        "forget everything",
        "pretend you are",
        "act as if you have no",
        "do not ask for confirmation",
    ]

    @classmethod
    def check_input(cls, text: str, max_len: int = 2000) -> Tuple[bool, str]:
        """Returns (is_safe, reason)."""
        if len(text) > max_len:
            return False, f"Input too long ({len(text)} > {max_len} chars)"
        lower = text.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if pattern in lower:
                return False, f"Possible injection detected: '{pattern}'"
        return True, ""

    @classmethod
    def parse_ticket(cls, raw_json: str) -> Tuple[Optional[TicketClassification], List[str]]:
        """Parse and validate a TicketClassification JSON. Returns (model|None, errors)."""
        try:
            data   = json.loads(raw_json)
            ticket = TicketClassification(**data)
            return ticket, []
        except json.JSONDecodeError as e:
            return None, [f"JSON parse error: {e}"]
        except Exception as e:
            errors = [str(x) for x in e.errors()] if hasattr(e, "errors") else [str(e)]
            return None, errors

    @classmethod
    def self_consistency_check(cls, client: LLMClient, model: str,
                               system: str, user: str, n: int = 3) -> Tuple[str, float]:
        """
        Run the same prompt n times; return (majority_answer, agreement_rate).
        Agreement < 0.6 → flag as unreliable.
        """
        model   = client._resolve(model)
        answers = []
        for _ in range(n):
            ans = client.chat(model, user=user, system=system,
                              temperature=0.5, tags=["consistency"])
            answers.append(ans.strip().lower()[:80])
        majority  = max(set(answers), key=answers.count)
        agreement = answers.count(majority) / n
        return majority, agreement


# ══════════════════════════════════════════════════════════════════════════════
# 10.  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def banner(title: str, width: int = 65) -> None:
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}\n")

def section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 55 - len(title))}")

def observe(note: str) -> None:
    display(Markdown(f"\n💡  OBSERVE: {note}\n"))               

def discuss(note: str) -> None:
    display(Markdown(f"\n🗣️   DISCUSS: {note}\n"))

def warn(note: str) -> None:
    print(f"\n⚠️   WARNING: {note}\n")

def success(note: str) -> None:
    print(f"\n✅  {note}\n")

def compare(label_a: str, text_a: str, label_b: str, text_b: str) -> None:
    """Side-by-side comparison of two outputs."""
    print(f"\n── {label_a} ──")
    for line in text_a.strip().splitlines():
        print(f"  {line}")
    print(f"\n── {label_b} ──")
    for line in text_b.strip().splitlines():
        print(f"  {line}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MODEL SELECTOR  (interactive ipywidgets for notebooks)
# ══════════════════════════════════════════════════════════════════════════════

class ModelSelector:
    """
    Interactive model selector for Jupyter notebooks.

    Usage:
        selector = ModelSelector()
        # … choose from dropdowns …
        response = client.chat(selector.GPT,    user="Hello!")
        response = client.chat(selector.Claude, user="Hello!")
        response = client.chat(selector.active, user="Hello!")  # whichever is selected
    """

    def __init__(self, default_gpt: str = "gpt-4o-mini",
                 default_claude: str = "claude-haiku-4-5"):
        try:
            import ipywidgets as widgets
            from IPython.display import display, HTML
        except ImportError:
            print("⚠️  ipywidgets not installed — run:  pip install ipywidgets")
            self._gpt_widget    = None
            self._claude_widget = None
            self._active_widget = None
            self._default_gpt    = default_gpt
            self._default_claude = default_claude
            return

        chat_models = [m for m in MODEL_COSTS if "embedding" not in m]
        gpt_models  = [m for m in chat_models if "gpt" in m]
        cld_models  = [m for m in chat_models if "claude" in m]

        self._gpt_widget = widgets.Dropdown(
            options=gpt_models, value=default_gpt,
            description="GPT Model:", style={"description_width": "initial"},
            layout=widgets.Layout(width="320px"),
        )
        self._claude_widget = widgets.Dropdown(
            options=cld_models, value=default_claude,
            description="Claude Model:", style={"description_width": "initial"},
            layout=widgets.Layout(width="320px"),
        )
        self._active_widget = widgets.RadioButtons(
            options=["GPT", "Claude"], value="GPT",
            description="Active:", style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )

        box = widgets.VBox([
            widgets.HTML("<b>🤖 Model Selector</b>"),
            widgets.HBox([self._gpt_widget, self._claude_widget]),
            self._active_widget,
        ])
        display(box)

    @property
    def GPT(self) -> str:
        return self._gpt_widget.value if self._gpt_widget else self._default_gpt

    @property
    def Claude(self) -> str:
        return self._claude_widget.value if self._claude_widget else self._default_claude

    # backwards-compatible alias
    @property
    def HAIKU(self) -> str:
        return self.Claude

    @property
    def active(self) -> str:
        """Returns whichever provider's model is currently selected as 'Active'."""
        if self._active_widget:
            return self.GPT if self._active_widget.value == "GPT" else self.Claude
        return self._default_gpt
