"""P4: Enterprise support-ticket router with structured output, resilient API wrapper, prompt benchmark.

Pipeline
  raw email -> system prompt (zero-shot | few-shot | chain-of-thought, all enforce a JSON schema)
            -> resilient LLM call (rate limiting, retries, exponential backoff + jitter)
            -> JSON extraction -> Pydantic validation (with one "repair" retry on bad output)
            -> TicketResult(priority, category, draft_response)

Providers
  --provider mock        no API key needed; simulates a flaky LLM so you can test the pipeline
  --provider anthropic   real calls; needs `pip install anthropic` and ANTHROPIC_API_KEY

IMPORTANT: mock-provider accuracy numbers only prove the plumbing works. They say nothing about
how real models perform. Run with --provider anthropic for the benchmark you put in your report.

Needs: pip install pydantic   (and: pip install anthropic   for the real provider)
Run:   python p4_ticket_router.py --provider mock
       python p4_ticket_router.py --provider anthropic --limit 24
"""
import argparse
import csv
import json
import os
import random
import re
import statistics
import time
import zlib
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field, ValidationError

CATEGORIES = ["billing", "technical_api", "feature_request", "account_access"]
PRIORITIES = ["low", "medium", "high", "urgent"]
STYLES = ["zero_shot", "few_shot", "chain_of_thought"]


# ====================================================================== schema
class TicketResult(BaseModel):
    reasoning: Optional[str] = Field(default=None, max_length=1500)
    priority: Literal["low", "medium", "high", "urgent"]
    category: Literal["billing", "technical_api", "feature_request", "account_access"]
    draft_response: str = Field(min_length=10, max_length=700)


# ===================================================================== prompts
SCHEMA_RULES = f"""You are a support-ticket triage assistant for a B2B SaaS company.
Read the customer email and respond with ONE JSON object and nothing else: no markdown fences, no prose before or after.

Required keys:
  "priority": one of {PRIORITIES}
  "category": one of {CATEGORIES}
  "draft_response": a polite, concise reply (max 3 sentences) addressed to the customer

Category guide:
  billing          invoices, charges, refunds, plan or payment problems
  technical_api    API errors, timeouts, rate limits, webhooks, SDK bugs, outages
  feature_request  asks for new functionality or an improvement
  account_access   login, SSO, password reset, locked accounts, permissions

Priority guide:
  urgent  production is down or money is being lost right now
  high    a core workflow is blocked or a customer is charged incorrectly
  medium  degraded experience, workaround exists
  low     questions, suggestions, no business impact"""

FEW_SHOT_EXAMPLES = """
<examples>
Email: "Our production checkout is returning 500 from your /v2/orders endpoint since 09:00. We are losing sales."
{"priority": "urgent", "category": "technical_api", "draft_response": "I'm sorry about the outage. Our on-call engineers are investigating the /v2/orders errors now and I will update you within 30 minutes."}

Email: "We were charged twice for March. Please refund the duplicate invoice."
{"priority": "high", "category": "billing", "draft_response": "Thank you for flagging this. I have asked our billing team to verify the duplicate charge and issue a refund; you will receive a confirmation email shortly."}

Email: "It would be great if the dashboard could export reports as PDF."
{"priority": "low", "category": "feature_request", "draft_response": "Thanks for the suggestion. I have logged PDF export with our product team and will let you know if it is scheduled."}

Email: "Nobody on my team can sign in through SSO since this morning."
{"priority": "high", "category": "account_access", "draft_response": "Sorry for the disruption. Please confirm your SSO provider and I will have our identity team check your configuration right away."}
</examples>"""

COT_RULES = """
Before answering, reason step by step inside the JSON field "reasoning" (2-4 short sentences):
identify the business impact, then choose category and priority from the guides.
Put "reasoning" FIRST in the JSON object, then the three required keys."""


def build_system_prompt(style: str) -> str:
    if style == "zero_shot":
        return SCHEMA_RULES
    if style == "few_shot":
        return SCHEMA_RULES + "\n" + FEW_SHOT_EXAMPLES
    if style == "chain_of_thought":
        return SCHEMA_RULES + "\n" + COT_RULES
    raise ValueError(f"unknown style {style}")


def build_user_prompt(email: str) -> str:
    return f'Customer email:\n"""\n{email}\n"""'


# ================================================================ LLM clients
class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    pass


class TransientError(LLMError):
    pass


class LLMClient(Protocol):
    def complete(self, system: str, messages: list[dict]) -> str: ...


class AnthropicClient:
    def __init__(self, model: Optional[str] = None, max_tokens: int = 700):
        import anthropic  # imported lazily so the mock path needs no SDK
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set in this terminal.\n"
                "  PowerShell:  $env:ANTHROPIC_API_KEY=\"your-key\"   (then run the script again in the SAME window)\n"
                "  No key yet?  Get one at console.anthropic.com, or run with --provider mock.")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=0)  # we do our own retries
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict]) -> str:
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                system=system, messages=messages)
        except a.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except (a.APIConnectionError, a.APITimeoutError) as e:
            raise TransientError(str(e)) from e
        except a.APIStatusError as e:
            if e.status_code >= 500 or e.status_code == 529:
                raise TransientError(str(e)) from e
            raise  # 4xx such as bad key or bad request: retrying will not help
        return "".join(b.text for b in resp.content if b.type == "text")


class MockClient:
    """Deterministic fake LLM: keyword classifier plus injected failures.

    Per-style behaviour is invented to exercise the pipeline (fenced JSON, bad enum values,
    rate limits, a little extra latency for chain-of-thought). It is NOT evidence about real models.
    """
    KEYWORDS = {
        "billing": ["invoice", "charge", "charged", "refund", "payment", "billing", "plan", "subscription", "price"],
        "technical_api": ["api", "500", "timeout", "webhook", "endpoint", "sdk", "rate limit", "429", "error", "outage"],
        "feature_request": ["would be great", "feature", "please add", "wish", "support for", "could you add", "suggest"],
        "account_access": ["sso", "password", "log in", "login", "sign in", "locked", "permission", "2fa", "access"],
    }
    # Chance of a wrong label / of a formatting slip, by prompt style (simulation only).
    P_WRONG = {"zero_shot": 0.22, "few_shot": 0.10, "chain_of_thought": 0.06}
    P_FORMAT = {"zero_shot": 0.30, "few_shot": 0.08, "chain_of_thought": 0.12}
    LATENCY = {"zero_shot": 0.010, "few_shot": 0.014, "chain_of_thought": 0.030}

    def __init__(self, p_rate_limit=0.12, seed=0):
        self.rng = random.Random(seed)
        self.p_rate_limit = p_rate_limit

    @staticmethod
    def _style(system: str) -> str:
        if "reason step by step" in system:
            return "chain_of_thought"
        return "few_shot" if "<examples>" in system else "zero_shot"

    def complete(self, system: str, messages: list[dict]) -> str:
        style = self._style(system)
        email = messages[0]["content"].lower()
        is_retry = len(messages) > 1                      # repair attempt after bad output
        time.sleep(self.LATENCY[style])
        if self.rng.random() < self.p_rate_limit:
            raise RateLimitError("429 simulated rate limit")

        h = zlib.crc32((style + email).encode())          # stable per (style, email)
        r = random.Random(h)
        scores = {c: sum(k in email for k in kws) for c, kws in self.KEYWORDS.items()}
        category = max(scores, key=scores.get) if max(scores.values()) else "technical_api"
        if r.random() < self.P_WRONG[style]:
            category = r.choice([c for c in CATEGORIES if c != category])
        urgent = any(w in email for w in ["down", "production", "losing", "outage", "all customers", "locked out"])
        priority = "urgent" if urgent else ("high" if "charged" in email or "blocked" in email
                                            else "low" if category == "feature_request" else "medium")

        payload = {"priority": priority, "category": category,
                   "draft_response": "Thanks for contacting us. We are looking into this and will follow up shortly."}
        if style == "chain_of_thought":
            payload = {"reasoning": f"Impact looks {priority}; best fit category is {category}.", **payload}

        if not is_retry and r.random() < self.P_FORMAT[style]:
            slip = r.choice(["fence", "prose", "bad_enum"])
            if slip == "fence":
                return "```json\n" + json.dumps(payload) + "\n```"
            if slip == "prose":
                return "Sure! Here is the triage result:\n" + json.dumps(payload) + "\nHope this helps."
            payload["category"] = "Billing Issue"          # invalid enum -> triggers repair retry
        return json.dumps(payload)


# ========================================================= resilient wrapper
@dataclass
class CallStats:
    attempts: int = 0
    rate_limit_hits: int = 0
    transient_hits: int = 0
    repair_used: bool = False
    backoff_seconds: float = 0.0


@dataclass
class RouterConfig:
    max_retries: int = 5              # retries for rate-limit / transient errors
    base_delay: float = 1.0           # first backoff, doubles each retry
    max_delay: float = 30.0
    min_interval: float = 0.0         # client-side throttle between calls (seconds)
    sleep: callable = field(default=time.sleep)


class TicketRouter:
    def __init__(self, client: LLMClient, config: Optional[RouterConfig] = None):
        self.client = client
        self.cfg = config or RouterConfig()
        self._last_call = 0.0

    def _throttle(self):
        wait = self.cfg.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            self.cfg.sleep(wait)
        self._last_call = time.monotonic()

    def _call(self, system: str, messages: list[dict], stats: CallStats) -> str:
        """One logical LLM call with exponential backoff on 429 / transient failures."""
        for attempt in range(self.cfg.max_retries + 1):
            self._throttle()
            stats.attempts += 1
            try:
                return self.client.complete(system, messages)
            except (RateLimitError, TransientError) as e:
                if isinstance(e, RateLimitError):
                    stats.rate_limit_hits += 1
                else:
                    stats.transient_hits += 1
                if attempt == self.cfg.max_retries:
                    raise
                delay = min(self.cfg.max_delay, self.cfg.base_delay * 2 ** attempt)
                delay *= random.uniform(0.5, 1.0)          # jitter avoids thundering herd
                stats.backoff_seconds += delay
                self.cfg.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def extract_json(text: str) -> str:
        """Pull the first balanced {...} object out of a reply (handles fences / chatty prose)."""
        text = re.sub(r"```(?:json)?", "", text)
        start = text.find("{")
        if start == -1:
            raise ValueError("no JSON object found")
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        raise ValueError("unbalanced JSON object")

    def route(self, email: str, style: str = "few_shot") -> tuple[TicketResult, CallStats]:
        system = build_system_prompt(style)
        messages = [{"role": "user", "content": build_user_prompt(email)}]
        stats = CallStats()
        raw = self._call(system, messages, stats)
        try:
            return TicketResult.model_validate_json(self.extract_json(raw)), stats
        except (ValueError, ValidationError) as err:
            # One repair attempt: show the model its own output and the validation error.
            stats.repair_used = True
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That response failed validation: "
                 f"{str(err)[:400]}\nReturn ONLY the corrected JSON object."},
            ]
            raw = self._call(system, messages, stats)
            return TicketResult.model_validate_json(self.extract_json(raw)), stats


# ================================================================== benchmark
# (email, true category, true priority)
DATASET = [
    ("Our production API has been returning 500 errors for the last hour, all customers are affected.", "technical_api", "urgent"),
    ("The /v2/webhooks endpoint times out roughly one call in ten. We can work around it with retries.", "technical_api", "medium"),
    ("We keep hitting 429 rate limit errors during our nightly sync. Can the limit be raised?", "technical_api", "medium"),
    ("Python SDK 3.2 throws a TypeError when pagination returns an empty page.", "technical_api", "medium"),
    ("API outage in eu-west: checkout is down and we are losing orders right now.", "technical_api", "urgent"),
    ("Quick question: does the API return timestamps in UTC? The docs are unclear.", "technical_api", "low"),
    ("We were charged twice for October. Please refund the duplicate invoice.", "billing", "high"),
    ("Our invoice shows 50 seats but we only bought 40. Please correct it.", "billing", "medium"),
    ("Can you send last quarter's invoices as PDFs for our accounting team?", "billing", "low"),
    ("We downgraded our plan but were still charged the old price. Please fix and refund the difference.", "billing", "high"),
    ("Payment failed on our corporate card and the account is being suspended tomorrow.", "billing", "urgent"),
    ("What payment methods do you accept besides credit card?", "billing", "low"),
    ("It would be great if the dashboard could export reports as PDF.", "feature_request", "low"),
    ("Please add support for SAML group mapping; it would simplify onboarding for us.", "feature_request", "low"),
    ("We wish the audit log could be filtered by user. Could you add that?", "feature_request", "low"),
    ("Feature suggestion: dark mode for the admin console.", "feature_request", "low"),
    ("Could you add a bulk-edit option for tags? Doing it one by one takes hours.", "feature_request", "medium"),
    ("Would be nice to get Slack notifications when a report finishes. Just a suggestion.", "feature_request", "low"),
    ("Nobody on my team can sign in through SSO since this morning. Everyone is locked out.", "account_access", "urgent"),
    ("I forgot my password and the reset email never arrives.", "account_access", "medium"),
    ("New hire cannot see the analytics workspace. Their permissions seem wrong.", "account_access", "medium"),
    ("Our admin account is locked after too many login attempts and we need access to approve payroll today.", "account_access", "high"),
    ("2FA codes are not being accepted for one of our users.", "account_access", "medium"),
    ("How do I add another admin to our account?", "account_access", "low"),
]


def run_benchmark(router: TicketRouter, styles, dataset, label: str):
    rows = []
    for style in styles:
        cat_ok = pri_ok = failed = repairs = rl = 0
        lat = []
        for email, true_cat, true_pri in dataset:
            t0 = time.perf_counter()
            try:
                res, st = router.route(email, style)
            except Exception as e:                 # unrecoverable after retries / repair
                failed += 1
                print(f"  [{style}] failed: {type(e).__name__}: {str(e)[:80]}")
                continue
            lat.append(time.perf_counter() - t0)
            cat_ok += res.category == true_cat
            pri_ok += res.priority == true_pri
            repairs += st.repair_used
            rl += st.rate_limit_hits
        n = len(dataset)
        rows.append({
            "provider": label, "prompt_style": style, "n": n,
            "category_acc": round(cat_ok / n, 3),
            "priority_acc": round(pri_ok / n, 3),
            "failures": failed, "repair_retries": repairs, "rate_limit_hits": rl,
            "mean_latency_s": round(statistics.mean(lat), 3) if lat else None,
            "p95_latency_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3) if lat else None,
        })
    return rows


def print_matrix(rows):
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    ap.add_argument("--limit", type=int, default=len(DATASET), help="tickets to benchmark")
    ap.add_argument("--out", default="p4_benchmark.csv")
    args = ap.parse_args()

    if args.provider == "mock":
        client = MockClient()
        cfg = RouterConfig(base_delay=0.01, max_delay=0.1)     # fast backoff for the simulation
        label = "mock (simulated)"
    else:
        client = AnthropicClient()
        cfg = RouterConfig(min_interval=0.2)
        label = client.model

    router = TicketRouter(client, cfg)
    data = DATASET[:args.limit]
    print(f"Benchmarking {len(data)} tickets x {len(STYLES)} prompt styles on: {label}\n")

    # Demo: one ticket, printed in full
    demo, st = router.route(data[0][0], "few_shot")
    print("Sample result:", demo.model_dump_json(indent=2))
    print(f"(attempts={st.attempts}, rate_limit_hits={st.rate_limit_hits}, repair={st.repair_used})\n")

    rows = run_benchmark(router, STYLES, data, label)
    print("\n=== Prompt benchmark matrix ===")
    print_matrix(rows)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {args.out}")
    if args.provider == "mock":
        print("NOTE: mock numbers are simulated. Re-run with --provider anthropic for real results.")


if __name__ == "__main__":
    main()