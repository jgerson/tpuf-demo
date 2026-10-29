"""
turbopuffer playground: security questionnaire assistant
========================================================

A small hobby web app for exploring what turbopuffer can do. I wanted to get
hands-on with hybrid search, built-in embeddings and reranking, and the
one-namespace-per-customer pattern, so this wraps them in a toy app.

Tabs
  1. Start here     load the sample data, see what's in each namespace
  2. Search         meaning, keyword, hybrid and reranked search, with filters,
                    the exact API call, and the cost of every query
  3. Auto-fill      draft answers to a whole questionnaire with Claude
  4. Real docs      load HashiCorp's public Vault documentation (thousands of passages)
  5. Many tenants   create hundreds of customer namespaces, test cold vs warm

Secrets (Streamlit Community Cloud: App settings -> Secrets, in TOML format)
  TURBOPUFFER_API_KEY   required
  APP_PASSWORD          recommended, protects the page
  ANTHROPIC_API_KEY     optional, enables Claude-drafted answers in Auto-fill
  GITHUB_TOKEN          optional, only if GitHub rate-limits the docs loader

All company names and answers in the sample libraries are fictional.
"""

import concurrent.futures as cf
import json
import os
import random
import re
import threading
import time

import pandas as pd
import requests
import streamlit as st
import turbopuffer

st.set_page_config(page_title="turbopuffer playground", page_icon="🐡", layout="wide")

# Streamlit keeps secrets in st.secrets. Copy them into environment variables
# so the rest of the code can read them with os.getenv, the same as in Colab.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # no secrets file (for example when running locally with env vars)


class AppError(Exception):
    """An error with a message that is safe to show on the page."""

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
REGION = os.getenv("TURBOPUFFER_REGION", "gcp-us-central1")
PREFIX = os.getenv("NAMESPACE_PREFIX", "tpdemo")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-8b")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "1024"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "zeroentropy/zerank-2")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
VAULT_VERSION = os.getenv("VAULT_DOCS_VERSION", "v2.x")

MAX_AUTOFILL_QUESTIONS = 30
MAX_TENANTS = 1000
MAX_VAULT_PAGES = 700

# Price estimates for the cost meter, from turbopuffer's public pricing log
# (Sep 2026). Storage and write rates are the "up to" ceilings on the pricing
# page; embedding and reranking charges are not included.
QUERY_USD_PER_BYTE = 1.0 / 1e15          # $1 per PB queried
MIN_QUERY_BYTES = 1.28e9                 # minimum billed per query
WRITE_USD_PER_GB = 2.00                  # ceiling
STORAGE_USD_PER_GB_MONTH = 0.33          # ceiling

NS = {
    "Lumen Ledger (fintech SaaS)": f"{PREFIX}-lumen-ledger",
    "Harbor Health (healthcare)": f"{PREFIX}-harbor-health",
    "HashiCorp Vault docs": f"{PREFIX}-vault-docs",
}
ANSWER_LIBRARIES = ["Lumen Ledger (fintech SaaS)", "Harbor Health (healthcare)"]
TENANT_PREFIX = f"{PREFIX}-tenant-"

SCHEMA = {
    "content": {
        "type": "string",
        "full_text_search": True,
        "embed": {"model": EMBED_MODEL, "dims": EMBED_DIMS},
    }
}

# ----------------------------------------------------------------------------
# Sample data (fictional)
# ----------------------------------------------------------------------------
# (id, category, controls, reviewed, question, answer)
LUMEN = [
    (1, "Encryption", "SOC 2 CC6.1; ISO A.8.24", 2026, "Is customer data encrypted at rest?",
     "All production databases, object storage buckets and backups are encrypted with AES-256. Keys are managed in AWS KMS and rotated every 365 days."),
    (2, "Encryption", "SOC 2 CC6.7", 2026, "Is data encrypted in transit?",
     "All external traffic requires TLS 1.2 or higher; TLS 1.0 and 1.1 are disabled. Service-to-service traffic inside our VPC uses mutual TLS."),
    (3, "Encryption", "ISO A.8.24", 2025, "Can customers bring their own encryption keys?",
     "Enterprise plans support customer-managed keys through AWS KMS. Revoking the key makes the customer's data unreadable within minutes."),
    (4, "Access control", "SOC 2 CC6.1; ISO A.5.15", 2026, "How is employee access to production controlled?",
     "Production access is role-based and granted just-in-time through an approval workflow. Sessions expire after 8 hours and every command is logged."),
    (5, "Access control", "SOC 2 CC6.1; ISO A.8.5", 2026, "Is multi-factor authentication required?",
     "Hardware security keys are mandatory for all employees. Customers can enforce MFA for their users and connect SAML or OIDC single sign-on."),
    (6, "Access control", "SOC 2 CC6.2", 2025, "How quickly is access removed when an employee leaves?",
     "Offboarding is automated from our HR system; all accounts and tokens are revoked within 4 hours of termination."),
    (7, "Access control", "SOC 2 CC6.3", 2025, "How often are user access rights reviewed?",
     "Managers certify their team's access every quarter; unused entitlements are removed automatically after 60 days."),
    (8, "Data retention", "SOC 2 C1.2; ISO A.8.10", 2026, "How long do you retain customer data after contract termination?",
     "Customer data is deleted 30 days after termination and purged from backups within a further 35 days. A deletion certificate is available on request."),
    (9, "Data retention", "ISO A.8.10", 2026, "Can a customer request deletion of specific records?",
     "Admins can delete individual records through the API or console; deletions reach search indexes immediately and backups within 35 days."),
    (10, "Availability", "SOC 2 A1.2; ISO A.8.13", 2026, "Do you back up customer data?",
     "Snapshots are taken every hour and replicated to a second AWS region. Restores are tested every quarter."),
    (11, "Availability", "SOC 2 A1.2", 2026, "What are your recovery time and recovery point objectives?",
     "Our RTO is 4 hours and our RPO is 1 hour for the core ledger service."),
    (12, "Availability", "SOC 2 A1.1", 2025, "What uptime do you commit to?",
     "We offer a 99.95% monthly uptime SLA with service credits; historical uptime is published on our status page."),
    (13, "Incident response", "SOC 2 CC7.3; ISO A.5.24", 2026, "Do you have an incident response plan?",
     "A documented plan defines severity levels, on-call roles and communication steps. It is exercised with tabletop drills twice a year."),
    (14, "Incident response", "SOC 2 CC7.4", 2026, "How quickly will you notify us of a breach?",
     "Affected customers are notified within 72 hours of confirming a security incident involving their data, or sooner where the law requires."),
    (15, "Incident response", "SOC 2 CC7.2; ISO A.8.15", 2025, "How do you detect suspicious activity?",
     "Logs from every service flow into a central SIEM with alerts on anomalous logins, privilege changes and data exports. Logs are kept for one year."),
    (16, "Compliance", "SOC 2", 2026, "Do you have a SOC 2 Type II report?",
     "Yes. Our SOC 2 Type II report covers security, availability and confidentiality and is renewed annually. It is shared under NDA."),
    (17, "Compliance", "ISO 27001", 2026, "Are you ISO 27001 certified?",
     "We are certified to ISO/IEC 27001:2022; the certificate and Statement of Applicability are available in our trust center."),
    (18, "Compliance", "GDPR", 2025, "Do you sign a data processing agreement?",
     "Our standard DPA includes the EU Standard Contractual Clauses and lists all subprocessors."),
    (19, "Privacy", "GDPR; ISO A.5.34", 2026, "Where is customer data stored?",
     "Customers choose a US (Virginia) or EU (Frankfurt) region at signup, and data stays in that region."),
    (20, "Privacy", "SOC 2 CC9.2; ISO A.5.19", 2025, "How do you manage third-party vendors?",
     "Every vendor handling customer data passes a security review before contract and is reassessed annually; the subprocessor list is public."),
    (21, "Privacy", "ISO A.5.34", 2026, "Do you use customer data to train AI models?",
     "No. Customer data is never used to train models, ours or any vendor's, and our AI providers are contractually barred from retaining it."),
    (22, "Application security", "SOC 2 CC8.1", 2026, "How are code changes reviewed before release?",
     "Every change requires peer review, passing automated tests and a security scan before it can merge; deployments are logged and reversible."),
    (23, "Application security", "SOC 2 CC7.1", 2025, "Do you run penetration tests?",
     "An independent firm performs a penetration test every year, and critical findings are fixed within 14 days. A summary letter is available."),
    (24, "Application security", "SOC 2 CC7.1", 2025, "Do you have a bug bounty program?",
     "We run a private bug bounty with paid rewards and a public vulnerability disclosure policy."),
]

HARBOR = [
    (1, "Compliance", "HIPAA 164.504(e)", 2026, "Will you sign a business associate agreement?",
     "Yes. We sign a BAA with every customer that processes protected health information."),
    (2, "Encryption", "HIPAA 164.312(a)(2)(iv)", 2026, "Is PHI encrypted?",
     "Protected health information is encrypted with AES-256 at rest and TLS 1.3 in transit."),
    (3, "Data retention", "HIPAA 164.316", 2026, "How long are medical records retained?",
     "Records are retained for seven years after the last patient encounter, then securely destroyed."),
    (4, "Access control", "HIPAA 164.312(b)", 2026, "Is access to patient data logged?",
     "Every view, export and change to a patient record is written to an immutable audit log that customers can download."),
    (5, "Access control", "HIPAA 164.502(b)", 2025, "How do you enforce minimum necessary access?",
     "Clinical staff see only the records for their assigned care teams; broader access requires a documented break-glass reason."),
    (6, "Incident response", "HIPAA 164.410", 2026, "How quickly do you report a breach of PHI?",
     "We notify covered entities without unreasonable delay and no later than 30 days after discovering a breach of unsecured PHI."),
    (7, "Compliance", "HITRUST", 2026, "Are you HITRUST certified?",
     "We hold HITRUST r2 certification, renewed every two years, with interim assessments in between."),
    (8, "Privacy", "HIPAA", 2026, "Where is patient data hosted?",
     "All patient data is hosted in US regions only and never leaves the United States."),
]

EXAMPLES = {
    "Lumen Ledger (fintech SaaS)": [
        "If your data center floods, can you get our information back?",
        "CC6.1",
        "What's your RPO if a whole region goes down?",
        "Will our data end up training an AI model?",
    ],
    "Harbor Health (healthcare)": [
        "Do you sign a BAA?",
        "Who can see a patient's chart?",
        "164.312",
    ],
    "HashiCorp Vault docs": [
        "how do I rotate the root token",
        "AppRole secret_id",
        "unseal with a cloud KMS instead of key shares",
        "why did my lease expire",
    ],
}

SAMPLE_QUESTIONNAIRE = """Do you encrypt client information when it's stored?
What are your RPO and RTO?
Is MFA enforced for your staff?
If we terminate, how quickly is our data destroyed?
Which answers map to SOC 2 CC6.1?
Do you have a documented incident response process, and how often is it tested?
Can we keep our data in the EU?
Will our data be used to train your AI?
Do you perform third-party penetration testing?
How do you vet subprocessors?
Are you FedRAMP authorized?
Do you carry cyber insurance, and what is the coverage limit?"""


def library_rows(items):
    # Each answer is stored once. "content" holds everything searchable (question,
    # answer and control IDs) and is what turbopuffer embeds and keyword-indexes.
    # The question is also kept as a short "title" for display; the answer text is
    # read back out of "content" (see body_of) instead of being stored twice.
    return [
        {
            "id": i,
            "content": f"{q}\n{a}\nControls: {c}",
            "title": q,
            "category": cat,
            "controls": c,
            "reviewed": yr,
        }
        for (i, cat, c, yr, q, a) in items
    ]


def categories_for(library):
    if library == "HashiCorp Vault docs":
        return ["Any", "auth", "secrets", "concepts", "configuration", "commands",
                "platform", "enterprise", "agent-and-proxy", "audit", "api-docs"]
    items = LUMEN if library.startswith("Lumen") else HARBOR
    return ["Any"] + sorted({x[1] for x in items})


# ----------------------------------------------------------------------------
# turbopuffer client and helpers
# ----------------------------------------------------------------------------
def by_meaning(q):
    return ("content", "ANN", ("Embed", q))


def by_keyword(q):
    return ("content", "BM25", q)


def query_cost(bytes_queried):
    """Estimated $ for one query, including the minimum and tier discounts."""
    b = max(float(bytes_queried or 0), MIN_QUERY_BYTES)
    tier1 = min(b, 32e9)
    tier2 = min(max(b - 32e9, 0), 96e9) * 0.20
    tier3 = max(b - 128e9, 0) * 0.04
    return (tier1 + tier2 + tier3) * QUERY_USD_PER_BYTE


def fmt_usd(x):
    if x == 0:
        return "$0"
    if x < 0.01:
        return f"${x:.7f}".rstrip("0")
    return f"${x:,.2f}"


def fmt_bytes(b):
    b = float(b or 0)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1000:
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.2f} {unit}"
        b /= 1000
    return f"{b:.2f} PB"


def new_meter():
    return {"queries": 0, "q_bytes": 0.0, "q_usd": 0.0, "w_bytes": 0.0, "w_usd": 0.0, "claude_tokens": 0}


def meter_md(m):
    total = m["q_usd"] + m["w_usd"]
    parts = [
        f"**This session:** {m['queries']} queries ({fmt_usd(m['q_usd'])})",
        f"{fmt_bytes(m['w_bytes'])} written ({fmt_usd(m['w_usd'])})",
        f"**est. turbopuffer total {fmt_usd(total)}**",
    ]
    if m["claude_tokens"]:
        parts.append(f"Claude: {m['claude_tokens']:,} tokens")
    return " · ".join(parts)


def charge_query(m, res):
    bill = getattr(res, "billing", None)
    b = getattr(bill, "billable_logical_bytes_queried", None) if bill else None
    usd = query_cost(b)
    m["queries"] += 1
    m["q_bytes"] += float(b or 0)
    m["q_usd"] += usd
    return usd, b


def charge_write(m, res):
    bill = getattr(res, "billing", None)
    b = float(getattr(bill, "billable_logical_bytes_written", 0) or 0) if bill else 0.0
    usd = b / 1e9 * WRITE_USD_PER_GB
    m["w_bytes"] += b
    m["w_usd"] += usd
    return usd, b


def perf_line(res, client_ms, usd, billed_bytes, label=""):
    p = getattr(res, "performance", None)
    bits = []
    if label:
        bits.append(f"**{label}**")
    if p is not None and getattr(p, "server_total_ms", None) is not None:
        bits.append(f"server {p.server_total_ms} ms")
    bits.append(f"round trip {client_ms:.0f} ms")
    if p is not None and getattr(p, "embedding_ms", None):
        bits.append(f"embedding {p.embedding_ms} ms")
    if p is not None and getattr(p, "cache_temperature", None):
        bits.append(f"cache **{p.cache_temperature}**")
    bits.append(f"billed {fmt_bytes(max(float(billed_bytes or 0), MIN_QUERY_BYTES))}")
    bits.append(f"cost ≈ {fmt_usd(usd)}")
    return " · ".join(bits)


def body_of(content):
    """The display text of a document: its content minus the title line (and the
    trailing "Controls:" line on answer-library documents)."""
    lines = (content or "").split("\n")[1:]
    if lines and lines[-1].startswith("Controls: "):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def row_dict(r):
    d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
    score = d.get("$dist", d.get("$score"))
    d["_score"] = round(float(score), 4) if isinstance(score, (int, float)) else ""
    d["body"] = body_of(d.get("content"))
    return d


def rows_table(rows, score_name="score"):
    data = [row_dict(r) for r in rows]
    out = []
    for i, d in enumerate(data, 1):
        out.append({
            "#": i,
            score_name: d["_score"],
            "Title": d.get("title", ""),
            "Text": d["body"][:220] + ("…" if len(d["body"]) > 220 else ""),
            "Category": d.get("category", ""),
            "Controls / source": d.get("controls") or d.get("url") or "",
        })
    cols = ["#", score_name, "Title", "Text", "Category", "Controls / source"]
    return pd.DataFrame(out, columns=cols)


# Fields to return with search results. The two kinds of namespace store
# slightly different fields, so each gets its own list.
ANSWER_ATTRS = ["title", "content", "category", "controls", "reviewed"]
DOCS_ATTRS = ["title", "content", "category", "url"]


def attrs_for(library):
    return ANSWER_ATTRS if library in ANSWER_LIBRARIES else DOCS_ATTRS


def build_filters(library, category, min_year):
    f = []
    if category and category != "Any":
        f.append(("category", "Eq", category))
    if library in ANSWER_LIBRARIES and min_year and int(min_year) > 2024:
        f.append(("reviewed", "Gte", int(min_year)))
    if not f:
        return None
    return f[0] if len(f) == 1 else ("And", tuple(f))


def code_for(ns_name, mode, q, top_k, filters, attrs):
    fl = f"\n    filters={filters!r}," if filters else ""
    q_ = json.dumps(q)
    if mode.startswith("Meaning"):
        return (f'ns = tpuf.namespace("{ns_name}")\nres = ns.query(\n    rank_by=("content", "ANN", ("Embed", {q_})),\n'
                f'    top_k={top_k},{fl}\n    include_attributes={attrs!r},\n)')
    if mode.startswith("Keyword"):
        return (f'ns = tpuf.namespace("{ns_name}")\nres = ns.query(\n    rank_by=("content", "BM25", {q_}),\n'
                f'    top_k={top_k},{fl}\n    include_attributes={attrs!r},\n)')
    rr = '("RRF",)' if mode.startswith("Hybrid (fused") else (
        f'{{"model": "{RERANK_MODEL}", "attribute": "content", "query": {q_}}}')
    sub = f'"top_k": {top_k * 3}' + (f', "filters": {filters!r}' if filters else "")
    return (f'ns = tpuf.namespace("{ns_name}")\nres = ns.multi_query(\n    queries=[\n'
            f'        {{"rank_by": ("content", "ANN", ("Embed", {q_})), {sub}}},\n'
            f'        {{"rank_by": ("content", "BM25", {q_}), {sub}}},\n    ],\n    rerank_by={rr},\n)')


def run_search(ns, mode, q, top_k, filters, include):
    """Returns (rows, response, client_ms, note)."""
    base = {"top_k": int(top_k), "include_attributes": include}
    if filters:
        base["filters"] = filters
    t0 = time.time()
    note = ""
    if mode.startswith("Meaning"):
        res = ns.query(rank_by=by_meaning(q), **base)
        rows = res.rows
    elif mode.startswith("Keyword"):
        res = ns.query(rank_by=by_keyword(q), **base)
        rows = res.rows
    else:
        sub = dict(base, top_k=int(top_k) * 3)
        queries = [dict(sub, rank_by=by_meaning(q)), dict(sub, rank_by=by_keyword(q))]
        if mode.startswith("Hybrid + rerank"):
            try:
                res = ns.multi_query(queries=queries, rerank_by={"model": RERANK_MODEL, "attribute": "content", "query": q})
            except Exception as e:
                note = f"Reranking isn't available on this account ({str(e)[:120]}). Showing fused results instead."
                res = ns.multi_query(queries=queries, rerank_by=("RRF",))
        else:
            res = ns.multi_query(queries=queries, rerank_by=("RRF",))
        rows = res.results[0].rows[: int(top_k)]
    return rows, res, (time.time() - t0) * 1000, note


def friendly(e):
    s = str(e)
    if "404" in s or "not found" in s.lower():
        return "That namespace doesn't exist yet. Load its data on the Start here or Real docs tab first."
    if "401" in s or "403" in s or "unauthor" in s.lower():
        return "turbopuffer rejected the API key. Check the TURBOPUFFER_API_KEY secret."
    return s[:300]


# Streamlit reruns this whole file on every click, so the client is cached
# once per server instead of being rebuilt each time.
@st.cache_resource(show_spinner=False)
def _make_client(key):
    return turbopuffer.Turbopuffer(api_key=key, region=REGION)


def tpuf():
    key = os.getenv("TURBOPUFFER_API_KEY")
    if not key:
        raise AppError("Add your turbopuffer API key as the secret TURBOPUFFER_API_KEY (App settings → Secrets), then reboot the app.")
    return _make_client(key)


def no_progress(frac, desc=""):
    pass


# ----------------------------------------------------------------------------
# Tab 1: Start here
# ----------------------------------------------------------------------------
def status_table():
    c = tpuf()
    out = []
    for label, name in NS.items():
        try:
            md = c.namespace(name).metadata()
            rows = getattr(md, "approx_row_count", None)
            size = getattr(md, "approx_logical_bytes", None)
            out.append({"Library": label, "Namespace": name, "Documents": str(rows),
                        "Size": fmt_bytes(size), "Storage / month ≈": fmt_usd((size or 0) / 1e9 * STORAGE_USD_PER_GB_MONTH)})
        except Exception:
            out.append({"Library": label, "Namespace": name, "Documents": "not loaded", "Size": "", "Storage / month ≈": ""})
    return pd.DataFrame(out)


def load_samples(meter):
    c = tpuf()
    lines = []
    for label, items in [("Lumen Ledger (fintech SaaS)", LUMEN), ("Harbor Health (healthcare)", HARBOR)]:
        t0 = time.time()
        res = c.namespace(NS[label]).write(upsert_rows=library_rows(items), distance_metric="cosine_distance", schema=SCHEMA)
        usd, b = charge_write(meter, res)
        lines.append(f"- **{label}**: {len(items)} answers written and embedded in {(time.time() - t0) * 1000:.0f} ms "
                     f"({fmt_bytes(b)}, ≈ {fmt_usd(usd)})")
    return "\n".join(lines) + "\n\nNothing had to be set up first: each namespace was created by its first write."


# ----------------------------------------------------------------------------
# Tab 2: Search
# ----------------------------------------------------------------------------
MODES = ["Meaning (vector)", "Keyword (BM25)", "Hybrid (fused)", "Hybrid + rerank"]


def do_search(library, q, mode, top_k, category, min_year, meter):
    if not (q or "").strip():
        raise AppError("Type a question first.")
    ns_name = NS[library]
    filters = build_filters(library, category, min_year)
    try:
        attrs = attrs_for(library)
        rows, res, ms, note = run_search(tpuf().namespace(ns_name), mode, q, top_k, filters, attrs)
    except AppError:
        raise
    except Exception as e:
        raise AppError(friendly(e))
    usd, b = charge_query(meter, res)
    score = "distance ↓" if mode.startswith("Meaning") else ("BM25 ↑" if mode.startswith("Keyword") else "score ↑")
    explain = {"Meaning": "Score is vector distance: lower means closer in meaning.",
               "Keyword": "Score is BM25 relevance: higher means a stronger keyword match.",
               "Hybrid (": "Vector and keyword results fused with reciprocal rank fusion.",
               "Hybrid +": "Vector and keyword candidates reordered by a reranking model."}
    stats = perf_line(res, ms, usd, b) + "  \n" + next(v for k, v in explain.items() if mode.startswith(k))
    if note:
        stats = f"⚠️ {note}\n\n{stats}"
    return {"table": rows_table(rows, score), "stats": stats, "code": code_for(ns_name, mode, q, int(top_k), filters, attrs)}


def do_compare(library, q, top_k, category, min_year, meter):
    if not (q or "").strip():
        raise AppError("Type a question first.")
    ns = tpuf().namespace(NS[library])
    filters = build_filters(library, category, min_year)
    cols, lines = {}, []
    for mode in MODES:
        try:
            rows, res, ms, note = run_search(ns, mode, q, top_k, filters, include=["title"])
        except Exception as e:
            raise AppError(friendly(e))
        usd, b = charge_query(meter, res)
        cols[mode] = [row_dict(r).get("title", "") for r in rows]
        lines.append(perf_line(res, ms, usd, b, label=mode) + (f" ({note})" if note else ""))
    n = max((len(v) for v in cols.values()), default=0)
    df = pd.DataFrame({"#": list(range(1, n + 1)), **{k: v + [""] * (n - len(v)) for k, v in cols.items()}})
    return {"table": df, "stats": "\n\n".join(lines), "code": None}


# ----------------------------------------------------------------------------
# Tab 3: Auto-fill a questionnaire
# ----------------------------------------------------------------------------
CLAUDE_SYSTEM = (
    "You draft answers to security questionnaires for a software company. Use ONLY the approved answers provided. "
    "Do not invent facts, numbers or certifications. If the approved answers don't fully answer the question, "
    "say what is missing and set confidence to low. Keep answers to 1-3 sentences in a confident, plain style. "
    'Reply with JSON only: {"answer": str, "source_ids": [int], "confidence": "high"|"medium"|"low", "gap": str}'
)


def parse_questions(text, upload):
    """upload is a Streamlit UploadedFile (or None)."""
    qs = []
    if upload is not None:
        raw = upload.getvalue()
        try:
            import io
            df = pd.read_csv(io.BytesIO(raw))
            col = next((c for c in df.columns if str(c).strip().lower() in ("question", "questions")), df.columns[0])
            qs = [str(x).strip() for x in df[col].dropna().tolist()]
        except Exception:
            qs = [ln.strip() for ln in raw.decode("utf-8", errors="ignore").splitlines() if ln.strip()]
    if not qs:
        qs = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return qs[:MAX_AUTOFILL_QUESTIONS]


def ask_claude(client, question, sources):
    src = "\n\n".join(f"[{s['id']}] Q: {s.get('title', '')}\nA: {s.get('body', '')}\nControls: {s.get('controls', '')}" for s in sources)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=CLAUDE_SYSTEM,
        messages=[{"role": "user", "content": f"Question from the customer's questionnaire:\n{question}\n\nApproved answers:\n{src}"}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content)
    tokens = (getattr(msg.usage, "input_tokens", 0) or 0) + (getattr(msg.usage, "output_tokens", 0) or 0)
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0)) if m else {"answer": text.strip(), "source_ids": [], "confidence": "low", "gap": "Unparsed reply"}
    return data, tokens


def autofill(library, text, upload, meter, progress=no_progress):
    qs = parse_questions(text, upload)
    if not qs:
        raise AppError("Paste some questions (one per line) or upload a CSV.")
    ns = tpuf().namespace(NS[library])
    claude = None
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        claude = anthropic.Anthropic()
    lock = threading.Lock()
    t_start = time.time()

    def one(i, q):
        queries = [
            {"rank_by": by_meaning(q), "top_k": 8, "include_attributes": ["title", "content", "controls"]},
            {"rank_by": by_keyword(q), "top_k": 8, "include_attributes": ["title", "content", "controls"]},
        ]
        try:
            res = ns.multi_query(queries=queries, rerank_by={"model": RERANK_MODEL, "attribute": "content", "query": q})
        except Exception:
            res = ns.multi_query(queries=queries, rerank_by=("RRF",))
        top = [row_dict(r) for r in res.results[0].rows[:4]]
        vec = ns.query(rank_by=by_meaning(q), top_k=1)
        best_dist = row_dict(vec.rows[0])["_score"] if vec.rows else 1.0
        with lock:
            charge_query(meter, res)
            charge_query(meter, vec)
        tokens = 0
        if claude is not None:
            try:
                data, tokens = ask_claude(claude, q, top)
            except Exception as e:
                data = {"answer": f"(Claude call failed: {str(e)[:120]})", "source_ids": [], "confidence": "low", "gap": ""}
        else:
            # No LLM: reuse the best approved answer; confidence from vector distance (a rough heuristic)
            conf = "high" if best_dist < 0.30 else "medium" if best_dist < 0.45 else "low"
            data = {"answer": top[0]["body"] if top else "", "source_ids": [top[0]["id"]] if top else [],
                    "confidence": conf, "gap": "" if conf == "high" else "Closest approved answer may not fully match"}
        with lock:
            meter["claude_tokens"] += tokens
        by_id = {d["id"]: d for d in top}
        srcs = [by_id[s]["title"] for s in data.get("source_ids", []) if s in by_id] or ([top[0]["title"]] if top else [])
        conf = str(data.get("confidence", "low")).lower()
        return {
            "#": i + 1,
            "Question": q,
            "Draft answer": data.get("answer", ""),
            "Confidence": conf,
            "Review?": "yes" if conf != "high" else "",
            "Based on": "; ".join(srcs),
            "Gap": data.get("gap", ""),
        }

    results = []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(one, i, q) for i, q in enumerate(qs)]
        for n, f in enumerate(cf.as_completed(futs), 1):
            progress(n / len(qs), f"Drafted {n} of {len(qs)}")
            try:
                results.append(f.result())
            except Exception as e:
                raise AppError(friendly(e))
    df = pd.DataFrame(sorted(results, key=lambda r: r["#"]))
    flagged = int((df["Review?"] == "yes").sum())
    secs = time.time() - t_start
    engine = f"Claude ({CLAUDE_MODEL})" if claude else "best-match only (add ANTHROPIC_API_KEY for drafted answers)"
    summary = (f"**{len(qs)} questions in {secs:.1f} s** · {len(qs) - flagged} ready to send · **{flagged} flagged for review** · "
               f"drafting: {engine}\n\nRetrieval for the whole questionnaire cost about "
               f"{fmt_usd(query_cost(0) * 2 * len(qs))} in turbopuffer queries.")
    return df, summary


# ----------------------------------------------------------------------------
# Tab 4: Real docs (HashiCorp Vault, public)
# ----------------------------------------------------------------------------
GH_REPO = "hashicorp/web-unified-docs"
RAW = "https://raw.githubusercontent.com/hashicorp/web-unified-docs/main/"


def gh_get(url):
    headers = {"Accept": "application/vnd.github+json"}
    if os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        raise AppError("GitHub rate-limited this server. Add a GITHUB_TOKEN secret (any read-only token) and try again.")
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def vault_files(version=VAULT_VERSION):
    base = f"content/vault/{version}/content"
    listing = gh_get(f"https://api.github.com/repos/{GH_REPO}/contents/{base}")
    files = []
    for entry in listing:
        if entry.get("type") == "dir" and entry["name"] in ("docs", "api-docs"):
            tree = gh_get(f"https://api.github.com/repos/{GH_REPO}/git/trees/{entry['sha']}?recursive=1")
            files += [f"{base}/{entry['name']}/{t['path']}" for t in tree.get("tree", [])
                      if t.get("type") == "blob" and t["path"].endswith((".mdx", ".md"))]
    return sorted(files)


def clean_mdx(text):
    title = ""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        t = re.search(r"^page_title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip("'\"") if t else ""
        text = text[m.end():]
    text = re.sub(r"^\s*(import|export) .*$", "", text, flags=re.M)
    text = re.sub(r"^\s*@include.*$", "", text, flags=re.M)
    text = re.sub(r"</?[A-Z][A-Za-z]*[^>]*>", "", text)           # MDX components
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)           # links -> text
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text.strip()


def chunk_page(path, raw, max_chars=1600):
    title, body = clean_mdx(raw)
    rel = path.split(f"/{VAULT_VERSION}/content/", 1)[-1]
    section = "api-docs" if rel.startswith("api-docs") else (rel.split("/")[1] if rel.count("/") >= 2 else "docs")
    section = section.replace(".mdx", "")
    url_path = re.sub(r"(/index)?\.mdx?$", "", rel)
    url = f"https://developer.hashicorp.com/vault/{url_path}"
    # split on markdown headings, ignoring "#" lines inside code blocks
    parts, cur, in_code = [], [], False
    for line in body.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,3} ", line) and cur:
            parts.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        parts.append("\n".join(cur))
    chunks = []
    for part in parts:
        heading = ""
        if part.startswith("#"):
            first, _, part = part.partition("\n")
            heading = first.lstrip("# ").strip()
        buf = ""
        for para in part.split("\n\n"):
            if len(buf) + len(para) > max_chars and buf:
                chunks.append((heading, buf.strip()))
                buf = ""
            buf += para + "\n\n"
        if buf.strip():
            chunks.append((heading, buf.strip()))
    out = []
    for heading, text in chunks:
        if len(text) < 80:
            continue
        name = f"{title} › {heading}" if heading and heading.lower() != title.lower() else (title or rel)
        # One copy of the text, inside "content" (title line + passage)
        out.append({"title": name, "content": f"{name}\n{text}", "category": section, "url": url})
    return out


def load_vault(n_pages, meter, progress=no_progress):
    c = tpuf()
    ns = c.namespace(NS["HashiCorp Vault docs"])
    progress(0.0, "Listing Vault docs on GitHub")
    files = list(vault_files())
    random.Random(42).shuffle(files)
    files = files[: int(n_pages)]
    t0 = time.time()
    pages, chunks = 0, []

    def fetch(p):
        r = requests.get(RAW + p, timeout=30)
        return p, (r.text if r.ok else "")

    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for i, (p, raw) in enumerate(ex.map(fetch, files), 1):
            if raw:
                pages += 1
                chunks += chunk_page(p, raw)
            if i % 20 == 0:
                progress(0.4 * i / len(files), f"Downloaded {i} of {len(files)} pages")
    t_dl = time.time() - t0
    if not chunks:
        raise AppError("No pages could be downloaded from GitHub. Try again in a minute.")
    for i, ch in enumerate(chunks, 1):
        ch["id"] = i
    batch = 400
    t1 = time.time()
    written = 0
    for s in range(0, len(chunks), batch):
        res = ns.write(upsert_rows=chunks[s:s + batch], distance_metric="cosine_distance", schema=SCHEMA)
        charge_write(meter, res)
        written += len(chunks[s:s + batch])
        progress(0.4 + 0.6 * written / len(chunks), f"Embedded and wrote {written:,} of {len(chunks):,} passages")
    t_w = time.time() - t1
    chars = sum(len(ch["content"]) for ch in chunks)
    return (f"**Loaded {pages} pages → {len(chunks):,} passages** ({chars / 1e6:.1f} MB of text) from HashiCorp's public "
            f"Vault {VAULT_VERSION} docs.\n\n- Download and chunking: {t_dl:.1f} s\n- Embedding and writing to turbopuffer: {t_w:.1f} s "
            f"in {-(-len(chunks) // batch)} batches\n\nNow try the **Search** tab with *HashiCorp Vault docs* selected.")


# ----------------------------------------------------------------------------
# Tab 5: Many tenants
# ----------------------------------------------------------------------------
def list_tenants():
    c = tpuf()
    return [n.id for n in c.namespaces(prefix=TENANT_PREFIX)]


def create_tenants(n, meter, progress=no_progress):
    c = tpuf()
    n = int(n)
    existing = set(list_tenants())
    rng = random.Random(7)
    todo = [i for i in range(n) if f"{TENANT_PREFIX}{i:04d}" not in existing]
    samples = {i: library_rows(rng.sample(LUMEN, 3)) for i in todo}
    lock = threading.Lock()
    t0 = time.time()

    def make(i):
        res = c.namespace(f"{TENANT_PREFIX}{i:04d}").write(upsert_rows=samples[i], distance_metric="cosine_distance", schema=SCHEMA)
        with lock:
            charge_write(meter, res)

    done = 0
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for f in cf.as_completed([ex.submit(make, i) for i in todo]):
            f.result()
            done += 1
            if done % 10 == 0 or done == len(todo):
                progress(done / max(len(todo), 1), f"Created {done} of {len(todo)} tenant namespaces")
    secs = time.time() - t0
    total = len(existing) + len(todo)
    try:
        size = c.namespace(f"{TENANT_PREFIX}0000").metadata().approx_logical_bytes or 0
    except Exception:
        size = 0
    monthly = total * size / 1e9 * STORAGE_USD_PER_GB_MONTH
    return (f"**{len(todo)} new tenant namespaces in {secs:.1f} s** ({len(existing)} already existed, {total} total). "
            f"Each holds that customer's own answers, fully isolated.\n\n"
            f"Keeping all {total} idle tenants for a month costs about **{fmt_usd(monthly)}** in storage "
            f"({fmt_bytes(size)} each). Idle namespaces only cost storage.")


def probe_tenant(meter):
    tenants = list_tenants()
    if not tenants:
        raise AppError("Create some tenants first.")
    name = random.choice(tenants)
    ns = tpuf().namespace(name)
    rows = []
    for i in range(3):
        t0 = time.time()
        res = ns.query(rank_by=by_keyword("data encryption backups"), top_k=3)
        ms = (time.time() - t0) * 1000
        charge_query(meter, res)
        p = getattr(res, "performance", None)
        rows.append({"Query": i + 1, "Cache": str(getattr(p, "cache_temperature", "") if p else ""),
                     "Server ms": getattr(p, "server_total_ms", None) if p else None, "Round trip ms": round(ms)})
    note = (f"Queried **{name}**, picked at random from {len(tenants)} tenants. A tenant nobody has touched for a while is "
            f"served **cold** from object storage the first time, then **warm** from cache. Tenants created moments ago are "
            f"still warm; come back tomorrow and run this again to see a true cold start. In a real app you'd call "
            f"`ns.hint_cache_warm()` when the user opens the search box.")
    return pd.DataFrame(rows), note


def delete_tenants():
    c = tpuf()
    tenants = list_tenants()
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(lambda n: c.namespace(n).delete_all(), tenants))
    return f"Deleted {len(tenants)} tenant namespaces. One call each, done in parallel."


def delete_everything():
    c = tpuf()
    names = list(NS.values()) + list_tenants()
    gone = 0
    for n in names:
        try:
            c.namespace(n).delete_all()
            gone += 1
        except Exception:
            pass
    return f"Deleted {gone} namespaces."


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
INTRO = """
#### What this is
A weekend project for exploring **turbopuffer**: a security-questionnaire assistant where each fictional company's library
of approved answers lives in its own **namespace** on object storage. Search finds answers by meaning, by exact keyword,
or both, and Claude can draft whole questionnaires from them.

**Things to try**
1. **Start here** → *Load sample libraries*. There's no cluster or index to create first; namespaces appear on first write.
2. **Search** → try the example questions. *Meaning* finds the flood/backup answer, *Keyword* wins on `CC6.1`, and *Compare all modes* shows all four side by side.
3. **Search** → open *Show the API call* to see the exact SDK call behind each result.
4. **Auto-fill** → *Draft answers* for a sample questionnaire; anything the library can't support gets flagged.
5. **Real docs** → load HashiCorp's public Vault docs and search thousands of passages with the same API.
6. **Many tenants** → create a few hundred namespaces, query a random one, and see what idle ones cost to store.

The bar at the top tracks what this session has cost in turbopuffer usage.
"""

ss = st.session_state  # short name: Streamlit's per-visitor memory between clicks


def md(text):
    """Escape dollar signs: Streamlit treats text between two $ as a math formula."""
    return text.replace("$", "\\$")


def password_gate():
    """If APP_PASSWORD is set, ask for it once per browser session."""
    pw = os.getenv("APP_PASSWORD")
    if not pw or ss.get("authed"):
        return
    st.title("turbopuffer playground")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == pw:
            ss["authed"] = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


def guarded(fn, *args, **kwargs):
    """Run a handler; show friendly errors on the page instead of a stack trace."""
    try:
        return fn(*args, **kwargs)
    except AppError as e:
        st.error(str(e))
    except Exception as e:
        st.error(friendly(e))
    return None


def progress_bar():
    bar = st.progress(0.0)
    return lambda frac, desc="": bar.progress(min(max(float(frac), 0.0), 1.0), text=desc)


def main():
    password_gate()
    ss.setdefault("meter", new_meter())
    meter = ss["meter"]

    st.title("turbopuffer playground · security questionnaire assistant")
    meter_box = st.empty()

    t1, t2, t3, t4, t5 = st.tabs(["1 · Start here", "2 · Search", "3 · Auto-fill", "4 · Real docs", "5 · Many tenants"])

    # ---------------- Start here
    with t1:
        st.markdown(INTRO)
        c1, c2, _ = st.columns([1, 1, 4])
        if c1.button("Load sample libraries", type="primary"):
            with st.spinner("Writing and embedding…"):
                msg = guarded(load_samples, meter)
            if msg:
                ss["load_msg"] = msg
                ss["status"] = guarded(status_table)
        if c2.button("Refresh status"):
            ss["status"] = guarded(status_table)
        if ss.get("load_msg"):
            st.markdown(md(ss["load_msg"]))
        if ss.get("status") is not None:
            st.dataframe(ss["status"], hide_index=True, width="stretch")
        with st.expander("Reset everything"):
            ok = st.checkbox("Yes, delete every namespace this app created")
            if st.button("Delete everything", disabled=not ok):
                msg = guarded(delete_everything)
                if msg:
                    st.success(msg)
                    ss["status"] = guarded(status_table)
                    ss.pop("load_msg", None)

    # ---------------- Search
    with t2:
        c1, c2 = st.columns([2, 3])
        lib = c1.selectbox("Library (namespace)", list(NS.keys()), key="lib")
        mode = c2.radio("Search mode", MODES, horizontal=True, key="mode")

        def use_example():
            if ss.get(f"ex_{lib}"):
                ss["q"] = ss[f"ex_{lib}"]

        st.text_input("Question", placeholder="Ask anything a customer might ask", key="q")
        st.radio("Example questions (click to use)", EXAMPLES.get(lib, []), index=None,
                 key=f"ex_{lib}", on_change=use_example, horizontal=True)
        c1, c2, c3 = st.columns(3)
        top_k = c1.slider("Results", 3, 20, 5)
        cat = c2.selectbox("Filter: category", categories_for(lib), key=f"cat_{lib}")
        year = 2024
        if lib in ANSWER_LIBRARIES:
            year = c3.slider("Filter: reviewed in or after (2024 = no filter)", 2024, 2026, 2024)
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Search", type="primary"):
            with st.spinner("Searching…"):
                out = guarded(do_search, lib, ss.get("q", ""), mode, top_k, cat, year, meter)
            if out:
                ss["search_out"] = out
        if b2.button("Compare all modes"):
            with st.spinner("Running all four modes…"):
                out = guarded(do_compare, lib, ss.get("q", ""), top_k, cat, year, meter)
            if out:
                ss["search_out"] = out
        out = ss.get("search_out")
        if out:
            st.markdown(md(out["stats"]))
            st.dataframe(out["table"], hide_index=True, width="stretch")
            if out["code"]:
                with st.expander("Show the API call"):
                    st.code(out["code"], language="python")
        st.caption("Vector search matches meaning, keyword search matches exact terms like control IDs, "
                   "and hybrid gets both in one request. Each query here costs about a ten-thousandth of a cent.")

    # ---------------- Auto-fill
    with t3:
        st.markdown("Paste a customer's questionnaire, one question per line, or upload a CSV with a `question` column. "
                    "For each question turbopuffer runs a hybrid search with reranking, then Claude drafts an answer using only "
                    "the approved answers it found, and flags anything it can't support.")
        c1, c2 = st.columns(2)
        af_lib = c1.selectbox("Answer library", ANSWER_LIBRARIES, key="af_lib")
        af_file = c2.file_uploader("Or upload CSV", type=["csv", "txt"])
        af_text = st.text_area(f"Questions (up to {MAX_AUTOFILL_QUESTIONS})", SAMPLE_QUESTIONNAIRE, height=240)
        if st.button("Draft answers", type="primary"):
            res = guarded(autofill, af_lib, af_text, af_file, meter, progress=progress_bar())
            if res:
                ss["af_out"] = res
        if ss.get("af_out"):
            df, summary = ss["af_out"]
            st.markdown(md(summary))
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button("Download drafts (CSV)", df.to_csv(index=False).encode(), "questionnaire_drafts.csv", "text/csv")

    # ---------------- Real docs
    with t4:
        st.markdown(f"Loads HashiCorp's **public Vault documentation** ({VAULT_VERSION}) straight from GitHub, splits each page "
                    "into passages by heading, and writes them to turbopuffer, which embeds them on the way in. "
                    "200 pages is about 1,500 passages and takes a minute or two.")
        n_pages = st.slider("Pages to load", 25, MAX_VAULT_PAGES, 200, step=25)
        if st.button("Load Vault docs", type="primary"):
            msg = guarded(load_vault, n_pages, meter, progress=progress_bar())
            if msg:
                ss["vault_msg"] = msg
        if ss.get("vault_msg"):
            st.markdown(md(ss["vault_msg"]))

    # ---------------- Many tenants
    with t5:
        st.markdown("A common pattern for multi-tenant apps is **one namespace per customer** (Cursor uses one per codebase). "
                    "Create a few hundred here, then query one at random.")
        n_ten = st.slider("Tenant namespaces to create", 10, MAX_TENANTS, 200, step=10)
        if st.button("Create tenants", type="primary"):
            msg = guarded(create_tenants, n_ten, meter, progress=progress_bar())
            if msg:
                ss["ten_msg"] = msg
        if ss.get("ten_msg"):
            st.markdown(md(ss["ten_msg"]))
        if st.button("Query a random tenant 3 times"):
            res = guarded(probe_tenant, meter)
            if res:
                ss["probe"] = res
        if ss.get("probe"):
            df, note = ss["probe"]
            st.dataframe(df, hide_index=True)
            st.markdown(md(note))
        with st.expander("Delete tenants"):
            ok_t = st.checkbox("Yes, delete all tenant namespaces")
            if st.button("Delete tenants", disabled=not ok_t):
                msg = guarded(delete_tenants)
                if msg:
                    st.success(msg)

    # Drawn last so it includes whatever the clicks above just cost
    meter_box.info(md(meter_md(meter)))


main()
