"""P6: Legal contract Q&A with hybrid search (BM25 + dense), RRF fusion, cross-encoder re-ranking,
context compression and citation-grounded answers.

Pipeline
  contract PDFs -> parse into clauses -> one chunk per clause (title + parties + clause number kept in the chunk)
  query -> BM25 (exact clause numbers / standards)  \
                                                     >-- Reciprocal Rank Fusion --> top-N candidates
        -> dense vectors (paraphrase / intent)      /
        -> cross-encoder re-ranker -> top-n -> extractive compression -> LLM answer with [S#] citations
        -> citation validator (every sentence cited, ids valid, figures present in the sources)

Four retrieval modes are compared on a synthetic benchmark with three query types:
  id   exact references ("Northwind Schedule 4 audit", "SOC 2 Type II", "Orion clause 8.1")
  sem  paraphrases that share few words with the clause text
  mix  natural questions combining both

The contracts are SYNTHETIC. Seven near-identical agreements share the same clause topics with
different parameters, so a retriever must pick the right contract AND the right clause.

Run
  python p6_hybrid_rag.py --offline          plumbing test (hashing embedder + lexical re-ranker; not semantic)
  python p6_hybrid_rag.py                    real models (all-MiniLM-L6-v2 + cross-encoder/ms-marco-MiniLM-L-6-v2)
  python p6_hybrid_rag.py --generate 3       also answer 3 questions with citations (mock LLM)
  python p6_hybrid_rag.py --generate 3 --llm anthropic     needs ANTHROPIC_API_KEY

Needs: pip install numpy sentence-transformers pypdf reportlab   (and anthropic for --llm anthropic)
Disk tip: --hf-cache D:\\hf_cache keeps model downloads off C:.
"""
import argparse
import csv
import hashlib
import math
import os
import re
import statistics
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ===================================================================== corpus
CONTRACTS = [
    dict(id="orion", short="Orion", title="ORION CLOUD SERVICES AGREEMENT", vend="Orion Systems Ltd",
         cust="Hartwell Logistics GmbH", law="England and Wales", court="London", start="1 March 2025", term=36,
         renew=90, pay=30, interest="4%", uptime="99.9%", credit="5%", breach=48, cert="ISO/IEC 27001:2022",
         pentest="once a year", ip="All improvements to the Services made by the Vendor remain the property of the Vendor.",
         cap="USD 2,500,000", cure=30, conv=180, conf=5, fm=60,
         special=("Source Code Escrow", ["The Vendor shall deposit the source code of the Services with Meridian Escrow Services within 30 days of the Effective Date.",
                                         "The escrow agent shall release the source code to the Customer if the Vendor becomes insolvent."])),
    dict(id="northwind", short="Northwind", title="NORTHWIND DATA PROCESSING AGREEMENT", vend="Northwind Data Ltd",
         cust="Bluepeak Retail Inc.", law="the State of New York", court="New York County", start="15 June 2025", term=24,
         renew=60, pay=45, interest="6%", uptime="99.5%", credit="10%", breach=72, cert="SOC 2 Type II",
         pentest="twice a year", ip="The Customer owns all outputs generated from its data.",
         cap="USD 1,000,000", cure=45, conv=90, conf=3, fm=90,
         special=("Audit Rights", ["The Customer may audit the Vendor's compliance with this Agreement once per year on 15 Business Days' notice.",
                                   "Audits shall follow the procedure set out in Schedule 4."])),
    dict(id="cobalt", short="Cobalt", title="COBALT MANAGED HOSTING AGREEMENT", vend="Cobalt Hosting BV",
         cust="Ardent Motors plc", law="the Netherlands", court="Amsterdam", start="1 January 2026", term=48,
         renew=120, pay=60, interest="8%", uptime="99.95%", credit="15%", breach=24,
         cert="ISO/IEC 27001:2013 and ISO/IEC 27017", pentest="every quarter",
         ip="The Customer grants the Vendor a licence to host the Customer's applications for the term only.",
         cap="EUR 5,000,000", cure=20, conv=270, conf=7, fm=30,
         special=("Anti-Bribery", ["Each party shall comply with the UK Bribery Act 2010 and the US Foreign Corrupt Practices Act (FCPA).",
                                   "Either party may suspend performance immediately if the other is credibly accused of bribery."])),
    dict(id="juniper", short="Juniper", title="JUNIPER ANALYTICS LICENCE AGREEMENT", vend="Juniper Analytics Inc.",
         cust="Saffron Health Ltd", law="the State of Delaware", court="Wilmington", start="1 September 2025", term=12,
         renew=30, pay=30, interest="5%", uptime="99.0%", credit="2%", breach=72, cert="HITRUST CSF r2",
         pentest="once a year", ip="The Vendor grants the Customer a non-exclusive, non-transferable licence to use the Services.",
         cap="USD 750,000", cure=30, conv=60, conf=5, fm=45,
         special=("HIPAA Business Associate", ["The parties shall enter into the Business Associate Agreement attached as Schedule 6.",
                                               "The Vendor shall not use protected health information except as that Schedule permits."])),
    dict(id="sterling", short="Sterling", title="STERLING PAYMENTS PLATFORM AGREEMENT", vend="Sterling Pay Ltd",
         cust="Maple Grocers Ltd", law="Scotland", court="Edinburgh", start="1 April 2026", term=60,
         renew=180, pay=15, interest="3%", uptime="99.99%", credit="20%", breach=24, cert="PCI DSS v4.0",
         pentest="every quarter", ip="The Vendor retains all rights in the payment platform and its documentation.",
         cap="GBP 10,000,000", cure=14, conv=365, conf=10, fm=21,
         special=("Step-In Rights", ["If the Vendor fails to restore the Services within 5 Business Days, the Customer may appoint a third party to step in and operate them.",
                                     "The Vendor shall reimburse the reasonable costs of the third party."])),
    dict(id="vantage", short="Vantage", title="VANTAGE SUPPORT AND MAINTENANCE AGREEMENT", vend="Vantage Support Pty Ltd",
         cust="Kestrel Mining Ltd", law="New South Wales", court="Sydney", start="1 July 2025", term=18,
         renew=45, pay=20, interest="7%", uptime="99.5%", credit="8%", breach=36, cert="ISO/IEC 27001:2022",
         pentest="once a year", ip="Bespoke developments are owned by the Customer once paid for in full.",
         cap="AUD 1,200,000", cure=21, conv=120, conf=4, fm=50,
         special=("Most Favoured Customer", ["The Vendor shall offer the Customer prices no less favourable than those offered to any comparable customer.",
                                              "The Customer may request a certificate of compliance from the Vendor's finance director each year."])),
    dict(id="sapphire", short="Sapphire", title="SAPPHIRE TELECOM CONNECTIVITY AGREEMENT", vend="Sapphire Telecom SA",
         cust="Granite Foods Ltd", law="France", court="Paris", start="1 October 2025", term=30,
         renew=75, pay=40, interest="5%", uptime="99.8%", credit="12%", breach=48, cert="ISO/IEC 27001:2022 and ISO/IEC 22301",
         pentest="twice a year", ip="Network configurations created for the Customer are the Customer's property.",
         cap="EUR 3,000,000", cure=25, conv=150, conf=6, fm=40,
         special=("Export Controls", ["Each party shall comply with the US Export Administration Regulations (EAR) where applicable.",
                                      "Neither party shall transfer controlled technology to a restricted jurisdiction."])),
]

CLAUSES = [
    (1, "Definitions and Interpretation", [
        '"Services" means the services described in Schedule 1 to this Agreement.',
        '"Confidential Information" means all non-public information disclosed by one party to the other.',
        "References to clauses and Schedules are to the clauses of and Schedules to this Agreement."]),
    (2, "Term and Renewal", [
        "This Agreement commences on {start} and continues for an initial term of {term} months.",
        "It renews automatically for successive periods of twelve months unless either party gives {renew} days' written notice of non-renewal."]),
    (3, "Fees and Payment", [
        "The Customer shall pay each undisputed invoice within {pay} days of receipt.",
        "Overdue amounts accrue interest at {interest} per annum from the due date until payment."]),
    (4, "Service Levels", [
        "The Vendor shall ensure that the Services are available at least {uptime} of the time in each calendar month.",
        "If availability falls below this level the Customer is entitled to a service credit of {credit} of the monthly fee for each full percentage point of shortfall."]),
    (5, "Data Protection", [
        "The Vendor shall process Personal Data only on the Customer's documented instructions and in accordance with Article 28 of the GDPR.",
        "The Vendor shall notify the Customer of any personal data breach within {breach} hours of becoming aware of it."]),
    (6, "Information Security", [
        "The Vendor shall maintain certification to {cert} throughout the term.",
        "The Vendor shall commission an independent penetration test of the Services {pentest} and share a summary of the findings."]),
    (7, "Intellectual Property", [
        "Each party retains ownership of the intellectual property it owned before the Effective Date.",
        "{ip}"]),
    (8, "Limitation of Liability", [
        "Subject to clause 8.3, the total aggregate liability of either party under this Agreement shall not exceed {cap}.",
        "Neither party is liable for indirect or consequential loss.",
        "Nothing in this Agreement limits liability for death or personal injury caused by negligence, or for fraud."]),
    (9, "Termination", [
        "Either party may terminate this Agreement by written notice if the other commits a material breach that is not remedied within {cure} days of notice of the breach.",
        "The Customer may terminate this Agreement for convenience on {conv} days' written notice to the Vendor.",
        "On termination the Vendor shall return or delete the Customer's data within 30 days."]),
    (10, "Confidentiality", [
        "Each party shall keep the other's Confidential Information secret and use it only to perform this Agreement.",
        "These obligations continue for {conf} years after termination or expiry."]),
    (11, "Force Majeure", [
        "Neither party is liable for delay caused by events beyond its reasonable control, including flood, fire, war and failure of public utilities.",
        "If the event continues for more than {fm} days either party may terminate this Agreement by written notice."]),
    (12, "Governing Law and Jurisdiction", [
        "This Agreement is governed by the laws of {law}.",
        "The courts of {court} have exclusive jurisdiction over any dispute arising from this Agreement."]),
]


def contract_clauses(c):
    out = [(n, h, [t.format(**c) for t in subs]) for n, h, subs in CLAUSES]
    out.append((13, c["special"][0], list(c["special"][1])))
    return out


def write_contract_pdfs(folder: Path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    folder.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for c in CONTRACTS:
        path = folder / f"{c['id']}.pdf"
        if path.exists():
            continue
        story = [Paragraph(esc(c["title"]), styles["Title"]),
                 Paragraph(esc(f'This Agreement is made between {c["vend"]} (the "Vendor") and {c["cust"]} (the "Customer").'),
                           styles["BodyText"]), Spacer(1, 8)]
        for n, heading, subs in contract_clauses(c):
            story.append(Paragraph(esc(f"{n}. {heading}"), styles["Heading2"]))
            for k, text in enumerate(subs, 1):
                story.append(Paragraph(esc(f"{n}.{k} {text}"), styles["BodyText"]))
        SimpleDocTemplate(str(path), pagesize=A4).build(story)


# ===================================================================== parsing
@dataclass
class Chunk:
    id: str
    cid: str
    clause: int
    heading: str
    text: str           # header line(s) + body, what gets indexed
    units: list         # sub-clauses, used for compression


HEAD = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Za-z ,&/()'\-]{2,80})$")
SUB = re.compile(r"^\d{1,2}\.\d{1,2}\s")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u2019", "'").replace("\u2018", "'")
    return s


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", norm_text(s).lower()).strip()


def parse_contract_pdf(path: Path) -> list:
    from pypdf import PdfReader
    pages = []
    for page in PdfReader(str(path)).pages:
        try:
            pages.append(page.extract_text(extraction_mode="layout") or "")
        except TypeError:
            pages.append(page.extract_text() or "")
    lines = [re.sub(r"[ \t]+", " ", norm_text(l)).strip() for l in "\n".join(pages).splitlines()]

    i = 0
    while i < len(lines) and not lines[i]:
        i += 1
    title_parts = []
    while i < len(lines) and lines[i] and lines[i].isupper():       # titles can wrap over lines
        title_parts.append(lines[i])
        i += 1
    title = " ".join(title_parts).title()

    pre, clauses, cur = [], [], None
    for line in lines[i:]:
        m = HEAD.match(line)
        if m:
            cur = {"num": int(m.group(1)), "heading": m.group(2).strip(), "lines": []}
            clauses.append(cur)
        elif line:
            (cur["lines"] if cur else pre).append(line)

    pm = re.search(r'between (.+?) \(the "Vendor"\) and (.+?) \(the "Customer"\)', " ".join(pre))
    header = f"{title} ({pm.group(1)} and {pm.group(2)})" if pm else title

    chunks = []
    for c in clauses:
        units = []
        for line in c["lines"]:
            if SUB.match(line) or not units:
                units.append(line)
            else:
                units[-1] += " " + line                         # unwrap lines wrapped by the PDF layout
        text = f"{header}\nClause {c['num']}. {c['heading']}\n" + "\n".join(units)
        chunks.append(Chunk(f"{path.stem}::{c['num']}", path.stem, c["num"], c["heading"], text, units))
    return chunks


# ==================================================================== queries
def build_queries():
    by = {c["id"]: c for c in CONTRACTS}
    Q = []  # (type, query, contract id, clause number, fact string that must be in the clause)
    for cid in ("orion", "northwind", "cobalt"):
        c = by[cid]
        Q.append(("id", f"{c['short']} clause 8.1", cid, 8, c["cap"]))
        Q.append(("id", f"{c['short']} clause 9.2", cid, 9, f"{c['conv']} days' written notice"))
    Q += [("id", "SOC 2 Type II certification", "northwind", 6, "SOC 2 Type II"),
          ("id", "PCI DSS v4.0", "sterling", 6, "PCI DSS v4.0"),
          ("id", "HITRUST CSF r2", "juniper", 6, "HITRUST CSF r2"),
          ("id", "Schedule 4 audit", "northwind", 13, "Schedule 4"),
          ("id", "Schedule 6 business associate", "juniper", 13, "Schedule 6"),
          ("id", "UK Bribery Act 2010 FCPA", "cobalt", 13, "Bribery Act 2010"),
          ("id", "EAR export administration regulations", "sapphire", 13, "Export Administration Regulations")]
    sem = [
        ("What is the most {cust} could recover from {vend} if things go badly wrong?", 8, lambda c: c["cap"], ("orion", "juniper", "sterling")),
        ("How much advance warning must {cust} give to walk away from the deal early when nobody is at fault?", 9, lambda c: f"{c['conv']} days' written notice", ("cobalt", "vantage", "sapphire")),
        ("How quickly must {vend} tell {cust} if customer personal information is leaked?", 5, lambda c: f"within {c['breach']} hours", ("northwind", "sterling", "vantage")),
        ("How long does {cust} have to settle what it owes {vend}?", 3, lambda c: f"within {c['pay']} days of receipt", ("orion", "cobalt", "sapphire")),
        ("If a flood stops {vend} from performing, how long until either side can end the contract?", 11, lambda c: f"more than {c['fm']} days", ("northwind", "juniper", "vantage")),
    ]
    for tmpl, clause, fact, cids in sem:
        for cid in cids:
            c = by[cid]
            Q.append(("sem", tmpl.format(**c), cid, clause, fact(c)))
    mix = [
        ("Under the {short} agreement, what service credit applies if availability drops below {uptime}?", 4, lambda c: f"service credit of {c['credit']}", ("orion", "sterling")),
        ("{short}: how long does confidentiality last after the contract ends?", 10, lambda c: f"{c['conf']} years after termination", ("cobalt", "sapphire")),
        ("Which governing law and courts apply to the {short} agreement?", 12, lambda c: c["law"], ("juniper", "vantage")),
        ("How often must the vendor under the {short} agreement run penetration tests?", 6, lambda c: c["pentest"], ("northwind", "sterling")),
    ]
    for tmpl, clause, fact, cids in mix:
        for cid in cids:
            c = by[cid]
            Q.append(("mix", tmpl.format(**c), cid, clause, fact(c)))
    return Q


# =================================================================== retrieval
TOKEN = re.compile(r"[a-z0-9]+(?:[.:/\-][a-z0-9]+)*")


def tokenize(text: str) -> list:
    """Keeps identifiers whole ("8.1", "27001:2022", "iso/iec") and also adds their parts."""
    toks = []
    for t in TOKEN.findall(norm_text(text).lower()):
        toks.append(t)
        if re.search(r"[:/\-]", t):
            toks += [p for p in re.split(r"[:/\-]", t) if p]
    return toks


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def scores(self, query_tokens):
        s = np.zeros(len(self.tf))
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class HashingEmbedder:
    """Offline stand-in. Matches shared words only (no synonyms): for plumbing tests."""
    name = "hashing-offline"

    def __init__(self, dims=768):
        self.dims = dims

    def embed(self, texts, is_query=False):
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[r, h % self.dims] += 1.0 if (h >> 20) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class STEmbedder:
    def __init__(self, model_name, cache_folder=None):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)

    def embed(self, texts, is_query=False):
        if is_query and "bge" in self.name.lower():
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                            show_progress_bar=False), dtype=np.float32)


class LexicalReranker:
    """Offline stand-in for a cross-encoder: IDF-free token-overlap score."""
    name = "lexical-offline"

    def score(self, query, texts):
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / (len(q) or 1) for t in texts]


class CrossEncoderReranker:
    def __init__(self, model_name, cache_folder=None):
        from sentence_transformers import CrossEncoder
        self.name = model_name
        self.model = CrossEncoder(model_name, cache_folder=cache_folder)

    def score(self, query, texts):
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16,
                                                      show_progress_bar=False)]


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: score(d) = sum over rankers of 1 / (k + rank)."""
    score = Counter()
    for ranking in rankings:
        for r, d in enumerate(ranking, 1):
            score[d] += 1.0 / (k + r)
    return [d for d, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))]


class Retriever:
    def __init__(self, chunks, embedder, reranker, candidates=20, rrf_k=60):
        self.chunks, self.embedder, self.reranker = chunks, embedder, reranker
        self.candidates, self.rrf_k = candidates, rrf_k
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.matrix = embedder.embed([c.text for c in chunks])

    def bm25_rank(self, q, n):
        s = self.bm25.scores(tokenize(q))
        return [int(i) for i in np.argsort(-s, kind="stable")[:n] if s[i] > 0]

    def dense_rank(self, q, n):
        s = self.matrix @ self.embedder.embed([q], is_query=True)[0]
        return [int(i) for i in np.argsort(-s, kind="stable")[:n]]

    def hybrid_rank(self, q, n):
        N = self.candidates
        return rrf([self.bm25_rank(q, N), self.dense_rank(q, N)], self.rrf_k)[:n]

    def rerank(self, q, n):
        cands = self.hybrid_rank(q, self.candidates)
        scores = self.reranker.score(q, [self.chunks[i].text for i in cands])
        order = sorted(range(len(cands)), key=lambda j: -scores[j])
        return [cands[j] for j in order][:n], [scores[j] for j in order][:n]

    def search(self, mode, q, n=10):
        if mode == "bm25":
            return self.bm25_rank(q, n)
        if mode == "dense":
            return self.dense_rank(q, n)
        if mode == "hybrid_rrf":
            return self.hybrid_rank(q, n)
        if mode == "hybrid_rerank":
            return self.rerank(q, n)[0]
        raise ValueError(mode)


MODES = ["bm25", "dense", "hybrid_rrf", "hybrid_rerank"]


# ================================================================ compression
def compress(query, chunk, reranker, keep=2):
    """Extractive compression: keep the 2-line header plus the `keep` sub-clauses that best match."""
    if len(chunk.units) <= keep:
        return chunk.text
    header = "\n".join(chunk.text.split("\n")[:2])
    scores = reranker.score(query, chunk.units)
    best = sorted(sorted(range(len(chunk.units)), key=lambda i: -scores[i])[:keep])
    return header + "\n" + "\n".join(chunk.units[i] for i in best)


# =================================================================== evaluate
def evaluate(retriever, queries, chunk_index):
    results, per_query = [], []
    for mode in MODES:
        ranks, times, by_type = [], [], {}
        for typ, q, cid, clause, _ in queries:
            gold = chunk_index[f"{cid}::{clause}"]
            t0 = time.perf_counter()
            top = retriever.search(mode, q, 10)
            times.append((time.perf_counter() - t0) * 1000)
            rank = top.index(gold) + 1 if gold in top else None
            ranks.append(rank)
            by_type.setdefault(typ, []).append(rank)
        n = len(queries)
        rec = lambda k, rs=ranks: sum(1 for r in rs if r and r <= k) / len(rs)
        mrr = lambda rs: sum(1 / r for r in rs if r) / len(rs)
        ndcg5 = sum(1 / math.log2(r + 1) for r in ranks if r and r <= 5) / n
        results.append({"mode": mode, "recall@1": round(rec(1), 3), "recall@3": round(rec(3), 3),
                        "recall@5": round(rec(5), 3), "mrr@10": round(mrr(ranks), 3), "ndcg@5": round(ndcg5, 3),
                        **{f"mrr_{t}": round(mrr(r), 3) for t, r in sorted(by_type.items())},
                        "ms_per_query": round(statistics.mean(times), 1)})
        per_query.append(ranks)
    return results, per_query


def evaluate_compression(retriever, queries, chunk_index, final_n=3):
    """For each setting of `keep` (sub-clauses kept per chunk): words sent to the LLM, and whether
    the fact that answers the question survives when the gold clause was retrieved."""
    tops = [retriever.rerank(q, final_n)[0] for _, q, _, _, _ in queries]
    rows = []
    for keep in (None, 2, 1):                       # None = no compression
        words, kept, answerable = [], 0, 0
        for (typ, q, cid, clause, fact), top in zip(queries, tops):
            texts = [retriever.chunks[i].text if keep is None else compress(q, retriever.chunks[i], retriever.reranker, keep)
                     for i in top]
            words.append(sum(len(t.split()) for t in texts))
            if chunk_index[f"{cid}::{clause}"] in top:
                answerable += 1
                kept += norm(fact) in norm(" ".join(texts))
        rows.append({"sub_clauses_kept": "all" if keep is None else keep,
                     "avg_context_words": round(statistics.mean(words), 1),
                     "fact_retained": f"{kept}/{answerable}"})
    base = rows[0]["avg_context_words"]
    for r in rows:
        r["words_saved_pct"] = round(100 * (1 - r["avg_context_words"] / base), 1)
    return rows


# =================================================================== generation
SYSTEM_PROMPT = """You are a contracts analyst. Answer ONLY from the numbered sources provided.
Rules:
- After every sentence of your answer, cite the supporting source(s) like [S1] or [S1][S2].
- Quote figures, periods and party names exactly as written in the sources.
- Name the agreement each fact comes from when sources come from different agreements.
- If the sources do not contain the answer, reply exactly: INSUFFICIENT CONTEXT
- Never use outside knowledge."""


def build_user_prompt(question, contexts):
    src = "\n\n".join(f"[S{i}]\n{t}" for i, t in enumerate(contexts, 1))
    return f"Sources:\n{src}\n\nQuestion: {question}"


class MockLLM:
    """Plumbing stand-in: returns the best-overlapping sentence with a citation. Not a real model."""
    name = "mock-llm"

    def complete(self, system, user):
        question = user.rsplit("Question:", 1)[1]
        q = set(tokenize(question))
        best, best_s = None, -1
        for m in re.finditer(r"\[S(\d+)\]\n(.*?)(?=\n\n\[S|\n\nQuestion:)", user, re.S):
            for line in m.group(2).split("\n")[2:]:
                s = len(q & set(tokenize(line)))
                if s > best_s:
                    best, best_s = (m.group(1), line), s
        return f"{best[1]} [S{best[0]}]" if best else "INSUFFICIENT CONTEXT"


class AnthropicLLM:
    def __init__(self, model=None):
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set in this terminal.\n"
                             '  PowerShell:  $env:ANTHROPIC_API_KEY="your-key"  (then run again in the SAME window)\n'
                             "  Or run with --llm mock.")
        self.client = anthropic.Anthropic()          # SDK retries 429/5xx twice by default
        self.name = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")

    def complete(self, system, user):
        r = self.client.messages.create(model=self.name, max_tokens=500, system=system,
                                        messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if b.type == "text")


def validate_answer(answer, contexts):
    """Checks citation discipline: refusal is fine; otherwise every sentence must cite a valid
    source and every number in the answer must appear in the cited sources."""
    issues, a = [], answer.strip()
    if a == "INSUFFICIENT CONTEXT":
        return {"valid": True, "issues": [], "cited": []}
    # Models cite either before or after the full stop ("... [S1]." / "... . [S1]"): normalise to
    # "citation then punctuation" so a trailing citation stays attached to its own sentence.
    a = re.sub(r"([.!?])\s*((?:\[S\d+\])+)", r"\2\1", a)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", a) if s.strip()]
    cited_all = set()
    for s in sentences:
        ids = [int(x) for x in re.findall(r"\[S(\d+)\]", s)]
        if not ids:
            issues.append(f"uncited sentence: {s[:60]}")
        bad = [i for i in ids if not 1 <= i <= len(contexts)]
        if bad:
            issues.append(f"invalid source id(s) {bad}")
        cited_all.update(i for i in ids if 1 <= i <= len(contexts))
    body = " ".join(contexts[i - 1] for i in cited_all)
    stripped = re.sub(r"\[S\d+\]", "", a)
    for num in set(re.findall(r"\d[\d,.]*\d|\d", stripped)):
        if num.rstrip(".,") not in body:
            issues.append(f"figure '{num}' not found in cited sources")
    return {"valid": not issues, "issues": issues, "cited": sorted(cited_all)}


# ======================================================================= main
def print_table(rows):
    cols = list(rows[0])
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="p6_contracts")
    ap.add_argument("--offline", action="store_true", help="hashing embedder + lexical re-ranker (no downloads)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--candidates", type=int, default=20)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--generate", type=int, default=0, help="answer N demo questions with citations")
    ap.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    args = ap.parse_args()

    docs = Path(args.docs)
    write_contract_pdfs(docs)
    chunks = [c for p in sorted(docs.glob("*.pdf")) for c in parse_contract_pdf(p)]
    index = {c.id: i for i, c in enumerate(chunks)}
    queries = build_queries()
    for typ, q, cid, clause, fact in queries:          # sanity: every gold chunk exists and holds the fact
        gid = f"{cid}::{clause}"
        assert gid in index, f"gold chunk {gid} missing: PDF parsing problem"
        assert norm(fact) in norm(chunks[index[gid]].text), f"fact '{fact}' not in {gid}"
    print(f"Parsed {len(set(c.cid for c in chunks))} contracts into {len(chunks)} clause chunks; "
          f"{len(queries)} queries ({', '.join(f'{t}={sum(1 for x in queries if x[0]==t)}' for t in ('id','sem','mix'))})")

    if args.offline:
        embedder, reranker = HashingEmbedder(), LexicalReranker()
    else:
        embedder = STEmbedder(args.model, args.hf_cache)
        reranker = CrossEncoderReranker(args.reranker, args.hf_cache)
    print(f"Embedder: {embedder.name} | Re-ranker: {reranker.name}\n")

    retriever = Retriever(chunks, embedder, reranker, args.candidates, args.rrf_k)
    results, per_query = evaluate(retriever, queries, index)
    print("=== Retrieval benchmark (gold = the one correct clause; higher is better) ===")
    print_table(results)
    comp = evaluate_compression(retriever, queries, index)
    print("\n=== Context compression (top-3 after re-ranking) ===")
    print_table(comp)

    with open("p6_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    with open("p6_per_query.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "query", "gold"] + [f"rank_{m}" for m in MODES])
        for qi, (typ, q, cid, clause, _) in enumerate(queries):
            w.writerow([typ, q, f"{cid}::{clause}"] + [per_query[mi][qi] or "miss" for mi in range(len(MODES))])

    best = max(results, key=lambda r: (r["mrr@10"], r["recall@1"]))
    lines = ["# P6 Hybrid RAG Benchmark", "",
             f"- Embedder `{embedder.name}`, re-ranker `{reranker.name}`, candidates {args.candidates}, RRF k {args.rrf_k}",
             f"- {len(chunks)} clause chunks from {len(set(c.cid for c in chunks))} synthetic contracts; {len(queries)} queries", "",
             "| " + " | ".join(results[0]) + " |", "|" + "---|" * len(results[0]),
             *("| " + " | ".join(str(r[c]) for c in results[0]) + " |" for r in results), "",
             f"Best MRR@10 on this run: **{best['mode']}**. With {len(queries)} queries, gaps of a few points are within noise.", "",
             "## Context compression (top-3 after re-ranking)",
             "| " + " | ".join(comp[0]) + " |", "|" + "---|" * len(comp[0]),
             *("| " + " | ".join(str(r[c]) for c in comp[0]) + " |" for r in comp), ""]
    if args.offline:
        lines += ["> Offline stand-ins were used for dense retrieval and re-ranking: only the BM25 rows are "
                  "meaningful. Re-run without --offline for real results.", ""]
    Path("p6_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved p6_results.csv, p6_per_query.csv, p6_report.md")

    if args.generate:
        llm = AnthropicLLM() if args.llm == "anthropic" else MockLLM()
        print(f"\n=== Cited answers ({llm.name}) ===")
        for typ, q, cid, clause, fact in queries[::max(1, len(queries) // args.generate)][:args.generate]:
            top, _ = retriever.rerank(q, 3)
            ctx = [compress(q, chunks[i], reranker) for i in top]
            ans = llm.complete(SYSTEM_PROMPT, build_user_prompt(q, ctx))
            v = validate_answer(ans, ctx)
            print(f"\nQ: {q}\nA: {ans}\n   citations valid: {v['valid']} {v['issues'] or ''}  "
                  f"sources: {[chunks[top[i-1]].id for i in v['cited']]}  (gold: {cid}::{clause})")
    if args.offline:
        print("\nNOTE: --offline is for plumbing only. Dense and re-rank rows are not semantic.")


if __name__ == "__main__":
    main()