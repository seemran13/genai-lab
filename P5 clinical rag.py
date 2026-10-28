"""P5: Clinical document ingestion -> chunking comparison -> embeddings -> ChromaDB retrieval.

Pipeline
  PDF / Markdown / text  ->  parse (layout-aware, heading detection)
                         ->  chunk: (A) fixed-size with overlap   vs   (B) header-based
                         ->  dense embeddings (sentence-transformers)
                         ->  ChromaDB collection per strategy (cosine similarity)
                         ->  top-k retrieval over patient scenarios + chunking comparison report

The sample clinical documents are SYNTHETIC and for engineering practice only. They are not
clinical guidance. Drop your own .pdf/.md/.txt files in the docs folder to use real material
(and then write your own evaluation queries in QUERIES below).

Run
  python p5_clinical_rag.py                       default model: all-MiniLM-L6-v2 (downloads ~90 MB)
  python p5_clinical_rag.py --model BAAI/bge-small-en-v1.5
  python p5_clinical_rag.py --embedder hashing    offline plumbing test; NOT semantic (no synonyms)

Needs: pip install chromadb sentence-transformers pypdf reportlab
       (reportlab is only used to create the sample PDF)
Disk tip on Windows: models cache to C:\\Users\\you\\.cache\\huggingface by default. Use
       --hf-cache D:\\hf_cache to keep them off a nearly full C: drive.
"""
import argparse
import csv
import hashlib
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================== sample documents
SAMPLE_DOCS = {
"sepsis_screening_guideline.md": """# Sepsis Screening and Early Management

> Synthetic document for engineering practice. Not clinical guidance.

## 1. Purpose and Scope
This guideline describes how nursing and medical staff screen adult inpatients and emergency department patients for sepsis, a life-threatening organ dysfunction caused by a dysregulated response to infection. It does not apply to children or to pregnant patients, who follow separate pathways.

## 2. Screening Criteria
Screen every adult patient with suspected infection at triage and again every four hours on the ward. A patient screens positive when a quick SOFA (qSOFA) score is two or more. One point is given for each of: respiratory rate of 22 breaths per minute or more, altered mentation, and systolic blood pressure of 100 mmHg or less. Any positive screen triggers an immediate senior clinician review.

## 3. Initial Investigations
Draw blood cultures before starting antibiotics, provided this does not delay treatment by more than 45 minutes. Measure serum lactate on every positive screen. A lactate above 2 mmol/L indicates tissue hypoperfusion and requires repeat measurement within two hours. Send urine, wound and respiratory samples as clinically indicated.

## 4. Fluid Resuscitation
For hypotension or a lactate of 4 mmol/L or higher, give an initial crystalloid bolus of 30 mL/kg within the first three hours. Reassess volume status after each bolus using capillary refill, urine output and blood pressure trends. Start vasopressors if hypotension persists despite adequate fluids.

## 5. Antimicrobial Therapy
Give broad-spectrum intravenous antibiotics within one hour of recognising septic shock, and within three hours for sepsis without shock. Review the choice of agent after 48 hours once culture results return, and narrow the spectrum where possible.

## 6. Escalation
Contact the intensive care outreach team if the patient needs vasopressors, has a lactate that remains above 4 mmol/L after resuscitation, or shows a falling level of consciousness.
""",

"hypertension_management.txt": """HYPERTENSION MANAGEMENT IN PRIMARY CARE

Synthetic document for engineering practice. Not clinical guidance.

1. Diagnosis
Hypertension is diagnosed when clinic blood pressure is 140/90 mmHg or higher on repeated measurement, confirmed by ambulatory or home monitoring. A home average of 135/85 mmHg or higher supports the diagnosis. Measure in both arms at the first visit and use the arm with the higher reading thereafter.

2. Lifestyle Advice
Offer all patients advice on reducing dietary salt to under 5 g per day, regular aerobic exercise, weight reduction where the body mass index is above 25, and limiting alcohol. Smoking cessation support should be offered at every visit.

3. First-Line Drug Treatment
For patients under 55 who are not of Black African or African-Caribbean origin, start an ACE inhibitor or an angiotensin receptor blocker. For patients aged 55 or over, or of Black African or African-Caribbean origin, start a calcium channel blocker. Check kidney function and potassium before and one to two weeks after starting an ACE inhibitor.

4. Resistant Hypertension
Blood pressure that stays above target despite three drugs including a diuretic at optimal doses is termed resistant hypertension. Before labelling a patient as resistant, confirm adherence, exclude white coat effect with ambulatory monitoring, and screen for secondary causes such as primary aldosteronism and renal artery stenosis.

5. Hypertensive Emergency
A blood pressure above 180/120 mmHg with signs of acute organ damage, for example chest pain, confusion, papilloedema or acute kidney injury, is a hypertensive emergency. Admit the patient for monitored intravenous treatment and lower the pressure gradually, aiming for a reduction of no more than 25% in the first hour.
""",

"anticoagulant_interactions.pdf": """ANTICOAGULANT DRUG INTERACTIONS QUICK REFERENCE

Synthetic document for engineering practice. Not clinical guidance.

1. Warfarin
Warfarin is a vitamin K antagonist with a narrow therapeutic range, monitored through the international normalised ratio (INR). Many drugs alter its effect. Amiodarone, fluconazole and metronidazole raise the INR and increase bleeding risk. Rifampicin and carbamazepine induce liver enzymes and lower the INR, so the warfarin dose often has to be increased.

2. Direct Oral Anticoagulants
Apixaban and rivaroxaban are direct oral anticoagulants and do not require routine INR monitoring. Strong dual inhibitors of CYP3A4 and P-glycoprotein, such as ketoconazole and ritonavir, increase their blood levels and should be avoided. Rifampicin and phenytoin reduce their levels and the combination should generally be avoided.

3. Antiplatelets and NSAIDs
Combining an anticoagulant with aspirin, clopidogrel or a non-steroidal anti-inflammatory drug (NSAID) such as ibuprofen raises the risk of gastrointestinal bleeding. Where an analgesic is needed, paracetamol is preferred. If dual therapy is unavoidable, add a proton pump inhibitor for gastric protection.

4. Reversal of Anticoagulation
For life-threatening bleeding on warfarin, give intravenous vitamin K together with prothrombin complex concentrate. For bleeding on apixaban or rivaroxaban, specific reversal with andexanet alfa may be considered where available; otherwise prothrombin complex concentrate is used.

5. Perioperative Management
Warfarin is usually stopped five days before elective surgery, with the INR checked the day before. Direct oral anticoagulants are stopped for 24 to 72 hours depending on renal function and the bleeding risk of the procedure.
""",

"asthma_exacerbation_protocol.md": """# Acute Asthma Exacerbation Protocol

> Synthetic document for engineering practice. Not clinical guidance.

## 1. Severity Assessment
An acute severe attack is present when peak expiratory flow (PEF) is 33 to 50 percent of best or predicted, the respiratory rate is 25 or more, the heart rate is 110 or more, or the patient cannot complete sentences in one breath. A life-threatening attack is indicated by PEF below 33 percent, oxygen saturation below 92 percent, a silent chest, cyanosis, exhaustion or confusion.

## 2. Immediate Treatment
Give oxygen to maintain saturation between 94 and 98 percent. Give a short-acting beta-agonist such as salbutamol by oxygen-driven nebuliser. Give systemic corticosteroids as soon as possible, using oral prednisolone unless the patient cannot swallow, and continue for five days.

## 3. Escalation of Treatment
Add nebulised ipratropium bromide for severe or life-threatening attacks. Consider a single dose of intravenous magnesium sulfate on senior advice if the response is poor. Request intensive care review for any patient with a life-threatening feature that does not improve.

## 4. Discharge Criteria
Discharge when the patient is stable on four-hourly inhaled therapy, PEF is above 75 percent of best or predicted, and inhaler technique has been checked. Provide a written asthma action plan, a course of oral steroids and a follow-up appointment within two working days.
""",

"chest_pain_triage.txt": """CHEST PAIN TRIAGE AND SUSPECTED ACUTE CORONARY SYNDROME

Synthetic document for engineering practice. Not clinical guidance.

1. Initial Assessment
Any patient presenting with chest pain suggestive of a heart attack must have a 12-lead electrocardiogram (ECG) within ten minutes of arrival. Record vital signs, pain characteristics and risk factors such as diabetes, smoking and family history.

2. ST-Elevation Myocardial Infarction
ST-elevation myocardial infarction (STEMI) is identified by new ST elevation of at least 1 mm in two contiguous leads. Activate the cardiac catheterisation laboratory immediately. The target is primary percutaneous coronary intervention within 90 minutes of first medical contact. If this cannot be achieved within 120 minutes, give fibrinolytic therapy.

3. Non-ST-Elevation Presentations
In the absence of ST elevation, measure high-sensitivity troponin at presentation and again at three hours. A rising troponin with ischaemic symptoms indicates non-ST-elevation myocardial infarction (NSTEMI). A normal troponin on both samples with a low risk score allows safe early discharge.

4. Initial Medical Therapy
Give a loading dose of aspirin unless contraindicated. Provide oxygen only if saturation falls below 90 percent. Use sublingual glyceryl trinitrate for ongoing pain provided systolic blood pressure is above 90 mmHg.

5. Red Flags for Alternative Diagnoses
Tearing pain radiating to the back with unequal arm pressures suggests aortic dissection, and anticoagulation must be withheld until it is excluded. Pleuritic pain with sudden breathlessness suggests pulmonary embolism.
""",
}

# Patient scenarios (written with clinical synonyms on purpose, e.g. "blood thinner",
# "heart tracing", "blood poisoning"). `snippet` is the key fact that must appear in a
# retrieved chunk for the retrieval to count as a hit, so scoring works for ANY chunking.
QUERIES = [
    ("Patient on warfarin has just started amiodarone for rhythm control; what happens to clotting?", "raise the inr and increase bleeding risk"),
    ("Patient taking apixaban asks if they can use ibuprofen for back pain.", "paracetamol is preferred"),
    ("Man on a blood thinner has a brain bleed after a fall; how do we reverse it?", "intravenous vitamin k together with prothrombin complex concentrate"),
    ("Planning knee replacement for a patient on warfarin: when should it be stopped?", "stopped five days before elective surgery"),
    ("Crushing chest pain arriving by ambulance: how soon must we record the heart tracing?", "within ten minutes of arrival"),
    ("ST elevation on the ECG: how quickly should the stent or clot-buster happen?", "within 90 minutes of first medical contact"),
    ("Troponin was normal at arrival and at three hours with low risk; can we send them home?", "allows safe early discharge"),
    ("Sudden tearing back pain with different readings in each arm; start blood thinners?", "unequal arm pressures suggests aortic dissection"),
    ("Fever, fast breathing and confusion in A&E: which score flags possible blood poisoning?", "quick sofa (qsofa) score is two or more"),
    ("How much IV fluid for a septic patient with low blood pressure?", "30 ml/kg within the first three hours"),
    ("How fast must broad antibiotics start in septic shock?", "within one hour of recognising septic shock"),
    ("Clinic reading is 150/95 twice; how do we confirm high blood pressure at home?", "135/85 mmhg or higher supports the diagnosis"),
    ("45-year-old white patient newly diagnosed with high blood pressure: which tablet first?", "start an ace inhibitor or an angiotensin receptor blocker"),
    ("Blood pressure still high on three medicines including a water tablet; what do we rule out first?", "confirm adherence"),
    ("Wheezy adult who can only say a few words at a time, breathing 28 times a minute: how severe?", "cannot complete sentences in one breath"),
    ("Asthmatic who is exhausted with a very quiet chest and blue lips.", "silent chest"),
    ("When can a patient go home after an asthma attack treated with nebulisers?", "written asthma action plan"),
]
DEMO_QUERY = QUERIES[4][0]


def make_sample_docs(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in SAMPLE_DOCS.items():
        path = folder / name
        if path.exists():
            continue
        if name.endswith(".pdf"):
            _write_pdf(path, text)
        else:
            path.write_text(text, encoding="utf-8")


def _write_pdf(path: Path, text: str):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        print(f"  (reportlab not installed; skipping sample PDF {path.name}. pip install reportlab)")
        return
    styles = getSampleStyleSheet()
    story = []
    for block in text.strip().split("\n\n"):
        block = block.strip().replace("&", "&amp;")
        first = block.splitlines()[0]
        if first.isupper():
            story.append(Paragraph(block, styles["Title"]))
        elif re.match(r"^\d+\.\s", first) and "\n" in block:
            head, body = block.split("\n", 1)
            story.append(Paragraph(head, styles["Heading2"]))
            story.append(Paragraph(body.replace("\n", " "), styles["BodyText"]))
        else:
            story.append(Paragraph(block.replace("\n", " "), styles["BodyText"]))
        story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


# ======================================================================= parsing
@dataclass
class Section:
    path: list          # heading trail, e.g. ["2. Screening Criteria"]
    text: str


@dataclass
class ParsedDoc:
    source: str
    fmt: str
    title: str
    flat_text: str      # full text, headings kept as plain lines (input for fixed-size chunking)
    sections: list = field(default_factory=list)


NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z][A-Za-z0-9 ,&/()\-:']{2,80})$")
ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 ,&/():'\-]{4,100}$")


def heading_level(line: str):
    """Return heading depth for plain-text/PDF lines, or None. ALL CAPS = 1, '2.1 Title' = 2."""
    line = line.strip()
    m = NUMBERED.match(line)
    if m:
        return m.group(1).count(".") + 1
    if ALLCAPS.match(line) and sum(c.isalpha() for c in line) >= 4:
        return 1
    return None


def parse_markdown(text: str, source: str) -> ParsedDoc:
    title, stack, sections, buf = "", [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(list(stack), body))
        buf.clear()

    flat = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        line_clean = re.sub(r"^>\s?", "", line)
        if m:
            flush()
            level, head = len(m.group(1)), m.group(2).strip()
            flat.append(head)
            if level == 1 and not title:
                title = head
                continue
            depth = level - 1 if title else level
            stack = stack[:depth - 1] + [head]
        else:
            buf.append(line_clean)
            flat.append(line_clean)
    flush()
    return ParsedDoc(source, "markdown", title or source, "\n".join(flat), sections)


def parse_plain(text: str, source: str, fmt: str) -> ParsedDoc:
    title, stack, sections, buf = "", [], [], []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(list(stack), body))
        buf.clear()

    prev_caps_heading = None      # "title" or "stack" if the previous line was an ALL-CAPS heading
    for line in text.splitlines():
        lvl = heading_level(line) if line.strip() else None
        if lvl:
            head = line.strip()
            # A heading wrapped over two lines in the PDF: merge instead of starting a new section.
            if head.isupper() and prev_caps_heading == "title":
                title = (title + " " + head.title()).strip()
                continue
            if head.isupper() and prev_caps_heading == "stack" and stack:
                stack[-1] = stack[-1] + " " + head
                continue
            flush()
            if not title and lvl == 1 and head.isupper():
                title = head.title()
                prev_caps_heading = "title"
                continue
            stack = stack[:lvl - 1] + [head]
            prev_caps_heading = "stack" if head.isupper() else None
        else:
            if line.strip():
                prev_caps_heading = None
                buf.append(line.rstrip())
            else:
                prev_caps_heading = None
                buf.append("")
    flush()
    return ParsedDoc(source, fmt, title or source, text.strip(), sections)


def parse_pdf(path: Path) -> ParsedDoc:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            raw = page.extract_text(extraction_mode="layout")   # keeps line/column layout
        except TypeError:
            raw = page.extract_text()
        pages.append(raw or "")
    raw = unicodedata.normalize("NFKC", "\n".join(pages))        # fixes ligatures like "fi"
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
    return parse_plain("\n".join(lines), path.name, "pdf")


def parse_file(path: Path) -> ParsedDoc:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if ext in (".md", ".markdown"):
        return parse_markdown(text, path.name)
    if ext == ".txt":
        return parse_plain(text, path.name, "text")
    raise ValueError(f"unsupported file type: {path.name}")


# ===================================================================== chunking
@dataclass
class Chunk:
    id: str
    text: str
    source: str
    strategy: str
    heading: str


def fixed_size_chunks(doc: ParsedDoc, size: int, overlap: int) -> list:
    """Sliding window over the whole document text, ignoring its structure."""
    words = doc.flat_text.split()
    step = max(1, size - overlap)
    chunks = []
    for i, start in enumerate(range(0, max(1, len(words)), step)):
        piece = words[start:start + size]
        if not piece:
            break
        chunks.append(Chunk(f"{doc.source}::fixed::{i}", " ".join(piece), doc.source, "fixed", ""))
        if start + size >= len(words):
            break
    return chunks


def split_text(body: str, max_words: int) -> list:
    """Pack paragraphs (then sentences, for oversize paragraphs) into pieces of <= max_words."""
    units = [u.strip() for u in re.split(r"\n\s*\n", body) if u.strip()]
    expanded = []
    for u in units:
        if len(u.split()) > max_words:
            expanded += [s for s in re.split(r"(?<=[.!?])\s+", u) if s]
        else:
            expanded.append(u)
    pieces, cur, n = [], [], 0
    for u in expanded:
        w = len(u.split())
        if cur and n + w > max_words:
            pieces.append(" ".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += w
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def header_chunks(doc: ParsedDoc, max_words: int, min_words: int) -> list:
    """One chunk per heading section; the heading trail is prepended so each chunk carries its
    own context. Oversize sections are split on paragraphs; tiny ones merge into the next."""
    chunks, pending, n = [], "", 0

    def emit(text, path):
        nonlocal n
        trail = " > ".join([doc.title] + path)
        for piece in split_text(text, max_words):
            chunks.append(Chunk(f"{doc.source}::header::{n}", f"{trail}\n{piece}",
                                doc.source, "header", " > ".join(path)))
            n += 1

    for i, sec in enumerate(doc.sections):
        last = i == len(doc.sections) - 1
        label = (" > ".join(sec.path) + "\n") if sec.path else ""
        if len(sec.text.split()) < min_words and not last:
            pending += label + sec.text + "\n"
            continue
        emit((pending + sec.text) if pending else sec.text, sec.path)
        pending = ""
    if pending:
        emit(pending, [])
    return chunks


# =================================================================== embeddings
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, cache_folder=None):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)

    def embed(self, texts, is_query=False):
        if is_query and "bge" in self.name.lower():
            texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False).tolist()


class HashingEmbedder:
    """Offline stand-in: signed bag-of-words hashing. Matches shared WORDS only, so it cannot
    bridge synonyms. Use it to test plumbing, never to judge retrieval quality."""
    name = "hashing-offline"

    def __init__(self, dims=768):
        self.dims = dims

    def embed(self, texts, is_query=False):
        out = []
        for t in texts:
            v = [0.0] * self.dims
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dims] += 1.0 if (h >> 20) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


# ================================================================== vector store
def build_collection(client, name, chunks, embedder):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 128):
        batch = chunks[i:i + 128]
        col.add(ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embedder.embed([c.text for c in batch]),
                metadatas=[{"source": c.source, "strategy": c.strategy, "heading": c.heading}
                           for c in batch])
    return col


def retrieve(col, embedder, query, k):
    res = col.query(query_embeddings=embedder.embed([query], is_query=True),
                    n_results=min(k, col.count()),
                    include=["documents", "metadatas", "distances"])
    return [(doc, meta, 1.0 - dist) for doc, meta, dist in
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


# ================================================================== evaluation
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def evaluate(col, embedder, k=5):
    ranks, ctx_words = [], []
    per_query = []
    for q, snippet in QUERIES:
        hits = retrieve(col, embedder, q, k)
        rank = next((i + 1 for i, (doc, _, _) in enumerate(hits) if norm(snippet) in norm(doc)), None)
        ranks.append(rank)
        per_query.append(rank)
        ctx_words.append(sum(len(d.split()) for d, _, _ in hits[:3]))
    n = len(QUERIES)
    hit = lambda r: sum(1 for x in ranks if x is not None and x <= r) / n
    return {
        "hit@1": round(hit(1), 3), "hit@3": round(hit(3), 3), "hit@5": round(hit(5), 3),
        "mrr@5": round(sum(1 / x for x in ranks if x) / n, 3),
        "avg_context_words_top3": round(statistics.mean(ctx_words), 1),
    }, per_query


def short(text, n=110):
    return re.sub(r"\s+", " ", text)[:n] + ("..." if len(text) > n else "")


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="p5_docs")
    ap.add_argument("--db", default="chroma_db")
    ap.add_argument("--embedder", choices=["st", "hashing"], default="st")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--fixed-size", type=int, default=90, help="words per fixed chunk")
    ap.add_argument("--fixed-overlap", type=int, default=20)
    ap.add_argument("--max-words", type=int, default=160, help="header chunk size cap")
    ap.add_argument("--min-words", type=int, default=25, help="merge sections shorter than this")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    import chromadb

    docs_dir = Path(args.docs)
    make_sample_docs(docs_dir)
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"})
    docs = [parse_file(p) for p in files]
    print(f"Parsed {len(docs)} documents: " + ", ".join(f"{d.source} ({d.fmt}, {len(d.sections)} sections)" for d in docs))

    strategies = {
        "fixed": [c for d in docs for c in fixed_size_chunks(d, args.fixed_size, args.fixed_overlap)],
        "header": [c for d in docs for c in header_chunks(d, args.max_words, args.min_words)],
    }
    embedder = (HashingEmbedder() if args.embedder == "hashing"
                else SentenceTransformerEmbedder(args.model, args.hf_cache))
    print(f"Embedder: {embedder.name}\n")

    client = chromadb.PersistentClient(path=args.db)
    tag = re.sub(r"[^a-zA-Z0-9]+", "_", embedder.name)[-30:].strip("_")
    rows, per_query, cols = [], {}, {}
    for name, chunks in strategies.items():
        col = build_collection(client, f"clinical_{name}_{tag}"[:60], chunks, embedder)
        cols[name] = col
        wc = [len(c.text.split()) for c in chunks]
        metrics, pq = evaluate(col, embedder, args.k)
        per_query[name] = pq
        rows.append({"strategy": name, "chunks": len(chunks), "avg_words": round(statistics.mean(wc), 1),
                     "max_words": max(wc), **metrics})

    cols_order = list(rows[0])
    print("=== Chunking strategy comparison ===")
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols_order}
    print("  ".join(c.ljust(widths[c]) for c in cols_order))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols_order))

    print(f"\n=== Demo scenario ===\n\"{DEMO_QUERY}\"")
    for name in strategies:
        print(f"\n[{name}] top-3 by cosine similarity")
        for rank, (doc, meta, sim) in enumerate(retrieve(cols[name], embedder, DEMO_QUERY, 3), 1):
            print(f"  {rank}. sim={sim:.3f}  {meta['source']}  {meta['heading'] or '-'}\n     {short(doc)}")

    # ---- files
    with open("p5_chunking_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_order)
        w.writeheader()
        w.writerows(rows)

    f_, h_ = rows[0], rows[1]
    verdict = ("header-based chunking scored higher" if (h_["mrr@5"], h_["hit@3"]) > (f_["mrr@5"], f_["hit@3"])
               else "fixed-size chunking scored higher" if (f_["mrr@5"], f_["hit@3"]) > (h_["mrr@5"], h_["hit@3"])
               else "the two strategies tied")
    missed = {n: [QUERIES[i][0] for i, r in enumerate(pq) if r is None] for n, pq in per_query.items()}
    lines = [
        "# P5 Chunking Strategy Comparison", "",
        f"- Embedder: `{embedder.name}`", f"- Documents: {len(docs)} ({', '.join(d.fmt for d in docs)})",
        f"- Evaluation: {len(QUERIES)} patient-scenario queries, hit if a top-k chunk contains the key fact",
        f"- Fixed: {args.fixed_size} words, {args.fixed_overlap} overlap. Header: max {args.max_words}, min {args.min_words} words", "",
        "| " + " | ".join(cols_order) + " |", "|" + "---|" * len(cols_order),
        *("| " + " | ".join(str(r[c]) for c in cols_order) + " |" for r in rows), "",
        f"**Result on this run:** {verdict}. With only {len(QUERIES)} queries, differences of one or two "
        "queries are within noise; build a larger query set before generalising.", "",
        "## Queries with no hit in top-" + str(args.k),
        *(f"- **{n}**: " + ("; ".join(q) if q else "none") for n, q in missed.items()), "",
    ]
    if args.embedder == "hashing":
        lines += ["> Produced with the offline hashing embedder, which cannot match synonyms. "
                  "These numbers test the pipeline only; re-run with a sentence-transformers model.", ""]
    Path("p5_chunking_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved p5_chunking_results.csv and p5_chunking_report.md")
    if args.embedder == "hashing":
        print("NOTE: hashing embedder is not semantic. Re-run without --embedder hashing for real results.")


if __name__ == "__main__":
    main()