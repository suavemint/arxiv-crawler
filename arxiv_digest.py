#!/usr/bin/env python3
"""
Daily ArXiv ML/AI Digest
========================
Fetches new ML/AI papers (from HuggingFace Daily Papers or raw arxiv),
downloads full PDFs, extracts text, generates paragraph-by-paragraph
technical summaries via Claude API preserving all math,
typesets a LaTeX PDF, and emails it to a Kindle address.

Configuration: set env vars or edit the CONFIG dict below.
    ANTHROPIC_API_KEY  — required for summarization
    GMAIL_ADDRESS      — sender Gmail address
    GMAIL_APP_PASSWORD — Gmail app password (not your login password)
    KINDLE_EMAIL       — destination (default: ej.orcutt_xCKr75@kindle.com)
    PAPER_SOURCE       — "huggingface" (default, curated ~10-20/day)
                         or "arxiv" (raw feed, 50-100+/day)

Dependencies (system):
    pdftotext  — from poppler-utils (apt install poppler-utils / brew install poppler)
    pdflatex   — from texlive (apt install texlive-latex-recommended / brew install basictex)
"""

import os
import sys
import json
import re
import shutil
import smtplib
import subprocess
import tempfile
import textwrap
import time
import logging
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from urllib.error import HTTPError
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# Load .env file (no external dependency)
# ---------------------------------------------------------------------------
def _load_dotenv(path: str = None):
    """Load KEY=VALUE pairs from a .env file into os.environ (if not already set)."""
    if path is None:
        path = str(Path(__file__).resolve().parent / ".env")
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                k = key.strip()
                if not os.environ.get(k):
                    os.environ[k] = val.strip()
    except FileNotFoundError:
        pass

_load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "gmail_address": os.environ.get("GMAIL_ADDRESS", ""),
    "gmail_app_password": os.environ.get("GMAIL_APP_PASSWORD", ""),
    "kindle_email": os.environ.get("KINDLE_EMAIL", "ej.orcutt_xCKr75@kindle.com"),

    # Paper source: "huggingface" (curated, ~10-20/day) or "arxiv" (raw, 50-100+/day)
    "source": os.environ.get("PAPER_SOURCE", "huggingface"),

    # arxiv-specific settings (ignored when source=huggingface)
    "categories": ["cs.LG", "cs.AI", "cs.CL", "stat.ML"],
    "max_results_query": 200,

    # General settings
    "max_papers": 40,           # hard cap (mainly relevant for arxiv source)
    "model": "claude-sonnet-4-6",
    "max_input_chars": 80000,   # truncate extracted text beyond this
    "request_delay": 1.0,       # seconds between arxiv PDF downloads (be polite)
    "log_level": "INFO",

    # Local archiving: save PDFs to this directory (set to "" to disable)
    "archive_dir": os.environ.get("DIGEST_ARCHIVE_DIR", ""),

    # Deduplication: JSON file storing previously sent paper IDs
    "history_file": os.environ.get(
        "DIGEST_HISTORY_FILE",
        str(Path(__file__).resolve().parent / "sent_paper_ids.json")),
    "history_max_age_days": 30,  # prune entries older than this

    # PDF cache: reuse previously downloaded PDFs instead of re-fetching
    "pdf_cache_dir": os.environ.get(
        "DIGEST_PDF_CACHE",
        str(Path(__file__).resolve().parent / "pdf_cache")),
    "pdf_cache_max_age_days": 7,  # auto-clean cached PDFs older than this

    # Summary cache: avoid re-summarizing on retries/failures
    "summary_cache_file": os.environ.get(
        "DIGEST_SUMMARY_CACHE",
        str(Path(__file__).resolve().parent / "summary_cache.json")),
    "summary_cache_max_age_days": 7,

    # Obsidian knowledge graph
    "obsidian_vault": os.environ.get("OBSIDIAN_VAULT", ""),
    "obsidian_papers_dir": "Machine Learning/papers",
    "obsidian_concepts_dir": "Machine Learning/concepts",

    # Zotero integration (BibTeX export + PDF copy)
    "zotero_bib_dir": os.environ.get(
        "ZOTERO_BIB_DIR",
        str(Path(__file__).resolve().parent / "zotero_import")),
}

logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"]),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source: HuggingFace Daily Papers (curated, ~10-20 papers/day)
# ---------------------------------------------------------------------------
HF_DAILY_API = "https://huggingface.co/api/daily_papers"


def fetch_huggingface_papers(date_str: str) -> list[dict]:
    """Fetch today's curated papers from the HuggingFace Daily Papers API."""
    url = f"{HF_DAILY_API}?date={date_str}"
    log.info("Fetching HuggingFace Daily Papers: %s", url)

    req = Request(url, headers={"User-Agent": "ArxivDigest/1.0"})
    with urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    papers = []
    for entry in data:
        paper = entry.get("paper", {})
        arxiv_id = paper.get("id", "")
        title = paper.get("title", "")
        abstract = paper.get("summary", "")
        authors = [a.get("name", "") for a in paper.get("authors", [])]

        if not arxiv_id or not title:
            continue

        # HF provides upvotes — useful for sorting
        upvotes = entry.get("paper", {}).get("upvotes", 0)

        papers.append({
            "id": arxiv_id,
            "title": re.sub(r"\s+", " ", title.strip()),
            "authors": authors,
            "abstract": abstract.strip(),
            "published": paper.get("publishedAt", ""),
            "primary_category": "",  # HF API doesn't provide this
            "categories": [],
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "hf_url": f"https://huggingface.co/papers/{arxiv_id}",
            "upvotes": upvotes,
        })

    # Sort by upvotes (most popular first)
    papers.sort(key=lambda p: p.get("upvotes", 0), reverse=True)
    log.info("Got %d papers from HuggingFace Daily Papers", len(papers))
    return papers


# ---------------------------------------------------------------------------
# Source: Raw ArXiv API (comprehensive, 50-100+ papers/day)
# ---------------------------------------------------------------------------
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def fetch_arxiv_papers(categories: list[str], max_results: int = 200) -> list[dict]:
    """Fetch papers submitted in the last ~36 hours from given categories."""
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    params = {
        "search_query": cat_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    }
    url = f"{ARXIV_API}?{urlencode(params)}"
    log.info("Fetching arxiv: %s", url)

    req = Request(url, headers={"User-Agent": "ArxivDigest/1.0"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()

    root = ET.fromstring(data)
    entries = root.findall(f"{ATOM_NS}entry")
    log.info("Got %d entries from arxiv API", len(entries))

    cutoff = datetime.now(timezone.utc) - timedelta(hours=36)
    papers = []
    for entry in entries:
        published_str = entry.findtext(f"{ATOM_NS}published", "")
        try:
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if published < cutoff:
            continue

        primary_cat = ""
        all_cats = []
        for cat_el in entry.findall(f"{ARXIV_NS}primary_category"):
            primary_cat = cat_el.get("term", "")
        for cat_el in entry.findall(f"{ATOM_NS}category"):
            all_cats.append(cat_el.get("term", ""))

        id_url = entry.findtext(f"{ATOM_NS}id", "")
        arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url
        authors = [a.findtext(f"{ATOM_NS}name", "")
                   for a in entry.findall(f"{ATOM_NS}author")]

        papers.append({
            "id": arxiv_id,
            "title": re.sub(r"\s+", " ", entry.findtext(f"{ATOM_NS}title", "").strip()),
            "authors": authors,
            "abstract": entry.findtext(f"{ATOM_NS}summary", "").strip(),
            "published": published_str,
            "primary_category": primary_cat,
            "categories": all_cats,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        })

    log.info("After date filter: %d papers", len(papers))
    return papers


# ---------------------------------------------------------------------------
# Unified fetch dispatcher
# ---------------------------------------------------------------------------

def fetch_papers(source: str, date_str: str) -> list[dict]:
    """Fetch papers from the configured source."""
    if source == "huggingface":
        papers = fetch_huggingface_papers(date_str)
        if not papers:
            log.warning("HuggingFace returned 0 papers; falling back to arxiv")
            papers = fetch_arxiv_papers(CONFIG["categories"],
                                        CONFIG["max_results_query"])
        return papers
    elif source == "arxiv":
        return fetch_arxiv_papers(CONFIG["categories"],
                                  CONFIG["max_results_query"])
    else:
        log.error("Unknown source: %s (use 'huggingface' or 'arxiv')", source)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Full-text extraction: download PDF → pdftotext
# ---------------------------------------------------------------------------

def download_pdf(url: str, dest: Path, delay: float = 1.0) -> bool:
    """Download a PDF from arxiv. Returns True on success."""
    try:
        req = Request(url, headers={"User-Agent": "ArxivDigest/1.0"})
        with urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        time.sleep(delay)  # rate-limit: be polite to arxiv
        return True
    except Exception as e:
        log.warning("Failed to download %s: %s", url, e)
        return False


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using pdftotext (poppler)."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            # Strip null bytes and other control chars that break API JSON
            text = result.stdout
            text = text.replace("\x00", "")
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
            return text
    except Exception as e:
        log.warning("pdftotext failed for %s: %s", pdf_path, e)
    return ""


def get_full_text(paper: dict, cache_dir: Path, delay: float = 1.0,
                  max_chars: int = 80000) -> str:
    """Download PDF (with cache) and extract full text. Falls back to abstract."""
    filename = f"{paper['id'].replace('/', '_')}.pdf"
    cached = cache_dir / filename

    if cached.exists():
        log.info("Cache hit: %s", paper["id"])
    else:
        if not download_pdf(paper["pdf_url"], cached, delay):
            log.info("Falling back to abstract for %s", paper["id"])
            return paper["abstract"]

    text = extract_text_from_pdf(cached)

    if not text or len(text) < 500:
        log.info("Extraction too short for %s, using abstract", paper["id"])
        return paper["abstract"]

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... truncated ...]"

    return text


def prune_pdf_cache(cache_dir: Path, max_age_days: int):
    """Delete cached PDFs older than max_age_days."""
    if not cache_dir.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for f in cache_dir.glob("*.pdf"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    if removed:
        log.info("Pruned %d cached PDFs older than %d days", removed, max_age_days)


# ---------------------------------------------------------------------------
# Summarization via Anthropic API (direct HTTP — no SDK dependency)
# ---------------------------------------------------------------------------

# Pricing per million tokens (USD) — update if model changes
MODEL_PRICING = {
    "claude-sonnet-4-6":  {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":    {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}

# Running totals for the current invocation
_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}


def _call_claude_api(prompt: str, system: str, api_key: str, model: str,
                     max_tokens: int, max_retries: int) -> str:
    """Direct HTTPS call to api.anthropic.com (billed per-token)."""
    import urllib.request
    import urllib.error
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read())

            usage = result.get("usage", {})
            inp = usage.get("input_tokens", 0)
            out = usage.get("output_tokens", 0)
            _usage["input_tokens"] += inp
            _usage["output_tokens"] += out
            _usage["api_calls"] += 1

            pricing = MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})
            run_cost = ((_usage["input_tokens"] * pricing["input"]
                        + _usage["output_tokens"] * pricing["output"]) / 1_000_000)
            log.info("Tokens: %d in / %d out | Running total: %d in / %d out | "
                     "Est. cost so far: $%.4f",
                     inp, out, _usage["input_tokens"], _usage["output_tokens"],
                     run_cost)

            return result["content"][0]["text"]
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                err_body = "(unreadable)"
            if e.code in (429, 529) or e.code >= 500:
                if attempt == max_retries:
                    log.error("API %d after %d retries: %s", e.code,
                              max_retries, err_body)
                    raise
                wait = int(e.headers.get("Retry-After", 2 ** attempt * 5))
                log.info("Rate limited (%d), retrying in %ds (attempt %d/%d)",
                         e.code, wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                log.error("API error %d: %s", e.code, err_body)
                raise


def _call_claude_cli(prompt: str, system: str, model: str) -> str:
    """Shell out to the Claude Code CLI (billed against Pro/Max subscription).
    Scrubs ANTHROPIC_API_KEY from the child env so Claude Code falls back to
    subscription auth established by `claude login`."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    cmd = ["claude", "-p", "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    if system:
        cmd += ["--system-prompt", system]
    cmd.append(prompt)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                            env=env, timeout=300)
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude CLI timed out after 300s")
    if r.returncode != 0:
        raise RuntimeError(
            f"claude CLI exit {r.returncode}: {r.stderr.strip()[:500]}"
        )
    _usage["api_calls"] += 1
    log.info("Claude CLI call complete (subscription; %d CLI calls this run)",
             _usage["api_calls"])
    return r.stdout.strip()


def _cli_available() -> bool:
    """True if `claude` is on PATH."""
    import shutil
    return shutil.which("claude") is not None


def call_claude(prompt: str, system: str = "", api_key: str = "",
                model: str = "claude-sonnet-4-6",
                max_tokens: int = 4096,
                max_retries: int = 5) -> str:
    """Dispatch to CLI (subscription) or API (per-token) based on env.

    Uses the CLI when USE_CLAUDE_CLI=1 and `claude` is on PATH. Otherwise
    falls back to the direct HTTPS API. Set USE_CLAUDE_CLI=0 to force API."""
    prefer_cli = os.environ.get("USE_CLAUDE_CLI", "1") != "0"
    if prefer_cli and _cli_available():
        return _call_claude_cli(prompt, system, model)
    return _call_claude_api(prompt, system, api_key, model,
                            max_tokens, max_retries)


SYSTEM_PROMPT = textwrap.dedent("""\
    You are a technical summarizer for ML/AI arxiv papers. The reader has a
    Master's in high-energy physics — assume full comfort with graduate-level
    math, statistics, and information theory.

    You will receive the FULL TEXT of an arxiv paper (extracted from PDF, so
    formatting may be imperfect). Produce a PARAGRAPH-BY-PARAGRAPH condensation
    that preserves the paper's technical substance:

    STRUCTURE — mirror the paper's own sections:
    1. One paragraph for introduction/motivation (what gap, why it matters).
    2. One or two paragraphs for the method/model — preserve the key equations,
       objective functions, architectural choices, and algorithmic steps.
       Reproduce important equations verbatim in LaTeX.
    3. One paragraph for experiments/results — concrete numbers, baselines,
       datasets, ablation findings.
    4. One short paragraph for limitations or open questions, if the paper
       discusses them.

    RULES:
    - ALL math in LaTeX with $ delimiters (single $ for inline).
    - Reproduce key equations exactly as they appear (loss functions, update
      rules, bounds, etc.). Do not simplify or paraphrase equations.
    - Use precise terminology. No filler ("in this paper", "it is worth
      noting", "interestingly").
    - Target ~400-600 words total. This should feel like reading the paper
      at 3x speed, not like reading a press release about it.
    - If the extracted text is garbled or too short (you'll know), summarize
      from whatever is available and note the limitation.

    Output ONLY the summary paragraphs. No headers, no bullet points, no
    meta-commentary about the summarization process.
""")

SYSTEM_PROMPT_ABSTRACT_FALLBACK = textwrap.dedent("""\
    You are a technical summarizer for ML/AI arxiv papers. The reader has a
    Master's in high-energy physics.

    You only have the ABSTRACT (full text was unavailable). Produce a concise
    technical summary (2-3 paragraphs, ~150-200 words) preserving all math
    in LaTeX $ delimiters. Focus on: what problem, what method, what result.
    No filler. Output ONLY the summary.
""")


def summarize_paper(paper: dict, full_text: str, api_key: str, model: str) -> str:
    """Generate a paragraph-by-paragraph technical summary."""
    is_full = len(full_text) > 1000 and full_text != paper["abstract"]

    if is_full:
        system = SYSTEM_PROMPT
        prompt = f"Title: {paper['title']}\n\nFull paper text:\n\n{full_text}"
        max_tok = 4096
    else:
        system = SYSTEM_PROMPT_ABSTRACT_FALLBACK
        prompt = f"Title: {paper['title']}\n\nAbstract:\n{paper['abstract']}"
        max_tok = 2048

    try:
        summary = call_claude(prompt, system=system, api_key=api_key,
                              model=model, max_tokens=max_tok)
        return ("claude", summary)
    except Exception as e:
        log.warning("Summarization failed for %s: %s", paper["id"], e)
        return ("raw", paper["abstract"])


# ---------------------------------------------------------------------------
# LaTeX PDF generation
# ---------------------------------------------------------------------------

def escape_latex(text: str) -> str:
    """Escape ALL special LaTeX chars in plain text (titles, authors, etc.).
    NOT for use on Claude-generated summaries, which are already valid LaTeX."""
    for ch, repl in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(ch, repl)
    return text


def build_latex(papers: list[dict], date_str: str, source: str) -> str:
    """Build a complete LaTeX document from summarized papers."""
    n_full = sum(1 for p in papers if p.get("_full_text", False))
    n_abstract = len(papers) - n_full
    source_label = "HuggingFace Daily Papers" if source == "huggingface" else "arxiv"

    header = textwrap.dedent(r"""
        \documentclass[10pt,a4paper]{article}
        \usepackage[margin=0.9in]{geometry}
        \usepackage{amsmath,amssymb,amsfonts}
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \usepackage{hyperref}
        \usepackage{parskip}
        \usepackage{titlesec}
        \usepackage{fancyhdr}
        \hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
        \titleformat{\subsection}{\normalfont\large\bfseries}{}{0em}{}
        \pagestyle{fancy}
        \fancyhf{}
        \fancyhead[L]{\small ArXiv ML/AI Digest}
        \fancyhead[R]{\small """ + escape_latex(date_str) + r"""}
        \fancyfoot[C]{\thepage}
        \begin{document}
        \begin{center}
        {\LARGE\bfseries ArXiv ML/AI Digest}\\[4pt]
        {\large """ + escape_latex(date_str) + r"""}\\[2pt]
        {\small Source: """ + escape_latex(source_label) + r"""
        \quad---\quad """ + str(len(papers)) + r""" papers}
    """)

    summary_line = str(n_full) + r" full-text summaries"
    if n_abstract > 0:
        summary_line += ", " + str(n_abstract) + r" abstract-only"
    header += r"\\[2pt]{\footnotesize " + summary_line + "}"

    header += textwrap.dedent(r"""
        \end{center}
        \vspace{0.5em}
        \hrule
        \vspace{1em}
    """)

    body_parts = []
    for i, p in enumerate(papers):
        title_esc = escape_latex(p["title"])
        authors_short = ", ".join(p["authors"][:4])
        if len(p["authors"]) > 4:
            authors_short += " et al."
        authors_esc = escape_latex(authors_short)
        url = p["url"]
        pdf_url = p["pdf_url"]

        # Claude-generated summaries are valid LaTeX; raw fallbacks need escaping.
        summary = p.get("summary", "")
        if not summary or not p.get("_summary_is_latex", False):
            summary = escape_latex(summary or p["abstract"])

        # Right-aligned metadata: category if available, upvotes if from HF
        meta_parts = []
        if p.get("primary_category"):
            meta_parts.append(r"\texttt{" + escape_latex(p["primary_category"]) + "}")
        if p.get("upvotes", 0) > 0:
            meta_parts.append(r"$\uparrow$" + str(p["upvotes"]))
        meta_right = " ".join(meta_parts) if meta_parts else ""

        source_tag = ""
        if not p.get("_full_text", False):
            source_tag = r" {\scriptsize [abstract only]}"

        links = r"\href{" + url + "}{" + escape_latex(p["id"]) + "}"
        links += r" \quad \href{" + pdf_url + "}{[pdf]}"
        if p.get("hf_url"):
            links += r" \quad \href{" + p["hf_url"] + "}{[hf]}"

        body_parts.append(textwrap.dedent(rf"""
            \subsection*{{{i+1}.\ {title_esc}{source_tag}}}
            \textit{{{authors_esc}}} \hfill {meta_right}\\
            {links}
            \smallskip

            {summary}

            \medskip
            \hrule
            \medskip
        """))

    footer = r"\end{document}"
    return header + "\n".join(body_parts) + footer


def compile_pdf(latex_src: str, output_path: Path) -> Path:
    """Compile LaTeX source to PDF. Returns path to PDF."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = Path(tmpdir) / "digest.tex"
        tex_file.write_text(latex_src, encoding="utf-8")

        for run in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir,
                 str(tex_file)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                log.warning("pdflatex pass %d exited with code %d",
                            run + 1, result.returncode)

        pdf_src = Path(tmpdir) / "digest.pdf"
        if not pdf_src.exists():
            log.error("pdflatex output (last 2000 chars):\n%s",
                      result.stdout[-2000:])
            raise RuntimeError("PDF not produced by pdflatex")

        shutil.copy2(pdf_src, output_path)
        log.info("PDF written to %s (%d bytes)", output_path,
                 output_path.stat().st_size)
        return output_path


# ---------------------------------------------------------------------------
# Concept extraction via Claude (for Obsidian knowledge graph)
# ---------------------------------------------------------------------------

CONCEPT_EXTRACTION_PROMPT = textwrap.dedent("""\
    Given this ML/AI paper summary, extract a structured list of DURABLE,
    REUSABLE concepts — things that show up across many papers, not
    paper-specific named entities. Return ONLY valid JSON:
    {
      "concepts": [
        {
          "name": "lowercase-hyphenated-slug",
          "display": "Human Readable Name",
          "type": "method|architecture|task|theory|domain",
          "brief": "One sentence defining or contextualizing this concept."
        }
      ]
    }

    INCLUDE:
    - General methods and techniques (e.g. "score-matching", "flow-matching",
      "variational-inference", "contrastive-learning", "lora", "rlhf").
    - Established architectures (e.g. "transformer", "diffusion-model", "vit",
      "mamba", "mixture-of-experts").
    - Task categories (e.g. "monocular-3d-detection", "visual-question-answering",
      "code-generation", "preference-learning").
    - Theoretical tools (e.g. "optimal-transport", "reparameterization-trick",
      "evidence-lower-bound").
    - WELL-KNOWN datasets and benchmarks that predate this paper (e.g.
      "imagenet", "mmlu", "gsm8k", "coco").

    EXCLUDE:
    - Datasets, benchmarks, or evaluation suites INTRODUCED by this paper
      (anything named "<Name>-Bench", "<Name>-Eval", "<Name>-30K", "<Name> Dataset",
      etc. that the paper itself proposes).
    - Paper-specific method names coined by this paper's authors (e.g. model
      names like "MyNet", "FORGE", "WildDet3D" — these belong in the paper
      metadata, not as first-class concepts).
    - Acronyms without their expansion — always use the canonical expanded form.

    Aim for 3-6 concepts per paper. Fewer is better than noise. If the paper's
    primary contribution IS a broadly-applicable new method that deserves its
    own concept node, include it — but only if the method is described in
    general enough terms to apply beyond this single paper.
""")


def extract_concepts(paper: dict, api_key: str, model: str) -> list[dict]:
    """Extract structured concepts from a paper's summary."""
    summary = paper.get("summary", paper["abstract"])
    prompt = (f"Title: {paper['title']}\n"
              f"Authors: {', '.join(paper['authors'][:5])}\n\n"
              f"Summary:\n{summary}")
    try:
        raw = call_claude(prompt, system=CONCEPT_EXTRACTION_PROMPT,
                          api_key=api_key, model=model, max_tokens=1024)
        # Strip markdown fencing if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        data = json.loads(raw)
        concepts = data.get("concepts", [])
        # Second-pass heuristic filter: drop paper-specific named entities
        # that slipped through the prompt (e.g. "WildDet3D-Data Dataset").
        kept, dropped = [], []
        for c in concepts:
            if is_paper_specific_concept(c):
                dropped.append(c.get("display", c.get("name", "?")))
            else:
                kept.append(c)
        if dropped:
            log.info("Filtered %d paper-specific concept(s) for %s: %s",
                     len(dropped), paper["id"], ", ".join(dropped))
        log.info("Extracted %d concepts for %s (kept %d)",
                 len(concepts), paper["id"], len(kept))
        return kept
    except Exception as e:
        log.warning("Concept extraction failed for %s: %s", paper["id"], e)
        return []


# ---------------------------------------------------------------------------
# Citekey generation
# ---------------------------------------------------------------------------

def make_citekey(paper: dict) -> str:
    """Generate a stable citekey: firstAuthorLastName_year_firstTitleWord."""
    # First author's last name
    first_author = paper["authors"][0] if paper["authors"] else "unknown"
    last_name = first_author.split()[-1].lower() if first_author else "unknown"
    last_name = re.sub(r"[^a-z]", "", last_name)

    # Year from published date or arxiv ID
    year = ""
    if paper.get("published"):
        m = re.search(r"(\d{4})", paper["published"])
        if m:
            year = m.group(1)
    if not year:
        # arxiv IDs like 2604.08626 → 2026
        m = re.match(r"(\d{2})(\d{2})\.", paper["id"])
        if m:
            year = f"20{m.group(1)}"

    # First significant title word (skip articles, prepositions)
    skip = {"a", "an", "the", "on", "in", "of", "for", "to", "and", "with",
            "by", "from", "is", "are", "via", "towards", "toward"}
    words = re.findall(r"[a-z]+", paper["title"].lower())
    title_word = next((w for w in words if w not in skip), "paper")

    return f"{last_name}{year}{title_word.capitalize()}"


# ---------------------------------------------------------------------------
# Obsidian knowledge graph: paper notes + concept notes
# ---------------------------------------------------------------------------

def write_obsidian_paper_note(paper: dict, concepts: list[dict],
                              citekey: str, vault: Path):
    """Write a paper note to the Obsidian vault."""
    papers_dir = vault / CONFIG["obsidian_papers_dir"]
    papers_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_title = re.sub(r'[<>:"/\\|?*]', '', paper["title"])[:80].strip()
    note_path = papers_dir / f"{safe_title}.md"

    # Build concept links and tags (use sanitized name for link target)
    _san = re.compile(r'[<>:"/\\|?*]')
    concept_links = " | ".join(
        f"[[{_san.sub('-', c['display']).strip()}|{c['display']}]]"
        for c in concepts
    )
    tags = ", ".join(c["name"] for c in concepts)
    concept_types = {c["name"]: c.get("type", "") for c in concepts}

    # Summary: strip LaTeX for Obsidian (convert $ to inline code)
    summary = paper.get("summary", paper["abstract"])

    authors_str = ", ".join(paper["authors"][:6])
    if len(paper["authors"]) > 6:
        authors_str += " et al."

    content = f"""---
arxiv_id: "{paper['id']}"
citekey: "{citekey}"
date: {datetime.now().strftime('%Y-%m-%d')}
authors: [{authors_str}]
tags: [{tags}]
url: {paper['url']}
pdf: {paper['pdf_url']}
---

# {paper['title']}

**Authors:** {authors_str}
**arXiv:** [{paper['id']}]({paper['url']}) | [PDF]({paper['pdf_url']})"""

    if paper.get("hf_url"):
        content += f" | [HF Papers]({paper['hf_url']})"

    if paper.get("upvotes", 0) > 0:
        content += f" | {paper['upvotes']} upvotes"

    content += f"""
**Zotero:** [[@{citekey}]]

## Concepts
{concept_links}

## Summary
{summary}

## Abstract
{paper['abstract']}
"""

    note_path.write_text(content, encoding="utf-8")
    log.info("Wrote Obsidian paper note: %s", note_path.name)
    return note_path


def write_obsidian_concept_note(concept: dict, paper: dict,
                                citekey: str, vault: Path,
                                api_key: str = "", model: str = ""):
    """Create or append to a concept note in the Obsidian vault."""
    concepts_dir = vault / CONFIG["obsidian_concepts_dir"]
    concepts_dir.mkdir(parents=True, exist_ok=True)

    safe_display = re.sub(r'[<>:"/\\|?*]', '-', concept['display']).strip()
    note_path = concepts_dir / f"{safe_display}.md"
    safe_title = re.sub(r'[<>:"/\\|?*]', '', paper["title"])[:80].strip()

    new_entry = (f"- [[{safe_title}|{paper['title'][:60]}]] "
                 f"({', '.join(paper['authors'][:2])}"
                 f"{' et al.' if len(paper['authors']) > 2 else ''}, "
                 f"{datetime.now().strftime('%Y-%m-%d')})")

    if note_path.exists():
        existing = note_path.read_text(encoding="utf-8")
        # Avoid duplicate entries
        if paper["id"] in existing or safe_title in existing:
            return
        # Append under Papers section
        if "## Papers" in existing:
            existing = existing.replace("## Papers\n",
                                        f"## Papers\n{new_entry}\n", 1)
        else:
            existing += f"\n## Papers\n{new_entry}\n"
        note_path.write_text(existing, encoding="utf-8")
    else:
        # Ask Claude for a short encyclopedic background paragraph.
        bg = None
        if api_key and model:
            bg = fetch_concept_background(concept["display"],
                                           concept.get("type", ""),
                                           api_key, model)
        wiki_section = ""
        if bg and bg.get("extract"):
            wiki_section = (f"\n## Background\n"
                            f"*Generated via {bg['source']}:*\n\n"
                            f"{bg['extract']}\n")
            log.info("Background written for '%s'", concept['display'])
        else:
            log.info("No background for '%s' (UNKNOWN or skipped)",
                     concept['display'])

        content = f"""---
concept: "{concept['name']}"
type: {concept.get('type', 'unknown')}
---

# {concept['display']}

{concept.get('brief', '')}
{wiki_section}
## Papers
{new_entry}
"""
        note_path.write_text(content, encoding="utf-8")
        log.info("Created concept note: %s", concept['display'])


# ---------------------------------------------------------------------------
# Paper-specific concept filter (heuristic second pass after Claude extraction)
# ---------------------------------------------------------------------------

# Patterns that strongly indicate a paper-specific named entity rather than a
# durable reusable concept. Applied to the `display` name.
_PAPER_SPECIFIC_PATTERNS = [
    re.compile(r"-Bench\b", re.IGNORECASE),
    re.compile(r"-Eval\b", re.IGNORECASE),
    re.compile(r"-\d+[KMB]\b"),                  # e.g. -30K, -10M
    re.compile(r"\b(?:Dataset|Benchmark|Eval|Suite|Corpus)\s*$", re.IGNORECASE),
    re.compile(r"-Data\b", re.IGNORECASE),
    re.compile(r"-v\d+\b", re.IGNORECASE),       # version-tagged artifact
]


def is_paper_specific_concept(concept: dict) -> bool:
    """True if a concept looks like a paper-specific named entity that should
    not become a first-class node in the knowledge graph."""
    name = concept.get("display", "") or concept.get("name", "")
    for pat in _PAPER_SPECIFIC_PATTERNS:
        if pat.search(name):
            return True
    return False


# ---------------------------------------------------------------------------
# Concept background via Claude
#
# PwC's REST API was retired when HuggingFace absorbed paperswithcode.com — the
# domain now serves only their Trending Papers UI. Wikipedia false-positives
# on ML terms. So we use Claude itself, with an UNKNOWN escape hatch so it
# declines rather than hallucinates for obscure or paper-specific names.
# ---------------------------------------------------------------------------

_bg_cache = {}  # in-memory cache for the current run

_BACKGROUND_SYSTEM = textwrap.dedent("""\
    You are writing encyclopedic background for an ML/AI concept note.
    Given a concept name and optional type hint, write 3–4 sentences that
    define the concept, sketch its mathematical or mechanistic basis where
    relevant, and note its typical use.

    Rules:
    - Write in the register of a graduate-level textbook, not a blog post.
    - Preserve mathematical notation inline with $...$ (LaTeX).
    - Do NOT list papers, authors, or dates.
    - If the concept is obscure, paper-specific (e.g. a single-paper dataset
      or method name), ambiguous, or something you cannot characterize with
      high confidence, respond with exactly: UNKNOWN
    - Output only the paragraph or the literal string UNKNOWN. No preamble.
""")


def fetch_concept_background(display: str, concept_type: str,
                              api_key: str, model: str):
    """Ask Claude for a short encyclopedic background paragraph on a concept.
    Returns {title, extract, source} or None if Claude returns UNKNOWN."""
    key = (display, concept_type)
    if key in _bg_cache:
        return _bg_cache[key]

    prompt = f"Concept: {display}"
    if concept_type:
        prompt += f"\nType hint: {concept_type}"
    try:
        raw = call_claude(prompt, system=_BACKGROUND_SYSTEM,
                          api_key=api_key, model=model, max_tokens=300)
    except Exception as e:
        log.warning("Background fetch failed for %s: %s", display, e)
        _bg_cache[key] = None
        return None

    text = raw.strip()
    if text == "UNKNOWN" or text.startswith("UNKNOWN"):
        _bg_cache[key] = None
        return None
    result = {
        "title": display,
        "extract": text,
        "source": f"Claude {model}",
    }
    _bg_cache[key] = result
    return result


# Back-compat aliases for older backfill scripts.
def fetch_paperswithcode_summary(query: str, concept_type: str = ""):
    return None  # PwC API retired; use fetch_concept_background instead

fetch_wikipedia_summary = fetch_paperswithcode_summary


# ---------------------------------------------------------------------------
# Zotero integration: BibTeX export + PDF staging
# ---------------------------------------------------------------------------

def export_bibtex_entry(paper: dict, citekey: str) -> str:
    """Generate a BibTeX @article entry for a paper."""
    authors_bib = " and ".join(paper["authors"][:20])
    # Escape BibTeX special chars in title
    title = paper["title"].replace("{", "").replace("}", "")

    year = ""
    m = re.search(r"(\d{4})", paper.get("published", ""))
    if m:
        year = m.group(1)
    elif re.match(r"(\d{2})\d{2}\.", paper["id"]):
        year = f"20{paper['id'][:2]}"

    return textwrap.dedent(f"""\
        @article{{{citekey},
          title     = {{{{{title}}}}},
          author    = {{{authors_bib}}},
          year      = {{{year}}},
          eprint    = {{{paper['id']}}},
          archivePrefix = {{arXiv}},
          url       = {{{paper['url']}}},
          abstract  = {{{paper['abstract'][:500]}}}
        }}
    """)


def stage_for_zotero(papers: list[dict], citekeys: dict,
                     cache_dir: Path, bib_dir: Path):
    """Write a combined .bib file and copy/link cached PDFs for Zotero import."""
    bib_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    bib_path = bib_dir / f"arxiv_digest_{date_str}.bib"

    entries = []
    for p in papers:
        ck = citekeys[p["id"]]
        entries.append(export_bibtex_entry(p, ck))

        # Copy cached PDF with citekey name for easy Zotero matching
        src = cache_dir / f"{p['id'].replace('/', '_')}.pdf"
        dest = bib_dir / f"{ck}.pdf"
        if src.exists() and not dest.exists():
            shutil.copy2(src, dest)

    bib_path.write_text("\n".join(entries), encoding="utf-8")
    log.info("Wrote %d BibTeX entries to %s", len(entries), bib_path)
    return bib_path


# ---------------------------------------------------------------------------
# Email via Gmail SMTP
# ---------------------------------------------------------------------------

def send_email(pdf_path: Path, recipient: str, sender: str,
               app_password: str, date_str: str, paper_count: int,
               source: str):
    """Send the PDF as an email attachment via Gmail SMTP."""
    source_label = "HuggingFace Daily Papers" if source == "huggingface" else "arxiv"
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = f"ArXiv ML/AI Digest — {date_str} ({paper_count} papers)"

    body = (f"Daily arxiv digest for {date_str}.\n"
            f"{paper_count} papers via {source_label}.\n"
            "PDF attached with full-text technical summaries.")
    msg.attach(MIMEText(body, "plain"))

    with open(pdf_path, "rb") as f:
        part = MIMEBase("application", "pdf")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename=arxiv_digest_{date_str}.pdf")
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(sender, app_password)
        server.send_message(msg)
    log.info("Email sent to %s", recipient)


# ---------------------------------------------------------------------------
# Deduplication: persistent history of sent paper IDs
# ---------------------------------------------------------------------------

def load_history(path: str) -> dict:
    """Load {arxiv_id: iso_date_str} from JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_history(path: str, history: dict):
    """Atomically write history dict to JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f)
    os.replace(tmp, path)


def prune_history(history: dict, max_age_days: int) -> dict:
    """Drop entries older than max_age_days to keep the file small.
    Values can be date strings or dicts with a 'date' key."""
    cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")

    def _date(v):
        if isinstance(v, dict):
            return v.get("date", "")
        return v

    return {k: v for k, v in history.items() if _date(v) >= cutoff}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    api_key = CONFIG["anthropic_api_key"]
    gmail = CONFIG["gmail_address"]
    gmail_pw = CONFIG["gmail_app_password"]
    kindle = CONFIG["kindle_email"]
    source = CONFIG["source"]

    missing = []
    if not api_key:
        missing.append("ANTHROPIC_API_KEY")
    if not gmail:
        missing.append("GMAIL_ADDRESS")
    if not gmail_pw:
        missing.append("GMAIL_APP_PASSWORD")
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    date_str = datetime.now().strftime("%Y-%m-%d")
    log.info("=== ArXiv Digest for %s (source=%s) ===", date_str, source)

    # 1. Fetch paper metadata
    papers = fetch_papers(source, date_str)
    if not papers:
        log.info("No new papers found. Nothing to send.")
        return

    # Dedup against previously sent papers
    history_path = CONFIG["history_file"]
    history = prune_history(load_history(history_path),
                            CONFIG["history_max_age_days"])
    before = len(papers)
    papers = [p for p in papers if p["id"] not in history]
    if before != len(papers):
        log.info("Dedup: %d → %d papers (skipped %d already sent)",
                 before, len(papers), before - len(papers))
    if not papers:
        log.info("All papers already sent in previous digests. Nothing to do.")
        return

    if len(papers) > CONFIG["max_papers"]:
        log.info("Capping from %d to %d papers", len(papers), CONFIG["max_papers"])
        papers = papers[:CONFIG["max_papers"]]

    # 2. Download full PDFs and extract text (with persistent cache)
    cache_dir = Path(CONFIG["pdf_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    prune_pdf_cache(cache_dir, CONFIG["pdf_cache_max_age_days"])

    for i, p in enumerate(papers):
        log.info("Processing %d/%d: %s", i + 1, len(papers), p["id"])
        full_text = get_full_text(
            p, cache_dir,
            delay=CONFIG["request_delay"],
            max_chars=CONFIG["max_input_chars"],
        )
        p["_full_text_content"] = full_text
        p["_full_text"] = (len(full_text) > 1000
                           and full_text != p["abstract"])

    # 3. Summarize via Claude API (with persistent cache)
    scache_path = CONFIG["summary_cache_file"]
    scache = prune_history(load_history(scache_path),
                           CONFIG["summary_cache_max_age_days"])
    # scache format: {arxiv_id: {"summary": ..., "source": "claude"|"raw", "date": ...}}

    for i, p in enumerate(papers):
        cached_entry = scache.get(p["id"])
        if cached_entry and isinstance(cached_entry, dict) and cached_entry.get("summary"):
            log.info("Summary cache hit %d/%d: %s", i + 1, len(papers), p["id"])
            p["summary"] = cached_entry["summary"]
            p["_summary_is_latex"] = (cached_entry.get("source") == "claude")
        else:
            log.info("Summarizing %d/%d: %s (full=%s)",
                     i + 1, len(papers), p["id"], p["_full_text"])
            source_type, text = summarize_paper(
                p, p["_full_text_content"], api_key, CONFIG["model"])
            p["summary"] = text
            p["_summary_is_latex"] = (source_type == "claude")
            scache[p["id"]] = {
                "summary": text,
                "source": source_type,
                "date": date_str,
            }
            # Save after each summarization so progress survives crashes
            save_history(scache_path, scache)
        del p["_full_text_content"]  # free memory

    # 4. Knowledge graph: extract concepts → Obsidian notes → Zotero BibTeX
    vault_path = CONFIG["obsidian_vault"]
    if vault_path:
        vault = Path(vault_path)
        if not vault.exists():
            log.warning("Obsidian vault not found at %s — skipping notes", vault_path)
        else:
            # Generate citekeys
            citekeys = {p["id"]: make_citekey(p) for p in papers}

            for i, p in enumerate(papers):
                log.info("Extracting concepts %d/%d: %s",
                         i + 1, len(papers), p["id"])
                concepts = extract_concepts(p, api_key, CONFIG["model"])
                p["_concepts"] = concepts
                ck = citekeys[p["id"]]

                write_obsidian_paper_note(p, concepts, ck, vault)
                for c in concepts:
                    write_obsidian_concept_note(c, p, ck, vault,
                                                 api_key=api_key,
                                                 model=CONFIG["model"])

            # Stage BibTeX + PDFs for Zotero import
            bib_dir = Path(CONFIG["zotero_bib_dir"])
            stage_for_zotero(papers, citekeys, cache_dir, bib_dir)
            log.info("Zotero import staged in %s", bib_dir)
    else:
        log.info("OBSIDIAN_VAULT not set — skipping knowledge graph")

    # 5. Generate PDF
    latex_src = build_latex(papers, date_str, source)
    script_dir = Path(__file__).resolve().parent
    pdf_path = script_dir / f"arxiv_digest_{date_str}.pdf"
    compile_pdf(latex_src, pdf_path)

    # 6. Email
    send_email(pdf_path, kindle, gmail, gmail_pw, date_str, len(papers), source)

    # Record sent paper IDs for dedup on future runs
    for p in papers:
        history[p["id"]] = date_str
    save_history(history_path, history)
    log.info("Updated dedup history (%d total entries)", len(history))

    n_full = sum(1 for p in papers if p.get("_full_text", False))
    pricing = MODEL_PRICING.get(CONFIG["model"], {"input": 3.0, "output": 15.0})
    total_cost = ((_usage["input_tokens"] * pricing["input"]
                  + _usage["output_tokens"] * pricing["output"]) / 1_000_000)
    log.info("Done. %d papers (%d full-text, %d abstract-only). "
             "API usage: %d calls, %d input + %d output tokens, ~$%.4f",
             len(papers), n_full, len(papers) - n_full,
             _usage["api_calls"], _usage["input_tokens"],
             _usage["output_tokens"], total_cost)

    # Archive the PDF (keep in script dir by default, copy to archive if set)
    archive_dir = CONFIG["archive_dir"]
    if archive_dir:
        archive_path = Path(archive_dir)
        archive_path.mkdir(parents=True, exist_ok=True)
        dest = archive_path / pdf_path.name
        shutil.copy2(pdf_path, dest)
        log.info("PDF archived to %s", dest)
    log.info("Digest PDF: %s", pdf_path)


if __name__ == "__main__":
    main()
